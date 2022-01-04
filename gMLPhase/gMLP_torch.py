from random import randrange
import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
import pytorch_lightning as pl
import numpy as np

from gMLPhase.EqT_utils import picker

# functions

def exists(val):
    return val is not None

def pair(val):
    return (val, val) if not isinstance(val, tuple) else val

def dropout_layers(layers, prob_survival):
    if prob_survival == 1:
        return layers

    num_layers = len(layers)
    to_drop = torch.zeros(num_layers).uniform_(0., 1.) > prob_survival

    # make sure at least one layer makes it
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers

def shift(t, amount, mask = None):
    if amount == 0:
        return t
    return F.pad(t, (0, 0, amount, -amount), value = 0.)

# helper classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class PreShiftTokens(nn.Module):
    def __init__(self, shifts, fn):
        super().__init__()
        self.fn = fn
        self.shifts = tuple(shifts)

    def forward(self, x, **kwargs):
        if self.shifts == (0,):
            return self.fn(x, **kwargs)

        shifts = self.shifts
        segments = len(shifts)
        feats_per_shift = x.shape[-1] // segments
        splitted = x.split(feats_per_shift, dim = -1)
        segments_to_shift, rest = splitted[:segments], splitted[segments:]
        segments_to_shift = list(map(lambda args: shift(*args), zip(segments_to_shift, shifts)))
        x = torch.cat((*segments_to_shift, *rest), dim = -1)
        return self.fn(x, **kwargs)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class Attention(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inner, causal = False):
        super().__init__()
        self.scale = dim_inner ** -0.5
        self.causal = causal

        self.to_qkv = nn.Linear(dim_in, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim_out)

    def forward(self, x):
        device = x.device
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if self.causal:
            mask = torch.ones(sim.shape[-2:], device = device).triu(1).bool()
            sim.masked_fill_(mask[None, ...], -torch.finfo(q.dtype).max)

        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        return self.to_out(out)

class SpatialGatingUnit(nn.Module):
    def __init__(
        self,
        dim,
        dim_seq,
        causal = False,
        act = nn.Identity(),
        heads = 1,
        init_eps = 1e-3,
        circulant_matrix = False
    ):
        super().__init__()
        dim_out = dim // 2
        self.heads = heads
        self.causal = causal
        self.norm = nn.LayerNorm(dim_out)

        self.act = act

        # parameters

        if circulant_matrix:
            self.circulant_pos_x = nn.Parameter(torch.ones(heads, dim_seq))
            self.circulant_pos_y = nn.Parameter(torch.ones(heads, dim_seq))

        self.circulant_matrix = circulant_matrix
        shape = (heads, dim_seq,) if circulant_matrix else (heads, dim_seq, dim_seq)
        weight = torch.zeros(shape)

        self.weight = nn.Parameter(weight)
        init_eps /= dim_seq
        nn.init.uniform_(self.weight, -init_eps, init_eps)

        self.bias = nn.Parameter(torch.ones(heads, dim_seq))

    def forward(self, x, gate_res = None):
        device, n, h = x.device, x.shape[1], self.heads

        res, gate = x.chunk(2, dim = -1)
        gate = self.norm(gate)

        weight, bias = self.weight, self.bias

        if self.circulant_matrix:
            # build the circulant matrix

            dim_seq = weight.shape[-1]
            weight = F.pad(weight, (0, dim_seq), value = 0)
            weight = repeat(weight, '... n -> ... (r n)', r = dim_seq)
            weight = weight[:, :-dim_seq].reshape(h, dim_seq, 2 * dim_seq - 1)
            weight = weight[:, :, (dim_seq - 1):]

            # give circulant matrix absolute position awareness

            pos_x, pos_y = self.circulant_pos_x, self.circulant_pos_y
            weight = weight * rearrange(pos_x, 'h i -> h i ()') * rearrange(pos_y, 'h j -> h () j')

        if self.causal:
            weight, bias = weight[:, :n, :n], bias[:, :n]
            mask = torch.ones(weight.shape[-2:], device = device).triu_(1).bool()
            mask = rearrange(mask, 'i j -> () i j')
            weight = weight.masked_fill(mask, 0.)

        gate = rearrange(gate, 'b n (h d) -> b h n d', h = h)

        gate = einsum('b h n d, h m n -> b h m d', gate, weight)
        gate = gate + rearrange(bias, 'h n -> () h n ()')

        gate = rearrange(gate, 'b h n d -> b n (h d)')

        if exists(gate_res):
            gate = gate + gate_res

        return self.act(gate) * res

class gMLPBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_ff,
        seq_len,
        heads = 1,
        attn_dim = None,
        causal = False,
        act = nn.Identity(),
        circulant_matrix = False
    ):
        super().__init__()
        self.proj_in = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.GELU()
        )

        self.attn = Attention(dim, dim_ff // 2, attn_dim, causal) if exists(attn_dim) else None

        self.sgu = SpatialGatingUnit(dim_ff, seq_len, causal, act, heads, circulant_matrix = circulant_matrix)
        self.proj_out = nn.Linear(dim_ff // 2, dim)

    def forward(self, x):
        gate_res = self.attn(x) if exists(self.attn) else None
        x = self.proj_in(x)
        x = self.sgu(x, gate_res = gate_res)
        x = self.proj_out(x)
        return x


def conv_block(n_in, n_out, k, stride ,padding, activation, dropout=0):
    if activation:
        return nn.Sequential(
            nn.Conv1d(n_in, n_out, k, stride=stride, padding=padding),
            activation,
            nn.Dropout(p=dropout),

        )
    else:
        return nn.Conv1d(n_in, n_out, k, stride=stride, padding=padding)

def deconv_block(n_in, n_out, k, stride,padding, output_padding, activation, dropout=0):
    if activation:
        return nn.Sequential(
            nn.ConvTranspose1d(n_in, n_out, k, stride=stride, padding=padding, output_padding=output_padding),
            activation,
            nn.Dropout(p=dropout),

        )
    else:                
        return nn.ConvTranspose1d(n_in, n_out, k, stride=stride, padding=padding, output_padding=output_padding
        )
    
def encoder(activation, dropout):

    return nn.Sequential(
        conv_block(3, 8, 3, 1, 1, activation, dropout = dropout),
        nn.BatchNorm1d(8),
        conv_block(8, 8, 3, 2, 1, activation),
        conv_block(8, 8, 3, 1, 1, activation, dropout = dropout),
        nn.BatchNorm1d(8),
        conv_block(8, 16, 3, 2, 1, activation),
        conv_block(16, 16, 3, 1, 1, activation, dropout = dropout),
        nn.BatchNorm1d(16),
        conv_block(16, 16, 3, 2, 1, activation),
        conv_block(16, 16, 3, 1, 1, activation, dropout = dropout),
        nn.BatchNorm1d(16),
        conv_block(16, 32, 3, 2, 1, activation),
        conv_block(32, 32, 3, 1, 1, activation, dropout = dropout),
        nn.BatchNorm1d(32),
        conv_block(32, 32, 3, 2, 1, activation),
        conv_block(32, 32, 3, 1, 1, activation, dropout = dropout),
        nn.BatchNorm1d(32),    
        conv_block(32, 64, 3, 2, 1, activation),
        conv_block(64, 64, 3, 1, 1, activation, dropout = dropout),
        nn.BatchNorm1d(64),
        conv_block(64, 64, 3, 2, 1, activation),
        conv_block(64, 64, 3, 1, 1, activation, dropout = dropout),
        nn.BatchNorm1d(64)
        )
    
    
def decoder(activation, dropout):

    return nn.Sequential(
        deconv_block(64, 64, 3, 2, padding = 1, output_padding=1, activation=activation),
        nn.BatchNorm1d(64),
        conv_block(64,64,3,1,1,activation, dropout = dropout),
        deconv_block(64, 32, 3, 2, padding = 1, output_padding=1, activation=activation),
        nn.BatchNorm1d(32),
        conv_block(32,32,3,1,1,activation, dropout = dropout),
        deconv_block(32, 32, 3, 2, padding = 1, output_padding=0, activation=activation),
        nn.BatchNorm1d(32),
        conv_block(32,32,3,1,1,activation, dropout = dropout),
        deconv_block(32, 16, 3, 2, padding = 1, output_padding=1, activation=activation),
        nn.BatchNorm1d(16),
        conv_block(16,16,3,1,1,activation, dropout = dropout),
        deconv_block(16, 16, 3, 2, padding = 1, output_padding=1, activation=activation),
        nn.BatchNorm1d(16),
        conv_block(16,16,3,1,1,activation, dropout = dropout),
        deconv_block(16, 8, 3, 2, padding = 1, output_padding=1, activation=activation),
        nn.BatchNorm1d(8),
        conv_block(8,8,3,1,1,activation, dropout = dropout),
        deconv_block(8, 8, 3, 2, padding = 1, output_padding=1, activation=activation),
        nn.BatchNorm1d(8),
        conv_block(8,8,3,1,1,activation, dropout = dropout),
        nn.Conv1d(8, 1, 3, stride=1, padding=1)
        
    )
class gMLPmodel(pl.LightningModule):
    def __init__(
        self,
        *,
        dim =64 ,
        depth = 3 ,
        seq_len = 47,
        heads = 1,
        ff_mult = 2,
        attn_dim = None,
        causal = False,
        circulant_matrix = True,
        shift_tokens = 0,
        act = nn.Identity(),
        dropout=0.1,
        activation= nn.GELU() ,
        loss_types = F.binary_cross_entropy_with_logits,
        loss_weights = [0.4,0.6]
    ):
        super().__init__()
        self.save_hyperparameters()
        self.loss_weights = loss_weights
        assert (dim % heads) == 0, 'dimension must be divisible by number of heads'
        self.dim = dim
        dim_ff = dim * ff_mult
                
    
        self.encoder = encoder(activation=activation,dropout = dropout)

        self.gMLPlayers = nn.ModuleList([Residual(PreNorm(dim, gMLPBlock(dim = dim, heads = heads, dim_ff = dim_ff, seq_len = seq_len, attn_dim = attn_dim, causal = causal, act = act, circulant_matrix = circulant_matrix))) for i in range(depth)])
     
        
        self.decoderP = decoder(activation=activation,dropout = dropout)
        self.decoderS = decoder(activation=activation,dropout = dropout)

        self.criterion =  loss_types #F.binary_cross_entropy_with_logits
        
        
    def forward(self, x):
        
        x = np.squeeze(x)
        x = self.encoder(x)

        x.transpose_(1, 2)
        x = nn.Sequential(*self.gMLPlayers)(x)
        x.transpose_(1, 2)
 
        x_P = self.decoderP(x)
        x_S = self.decoderS(x)
        
        return torch.cat((x_P,x_S), 1 )

    def training_step(self, batch, batch_idx):
        
        x, y = batch
        y = np.squeeze(y)
        y_hat = self.forward(x)
        y_hatP = y_hat[:,0,:].reshape(-1,1)
        yP = y[:,0,:].reshape(-1,1)
        lossP = self.criterion(y_hatP, yP)* self.loss_weights[0]

        y_hatS = y_hat[:,1,:].reshape(-1,1)
        yS = y[:,1,:].reshape(-1,1)
        lossS = self.criterion(y_hatS, yS)* self.loss_weights[1]
        
        loss = lossP+lossS

        
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_lossP", lossP, on_epoch=True, prog_bar=True)
        self.log("train_lossS", lossS, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = np.squeeze(y)

        y_hat = self.forward(x)
        
        y_hatP = y_hat[:,0,:].reshape(-1,1)
        yP = y[:,0,:].reshape(-1,1)
        lossP = self.criterion(y_hatP, yP)* self.loss_weights[0]

        y_hatS = y_hat[:,1,:].reshape(-1,1)
        yS = y[:,1,:].reshape(-1,1)
        lossS = self.criterion(y_hatS, yS)* self.loss_weights[1]
        
        loss = lossP+lossS
        
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_lossP", lossP, on_epoch=True, prog_bar=True)
        self.log("val_lossS", lossS, on_epoch=True, prog_bar=True)

        return {'val_loss': loss}
    
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "frequency": 1
                       },
        
        }
    
