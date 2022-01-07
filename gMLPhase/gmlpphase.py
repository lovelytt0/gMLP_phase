from .base import WaveformModel, ActivationLSTMCell, CustomLSTM

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
import numpy as np


def exists(val):
    return val is not None


# For implementation, potentially follow: https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28
class gmlpphase(WaveformModel):
    """
    The EQTranformer from Mousavi et al. (2020)

    Implementation adapted from the Github repository https://github.com/smousavi05/EQTransformer
    Assumes padding="same" and activation="relu" as in the pretrained EQTransformer models

    By instantiating the model with `from_pretrained("original")` a binary compatible version of the original
    EQTransformer with the original weights from Mousavi et al. (2020) can be loaded.

    :param in_channels: Number of input channels, by default 3.
    :param in_samples: Number of input samples per channel, by default 6000.
                       The model expects input shape (in_channels, in_samples)
    :param classes: Number of output classes, by default 2. The detection channel is not counted.
    :param phases: Phase hints for the classes, by default "PS". Can be None.
    :param res_cnn_blocks: Number of residual convolutional blocks
    :param lstm_blocks: Number of LSTM blocks
    :param drop_rate: Dropout rate
    :param original_compatible: If True, uses a few custom layers for binary compatibility with original model
                                from Mousavi et al. (2020).
                                This option defaults to False.
                                It is usually recommended to stick to the default value, as the custom layers show
                                slightly worse performance than the PyTorch builtins.
                                The exception is when loading the original weights using :py:func:`from_pretrained`.
    :param kwargs: Keyword arguments passed to the constructor of :py:class:`WaveformModel`.
    """

    def __init__(
        self,
        in_channels=3,
        in_samples=6000,
        classes=2,
        phases="PS",
        dim =64 ,
        depth=3,
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
        loss_weights = [0.05,0.4,0.55] ,   
        sampling_rate=100,
        **kwargs,
    ):
        citation = (
            "Mousavi, S.M., Ellsworth, W.L., Zhu, W., Chuang, L, Y., and Beroza, G, C. "
            "Earthquake transformer—an attentive deep-learning model for simultaneous earthquake "
            "detection and phase picking. Nat Commun 11, 3952 (2020). "
            "https://doi.org/10.1038/s41467-020-17591-w"
        )
        # Blinding defines how many samples at beginning and end of the prediction should be ignored
        # This is usually required to mitigate prediction problems from training properties, e.g.,
        # if all picks in the training fall between seconds 5 and 55.
        super().__init__(
            citation=citation,
            output_type="array",
            default_args={"overlap": 1800, "blinding": (500, 500)},
            in_samples=in_samples,
            pred_sample=(0, in_samples),
            labels=["Detection"] + list(phases),
            sampling_rate=sampling_rate,
            **kwargs,
        )

#         self.in_channels = in_channels
#         self.classes = classes
#         self.lstm_blocks = lstm_blocks
#         self.drop_rate = drop_rate
#         self.original_compatible = original_compatible

#         if original_compatible and in_samples != 6000:
#             raise ValueError("original_compatible=True requires in_samples=6000.")

#         self._phases = phases
#         if phases is not None and len(phases) != classes:
#             raise ValueError(
#                 f"Number of classes ({classes}) does not match number of phases ({len(phases)})."
#             )
        
#         self.save_hyperparameters()
        self.loss_weights = loss_weights
        assert (dim % heads) == 0, 'dimension must be divisible by number of heads'
        self.dim = dim
        dim_ff = dim * ff_mult
#         self.prob_survival = prob_survival
                
        
        self.conv0 =  conv_block(3, 8, 3, 1, 1, activation, dropout = dropout)
        self.batch_norm0 = nn.BatchNorm1d(8)

        self.conv1_0 = conv_block(8, 8, 3, 2, 1, activation)
        self.conv1_1 = conv_block(8, 8, 3, 1, 1, activation, dropout = dropout)
        self.batch_norm1 = nn.BatchNorm1d(8)
        
        self.conv2_0 = conv_block(8, 16, 3, 2, 1, activation)
        self.conv2_1 = conv_block(16, 16, 3, 1, 1, activation, dropout = dropout)
        self.batch_norm2 = nn.BatchNorm1d(16)

        self.conv3_0 = conv_block(16, 16, 3, 2, 1, activation)
        self.conv3_1 = conv_block(16, 16, 3, 1, 1, activation, dropout = dropout)
        self.batch_norm3 = nn.BatchNorm1d(16)

        self.conv4_0 = conv_block(16, 32, 3, 2, 1, activation)
        self.conv4_1 = conv_block(32, 32, 3, 1, 1, activation, dropout = dropout)
        self.batch_norm4 = nn.BatchNorm1d(32)

        
        self.conv5_0 = conv_block(32, 32, 3, 2, 1, activation)
        self.conv5_1 = conv_block(32, 32, 3, 1, 1, activation, dropout = dropout)
        self.batch_norm5 = nn.BatchNorm1d(32)
        
        self.conv6_0 = conv_block(32, 64, 3, 2, 1, activation)
        self.conv6_1 = conv_block(64, 64, 3, 1, 1, activation, dropout = dropout)
        self.batch_norm6 = nn.BatchNorm1d(64)
         
        self.conv7_0 = conv_block(64, 64, 3, 2, 1, activation)
        self.conv7_1 = conv_block(64, 64, 3, 1, 1, activation, dropout = dropout)
        self.batch_norm7 = nn.BatchNorm1d(64)
#         token_shifts = tuple(range(0 if causal else -shift_tokens, shift_tokens + 1))
#         self.gMLPlayers = nn.ModuleList([Residual(PreNorm(dim, PreShiftTokens(token_shifts, gMLPBlock(dim = dim, heads = heads, dim_ff = dim_ff, seq_len = seq_len, attn_dim = attn_dim, causal = causal, act = act, circulant_matrix = circulant_matrix)))) for i in range(depth)])

        self.gMLPlayers = nn.ModuleList([Residual(PreNorm(dim, gMLPBlock(dim = dim, heads = heads, dim_ff = dim_ff, seq_len = seq_len, attn_dim = attn_dim, causal = causal, act = act, circulant_matrix = circulant_matrix))) for i in range(depth)])
     
        
        self.deconv0_0 = deconv_block(64, 64, 3, 2, padding = 1, output_padding=1, activation=activation)
        self.deconv1_0 = deconv_block(64, 32, 3, 2, padding = 1, output_padding=1, activation=activation)
        self.deconv2_0 = deconv_block(32, 32, 3, 2, padding = 1, output_padding=0, activation=activation)
        self.deconv3_0 = deconv_block(32, 16, 3, 2, padding = 1, output_padding=1, activation=activation)
        self.deconv4_0 = deconv_block(16, 16, 3, 2, padding = 1, output_padding=1, activation=activation)
        self.deconv5_0 = deconv_block(16, 8, 3, 2, padding = 1, output_padding=1, activation=activation)
        self.deconv6_0 = deconv_block(8, 8, 3, 2, padding = 1, output_padding=1, activation=activation)
        
        

        self.deconv0_1 = conv_block(128,64,3,1,1,activation, dropout = dropout)
        self.deconv1_1 = conv_block(64,32,3,1,1,activation, dropout = dropout)
        self.deconv2_1 = conv_block(64,32,3,1,1,activation, dropout = dropout)
        self.deconv3_1 = conv_block(32,16,3,1,1,activation, dropout = dropout)
        self.deconv4_1 = conv_block(32,16,3,1,1,activation, dropout = dropout)
        self.deconv5_1 = conv_block(16,8,3,1,1,activation, dropout = dropout)
        self.deconv6_1 = conv_block(16,8,3,1,1,activation, dropout = dropout)
        self.deconv6_2 = nn.Conv1d(8, 3, 3, stride=1, padding=1)
        
        self.batch_norm8 = nn.BatchNorm1d(64)
        self.batch_norm9 = nn.BatchNorm1d(32)
        self.batch_norm10 = nn.BatchNorm1d(32)
        self.batch_norm11 = nn.BatchNorm1d(16)
        self.batch_norm12 = nn.BatchNorm1d(16)
        self.batch_norm13 = nn.BatchNorm1d(8)
        self.batch_norm14 = nn.BatchNorm1d(8)
        
        self.criterion =  loss_types #F.binary_cross_entropy_with_logits
        
        
    def forward(self, x):
        x = np.squeeze(x)
        x0 = self.conv0(x)
        x0 = self.batch_norm0(x0)
        x1 = self.conv1_1(self.batch_norm1(self.conv1_0(x0)))
        x2 = self.conv2_1(self.batch_norm2(self.conv2_0(x1)))
        x3 = self.conv3_1(self.batch_norm3(self.conv3_0(x2)))
        x4 = self.conv4_1(self.batch_norm4(self.conv4_0(x3)))
        x5 = self.conv5_1(self.batch_norm5(self.conv5_0(x4)))
        x6 = self.conv6_1(self.batch_norm6(self.conv6_0(x5)))
        x7 = self.conv7_1(self.batch_norm7(self.conv7_0(x6)))

        x7.transpose_(1, 2)
        x7 = nn.Sequential(*self.gMLPlayers)(x7)
        x7.transpose_(1, 2)
 
        x8 = torch.cat((self.batch_norm8(self.deconv0_0(x7)), x6), 1)
        x8 = self.deconv0_1(x8)

        x9 = torch.cat((self.batch_norm9(self.deconv1_0(x8)), x5), 1)
        x9 = self.deconv1_1(x9)
  
        x10 = torch.cat((self.batch_norm10(self.deconv2_0(x9)), x4), 1)
        x10 = self.deconv2_1(x10)
        
        x11 = torch.cat((self.batch_norm11(self.deconv3_0(x10)), x3), 1)
        x11 = self.deconv3_1(x11)
        
        x12 = torch.cat((self.batch_norm12(self.deconv4_0(x11)), x2), 1)
        x12 = self.deconv4_1(x12)
        
        x13 = torch.cat((self.batch_norm13(self.deconv5_0(x12)), x1), 1)
        x13 = self.deconv5_1(x13)
        
        x14 = torch.cat((self.batch_norm14(self.deconv6_0(x13)), x0), 1)
        x14 = self.deconv6_1(x14)
        x14 = self.deconv6_2(x14)

        return torch.sigmoid(x14)

    def annotate_window_post(self, pred, argdict):
        # Combine predictions in one array
        prenan, postnan = argdict.get("blinding", (0, 0))
        pred = np.stack(pred, axis=-1)
        pred[:prenan] = np.nan
        pred[-postnan:] = np.nan
        return pred

    def annotate_window_pre(self, window, argdict):
        # Add a demean and an amplitude normalization step to the preprocessing
        window = window - np.mean(window, axis=-1, keepdims=True)
        window = window / (np.std(window) + 1e-10)
        return window

    @property
    def phases(self):
        if self._phases is not None:
            return self._phases
        else:
            return list(range(self.classes))

    def classify_aggregate(self, annotations, argdict):
        """
        Converts the annotations to discrete picks using
        :py:func:`~seisbench.models.base.WaveformModel.picks_from_annotations`
        and to discrete detections using :py:func:`~seisbench.models.base.WaveformModel.detections_from_annotations`.
        Trigger onset thresholds for picks are derived from the argdict at keys "[phase]_threshold".
        Trigger onset thresholds for detections are derived from the argdict at key "detection_threshold".

        :param annotations: See description in superclass
        :param argdict: See description in superclass
        :return: List of picks, list of detections
        """
        picks = []
        for phase in self.phases:
            picks += self.picks_from_annotations(
                annotations.select(channel=f"EQTransformer_{phase}"),
                argdict.get(f"{phase}_threshold", 0.1),
                phase,
            )

        detections = self.detections_from_annotations(
            annotations.select(channel="EQTransformer_Detection"),
            argdict.get("detection_threshold", 0.3),
        )

        return sorted(picks), sorted(detections)


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