import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import time
from EdgeConv.DataGeneratorMulti import DataGeneratorMulti
from torch import nn

from torch.utils.data import DataLoader, Dataset
import torch
import torch.optim as optim
import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn.functional as F

from pytorch_lightning import loggers as pl_loggers
from gMLPhase.gMLP_torch import gMLPmodel
from gMLPhase.gMLP_torch import gMLPBlock, Residual, PreNorm ,  conv_block, deconv_block


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,SAGEConv
from torch import nn
from torch_geometric.nn import MessagePassing

import logging
root_logger= logging.getLogger()
root_logger.setLevel(logging.DEBUG) # or whatever
handler = logging.FileHandler('debug5.log', 'w', 'utf-8') # or whatever
handler.setFormatter(logging.Formatter('%(name)s %(message)s')) # or whatever
root_logger.addHandler(handler)

def cycle(loader):
    while True:
        for data in loader:
            yield data


def trainerMulti(input_hdf5=None,
            input_trainset = None,
            input_validset = None,
            output_name = None, 
            input_model = None,
            hparams_file = None,
            model_folder = None,
            input_dimention=(6000, 3),
            shuffle=True, 
            label_type='triangle',
            normalization_mode='std',
            augmentation=True,
            add_event_r=0.6,
            shift_event_r=0.99,
            add_noise_r=0.3, 
            drop_channel_r=0.5,
            add_gap_r=0.2,
            coda_ratio=1.4,
            scale_amplitude_r=None,
            pre_emphasis=False,                
            batch_size=1,
            epochs=200, 
            monitor='val_loss',
            patience=3):
        
    """
    
    Generate a model and train it.  
    
    Parameters
    ----------
    input_hdf5: str, default=None
        Path to an hdf5 file containing only one class of data with NumPy arrays containing 3 component waveforms each 1 min long.
    input_csv: str, default=None
        Path to a CSV file with one column (trace_name) listing the name of all datasets in the hdf5 file.
    output_name: str, default=None
        Output directory.
        
    input_dimention: tuple, default=(6000, 3)
        OLoss types for detection, P picking, and S picking respectively. 
        
    cnn_blocks: int, default=5
        The number of residual blocks of convolutional layers.
        
    lstm_blocks: int, default=2
        The number of residual blocks of BiLSTM layers.
    
        
    activation: str, default='relu'
        Activation function used in the hidden layers.
    drop_rate: float, default=0.1
        Dropout value.
    shuffle: bool, default=True
        To shuffle the list prior to the training.
    label_type: str, default='triangle'
        Labeling type. 'gaussian', 'triangle', or 'box'. 
    normalization_mode: str, default='std'
        Mode of normalization for data preprocessing, 'max': maximum amplitude among three components, 'std', standard deviation. 
    augmentation: bool, default=True
        If True, data will be augmented simultaneously during the training.
    add_event_r: float, default=0.6
        Rate of augmentation for adding a secondary event randomly into the empty part of a trace.
    shift_event_r: float, default=0.99
        Rate of augmentation for randomly shifting the event within a trace.
      
    add_noise_r: float, defaults=0.3 
        Rate of augmentation for adding Gaussian noise with different SNR into a trace.       
        
    drop_channel_r: float, defaults=0.4 
        Rate of augmentation for randomly dropping one of the channels.
    add_gap_r: float, defaults=0.2 
        Add an interval with zeros into the waveform representing filled gaps. 
    coda_ratio: float, defaults=0.4
        % of S-P time to extend event/coda envelope past S pick.
        
    scale_amplitude_r: float, defaults=None
        Rate of augmentation for randomly scaling the trace. 
              
    pre_emphasis: bool, defaults=False
        If True, waveforms will be pre-emphasized. Defaults to False.  
        
    loss_weights: list, defaults=[0.03, 0.40, 0.58]
        Loss weights for detection, P picking, and S picking respectively.
              
    loss_types: list, defaults=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'] 
        Loss types for detection, P picking, and S picking respectively.  
        
    train_valid_test_split: list, defaults=[0.85, 0.05, 0.10]
        Precentage of data split into the training, validation, and test sets respectively. 
          
    mode: str, defaults='generator'
        Mode of running. 'generator', or 'preload'. 
         
    batch_size: int, default=200
        Batch size.
          
    epochs: int, default=200
        The number of epochs.
          
    monitor: int, default='val_loss'
        The measure used for monitoring.
           
    patience: int, default=12
        The number of epochs without any improvement in the monitoring measure to automatically stop the training.          
           
        
    Returns
    -------- 
    output_name/models/output_name_.h5: This is where all good models will be saved.  
    
    output_name/final_model.h5: This is the full model for the last epoch.
    
    output_name/model_weights.h5: These are the weights for the last model.
    
    output_name/history.npy: Training history.
    
    output_name/X_report.txt: A summary of the parameters used for prediction and performance.
    
    output_name/test.npy: A number list containing the trace names for the test set.
    
    output_name/X_learning_curve_f1.png: The learning curve of Fi-scores.
    
    output_name/X_learning_curve_loss.png: The learning curve of loss.
    Notes
    -------- 
    'generator' mode is memory efficient and more suitable for machines with fast disks. 
    'pre_load' mode is faster but requires more memory and it comes with only box labeling.
        
    """     


    args = {
    "input_hdf5": input_hdf5,
    "input_trainset": input_trainset,
    "input_validset":  input_validset,
    "output_name": output_name,
    "input_model": input_model,
    "hparams_file": hparams_file,
    "model_folder": model_folder,
    "input_dimention": input_dimention,
    "shuffle": shuffle,
    "label_type": label_type,
    "normalization_mode": normalization_mode,
    "augmentation": augmentation,
    "add_event_r": add_event_r,
    "shift_event_r": shift_event_r,
    "add_noise_r": add_noise_r,
    "add_gap_r": add_gap_r,
    "coda_ratio": coda_ratio,
    "drop_channel_r": drop_channel_r,
    "scale_amplitude_r": scale_amplitude_r,
    "pre_emphasis": pre_emphasis,
    "batch_size": batch_size,
    "epochs": epochs,
    "monitor": monitor,
    "patience": patience                    
    }
    
    save_dir, save_models=_make_dir(args['output_name'])
    training = np.load(input_trainset)
    validation = np.load(input_validset)
    start_training = time.time()

    params_training = {'file_name': str(args['input_hdf5']), 
                      'dim': args['input_dimention'][0],
                      'batch_size': 1,
                      'n_channels': args['input_dimention'][-1],
                      'shuffle': args['shuffle'],  
                      'norm_mode': args['normalization_mode'],
                      'label_type': args['label_type'],
                      'augmentation': args['augmentation'],
                      'add_event_r': args['add_event_r'],
                      'add_gap_r': args['add_gap_r'],
                      'coda_ratio': args['coda_ratio'],
                      'shift_event_r': args['shift_event_r'],    
                      'add_noise_r': args['add_noise_r'],
                      'drop_channel_r': args['drop_channel_r'],
                      'scale_amplitude_r': args['scale_amplitude_r'],
                      'pre_emphasis': args['pre_emphasis']}

    params_validation = {'file_name': str(args['input_hdf5']),  
                         'dim': args['input_dimention'][0],
                         'batch_size': 1,
                         'n_channels': args['input_dimention'][-1],
                         'shuffle': False,  
                        'coda_ratio': args['coda_ratio'],
                         'norm_mode': args['normalization_mode'],
                        'label_type': args['label_type'],
                         'augmentation': False}         
    model = gMLPmodel.load_from_checkpoint(checkpoint_path=os.path.join(args['model_folder'],args['input_model']),hparams_file=os.path.join(args['model_folder'],args['hparams_file']))
# change into eval mode

    model.eval() 
    model_GNN = Graphmodel(pre_model=model)


    training_generator = DataGeneratorMulti(list_IDs=training, **params_training)
    validation_generator = DataGeneratorMulti(list_IDs=validation, **params_validation) 

#     for i in [3,5919,5920,9651,9652]:
#         x=training_generator.__getitem__(i)
#         print(x[-1])
#         print(x[0].shape)
#         print(x[1].shape)
#         print(x[2].shape)
#         print(x[0].max())

#         print(torch.sum(x[0]))
#         print(torch.sum(x[1]))


#     return
    
    checkpoint_callback = ModelCheckpoint(monitor=monitor,dirpath=save_models,save_top_k=3,verbose=True,save_last=True)
    early_stopping = EarlyStopping(monitor=monitor,patience=args['patience']) # patience=3
    tb_logger = pl_loggers.TensorBoardLogger(save_dir)

    trainer = pl.Trainer(precision=16, gpus=1,gradient_clip_val=0.5, accumulate_grad_batches=16, callbacks=[early_stopping, checkpoint_callback],check_val_every_n_epoch=1,profiler="simple",num_sanity_val_steps=0, logger =tb_logger)

    train_loader  = DataLoader(training_generator, batch_size = args['batch_size'], num_workers=8, pin_memory=True, prefetch_factor=5)

    val_loader   = DataLoader(validation_generator, batch_size = args['batch_size'], num_workers=8, pin_memory=True, prefetch_factor=5)

    print('Started training in generator mode ...') 

    trainer.fit(model_GNN, train_dataloaders = train_loader, val_dataloaders = val_loader)

    end_training = time.time()  
    print('Finished Training')
        
        
    
def _make_dir(output_name):
    
    """ 
    
    Make the output directories.
    Parameters
    ----------
    output_name: str
        Name of the output directory.
                   
    Returns
    -------   
    save_dir: str
        Full path to the output directory.
        
    save_models: str
        Full path to the model directory. 
        
    """   
    
    if output_name == None:
        print('Please specify output_name!') 
        return
    else:
        save_dir = os.path.join(os.getcwd(), str(output_name))
        save_models = os.path.join(save_dir, 'checkpoints')      
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)  
        os.makedirs(save_models)
        shutil.copyfile('EdgeConv/trainerMulti.py',os.path.join(save_dir,'trainer.py'))
    return save_dir, save_models





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

class EdgeConv(MessagePassing):
    def __init__(self, in_channels):
        super().__init__(aggr='max',node_dim=-3) #  "Max" aggregation.
        activation= nn.GELU() 
        dropout=0.1


        self.deconv1 = conv_block(in_channels*2, in_channels*2, 3, 1, 1, activation=activation, dropout=0.1)
        self.deconv2 = conv_block(in_channels*2, in_channels*2, 3, 1, 1, activation=activation,dropout=0.1)
        self.deconv3 = conv_block(in_channels*2, in_channels, 3, 1, 1,activation=activation,dropout=0.1)
        self.deconv4 = conv_block(in_channels, in_channels, 3, 1, 1,  activation=activation,dropout=0.1)
        self.deconv5 = conv_block(in_channels, in_channels, 3, 1, 1, activation=activation,dropout=0.1)

#         self.gMLPmessage = nn.ModuleList([Residual(PreNorm(in_channels*2, gMLPBlock(dim = in_channels*2, heads = 1, dim_ff = in_channels*2, seq_len = 47, attn_dim = None, causal = False))) for i in range(1)])
#         self.proj_out = nn.Linear(in_channels*2, in_channels)
#         self.conv3 = conv_block(in_channels, out_channels, 3, 1, 1, activation)
    

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

#         tmp = torch.cat([x_i, x_j], dim=2)  # tmp has shape [E, 2 * in_channels]
        tmp = torch.cat([x_i, x_j], dim=1)  # tmp has shape [E, 2 * in_channels]
#         tmp = nn.Sequential(*self.gMLPmessage)(tmp)
#         return self.proj_out(tmp)
        ans = self.deconv5(self.deconv4(self.deconv3(self.deconv2(self.deconv1(tmp)))))
        return ans
    
    
    
    
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





class Graphmodel(pl.LightningModule):
    def __init__(
        self,
        pre_model
    ):
        super().__init__()
        
        self.edgeconv1 = EdgeConv(64)
        for name, p in pre_model.named_parameters():
            if "encoder" in name  or "gMLPlayers" in name :
                p.requires_grad = False
                print(name)
            else:
                p.requires_grad = True
        
        self.encoder = pre_model.encoder
        self.gMLPlayers = pre_model.gMLPlayers
        
        self.decoderP = pre_model.decoderP
        self.decoderS = pre_model.decoderS

        self.criterion =  pre_model.criterion 
        self.loss_weights= pre_model.loss_weights
        print('loss weight', self.loss_weights)
        
    def forward(self, data):
        
        
        x, edge_index = data[0], data[2]
        x = np.squeeze(x)
        edge_index=np.squeeze(edge_index)

        x = self.encoder(x)

        x.transpose_(1, 2)
        x = nn.Sequential(*self.gMLPlayers)(x)
#         x = self.edgeconv1(x,edge_index)
        
        x.transpose_(1, 2)
        x = self.edgeconv1(x,edge_index)

        x_P = self.decoderP(x)
        x_S = self.decoderS(x)
        
        return torch.cat((x_P,x_S), 1 )



    
    
    def training_step(self, batch, batch_idx):
    
        y = batch[1][0]
        y = np.squeeze(y)
        
        y_hat = self.forward(batch)
        y_hatP = y_hat[:,0,:].reshape(-1,1)
        yP = y[:,0,:].reshape(-1,1)
        lossP = self.criterion(y_hatP, yP)* self.loss_weights[0]

        y_hatS = y_hat[:,1,:].reshape(-1,1)
        yS = y[:,1,:].reshape(-1,1)
        lossS = self.criterion(y_hatS, yS)* self.loss_weights[1]
        
        loss = lossP+lossS
        if np.isnan(loss.detach().cpu()):
            
            logging.debug('This message should go to the log file')
            logging.debug('help')
            logging.debug(batch[-1])
            logging.debug(np.sum(np.array(batch[0].detach().cpu())))
            logging.debug(batch_idx)
            logging.debug(batch)
            
            logging.debug(np.sum(np.array(y_hatP.detach().cpu())))
            logging.debug(np.array(y_hatP.detach().cpu()))

            logging.debug(np.sum(np.array(yP.detach().cpu())))
            logging.debug(np.sum(np.array(y_hatS.detach().cpu())))
            logging.debug(np.sum(np.array(yS.detach().cpu())))
            
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_lossP", lossP, on_epoch=True, prog_bar=True)
        self.log("train_lossS", lossS, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        y = batch[1][0]
        y = np.squeeze(y)

        y_hat = self.forward(batch)
        
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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train_loss",
                    "frequency": 5000
                       },
        
        }
    

    
    
    
    
    
# class Graphmodel(pl.LightningModule):
#     def __init__(self,
#         pre_model
#      ):
#         super().__init__()
# #         activation= nn.GELU() 
# #         self.save_hyperparameters()
#         self.edgeconv1 = EdgeConv(64)
        
# #         for name, p in pre_model.named_parameters():
# #             if "deconv6_" in name :
# #                 print(name)
# #                 p.requires_grad = True
# #             else:
# #             p.requires_grad = False

#         # repeat pre_model module
# #         self.conv0 = pre_model.conv0
# #         self.conv1_0 = pre_model.conv1_0
# #         self.conv2_0 = pre_model.conv2_0
# #         self.conv3_0 = pre_model.conv3_0
# #         self.conv4_0 = pre_model.conv4_0
# #         self.conv5_0 = pre_model.conv5_0
# #         self.conv6_0 = pre_model.conv6_0
# #         self.conv7_0 = pre_model.conv7_0

        
# #         self.conv1_1 = pre_model.conv1_1
# #         self.conv2_1 = pre_model.conv2_1
# #         self.conv3_1 = pre_model.conv3_1
# #         self.conv4_1 = pre_model.conv4_1
# #         self.conv5_1 = pre_model.conv5_1
# #         self.conv6_1 = pre_model.conv6_1
# #         self.conv7_1 = pre_model.conv7_1

        
# #         self.deconv0_0 = pre_model.deconv0_0
# #         self.deconv1_0 = pre_model.deconv1_0
# #         self.deconv2_0 = pre_model.deconv2_0
# #         self.deconv3_0 = pre_model.deconv3_0
# #         self.deconv4_0 = pre_model.deconv4_0
# #         self.deconv5_0 = pre_model.deconv5_0
# #         self.deconv6_0 = pre_model.deconv6_0

# #         self.deconv0_1 = pre_model.deconv0_1
# #         self.deconv1_1 = pre_model.deconv1_1
# #         self.deconv2_1 = pre_model.deconv2_1
# #         self.deconv3_1 = pre_model.deconv3_1
# #         self.deconv4_1 = pre_model.deconv4_1
# #         self.deconv4_1 = pre_model.deconv4_1
# #         self.deconv5_1 = pre_model.deconv5_1
# #         self.deconv6_1 = pre_model.deconv6_1
# #         self.deconv6_2 = pre_model.deconv6_2

# #         self.batch_norm0 = pre_model.batch_norm0
# #         self.batch_norm1 = pre_model.batch_norm1
# #         self.batch_norm2 = pre_model.batch_norm2
# #         self.batch_norm3 = pre_model.batch_norm3
# #         self.batch_norm4 = pre_model.batch_norm4
# #         self.batch_norm5 = pre_model.batch_norm5
# #         self.batch_norm6 = pre_model.batch_norm6
# #         self.batch_norm7 = pre_model.batch_norm7
# #         self.batch_norm8 = pre_model.batch_norm8
# #         self.batch_norm9 = pre_model.batch_norm9
# #         self.batch_norm10 = pre_model.batch_norm10
# #         self.batch_norm11 = pre_model.batch_norm11
# #         self.batch_norm12 = pre_model.batch_norm12
# #         self.batch_norm13 = pre_model.batch_norm13
# #         self.batch_norm14 = pre_model.batch_norm14
        
# #         self.gMLPlayers = pre_model.gMLPlayers
        
# #         self.criterion = pre_model.criterion
# #         self.loss_weights= pre_model.loss_weights
        
#     def forward(self, data):
        
#         x, edge_index = data[0], data[2]
#         x = np.squeeze(x)
#         edge_index=np.squeeze(edge_index)

#         x0 = self.conv0(x)
#         x0 = self.batch_norm0(x0)
#         x1 = self.conv1_1(self.batch_norm1(self.conv1_0(x0)))
#         x2 = self.conv2_1(self.batch_norm2(self.conv2_0(x1)))
#         x3 = self.conv3_1(self.batch_norm3(self.conv3_0(x2)))
#         x4 = self.conv4_1(self.batch_norm4(self.conv4_0(x3)))
#         x5 = self.conv5_1(self.batch_norm5(self.conv5_0(x4)))
#         x6 = self.conv6_1(self.batch_norm6(self.conv6_0(x5)))
#         x7 = self.conv7_1(self.batch_norm7(self.conv7_0(x6)))

#         x7.transpose_(1, 2)
# #         gMLPlayers=self.gMLPlayers if not self.training else dropout_layers(self.gMLPlayers, self.prob_survival)
#         x7 = nn.Sequential(*self.gMLPlayers)(x7)
    
#         x_exchange = self.edgeconv1(x7,edge_index)
#         x7 = x7+x_exchange
#         x7.transpose_(1, 2)
#         # exchange info


#         x8 = torch.cat((self.batch_norm8(self.deconv0_0(x7)), x6), 1)
#         x8 = self.deconv0_1(x8)

#         x9 = torch.cat((self.batch_norm9(self.deconv1_0(x8)), x5), 1)
#         x9 = self.deconv1_1(x9)
  
#         x10 = torch.cat((self.batch_norm10(self.deconv2_0(x9)), x4), 1)
#         x10 = self.deconv2_1(x10)
        
#         x11 = torch.cat((self.batch_norm11(self.deconv3_0(x10)), x3), 1)
#         x11 = self.deconv3_1(x11)
        
#         x12 = torch.cat((self.batch_norm12(self.deconv4_0(x11)), x2), 1)
#         x12 = self.deconv4_1(x12)
        
#         x13 = torch.cat((self.batch_norm13(self.deconv5_0(x12)), x1), 1)
#         x13 = self.deconv5_1(x13)
        
# #         x13.transpose_(1, 2)
# #         x13.transpose_(1, 2)

#         x14 = torch.cat((self.batch_norm14(self.deconv6_0(x13)), x0), 1)
#         x14 = self.deconv6_1(x14)
#         x14 = self.deconv6_2(x14)
        


#         return x14
    
    
    
#     def training_step(self, batch, batch_idx):
#         # training_step defined the train loop.
#         # It is independent of forward
#         y = batch[1][0]
#         y = np.squeeze(y)

#         y_hat = self.forward(batch)
# #         y_hat2 = y_hat.view(-1,1)
# #         y2 = y.view(-1,1)
# #         loss = self.criterion(y_hat2, y2)

#         y_hatD = y_hat[:,0,:].reshape(-1,1)
#         yD = y[:,0,:].reshape(-1,1)
#         lossD = self.criterion(y_hatD, yD)* self.loss_weights[0]

#         y_hatP = y_hat[:,1,:].reshape(-1,1)
#         yP = y[:,1,:].reshape(-1,1)
#         lossP = self.criterion(y_hatP, yP)* self.loss_weights[1]

#         y_hatS = y_hat[:,2,:].reshape(-1,1)
#         yS = y[:,2,:].reshape(-1,1)
#         lossS = self.criterion(y_hatS, yS)* self.loss_weights[2]
        
#         loss = lossD+lossP+lossS
        
#         self.log("train_loss", loss, on_epoch=True, prog_bar=True)

#         self.log("train_lossD", lossD, on_epoch=True, prog_bar=True)
#         self.log("train_lossP", lossP, on_epoch=True, prog_bar=True)
#         self.log("train_lossS", lossS, on_epoch=True, prog_bar=True)

        
#         return loss

#     def validation_step(self, batch, batch_idx): 
#         y = batch[1][0]
#         y = np.squeeze(y)

#         y_hat = self.forward(batch)
# #         y_hat2 = y_hat.view(-1,1)
# #         y2 = y.view(-1,1)
# #         loss = self.criterion(y_hat2, y2)
        
#         y_hatD = y_hat[:,0,:].reshape(-1,1)
#         yD = y[:,0,:].reshape(-1,1)
#         lossD = self.criterion(y_hatD, yD)* self.loss_weights[0]

#         y_hatP = y_hat[:,1,:].reshape(-1,1)
#         yP = y[:,1,:].reshape(-1,1)
#         lossP = self.criterion(y_hatP, yP)* self.loss_weights[1]

#         y_hatS = y_hat[:,2,:].reshape(-1,1)
#         yS = y[:,2,:].reshape(-1,1)
#         lossS = self.criterion(y_hatS, yS) *self.loss_weights[2]
        
#         loss = lossD+lossP+lossS
    
        
#         self.log("val_loss", loss, on_epoch=True, prog_bar=True)
# #         self.log("val_lossD", lossD, on_epoch=True, prog_bar=True)
#         self.log("val_lossP", lossP, on_epoch=True, prog_bar=True)
#         self.log("val_lossS", lossS, on_epoch=True, prog_bar=True)

#         return {'val_loss': loss}
    
    
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
#         return {
#                 "optimizer": optimizer,
#                 "lr_scheduler": {
#                     "scheduler": scheduler,
#                     "monitor": "val_loss",
#                     "frequency": 1
#                        },
        
#         }
    
    
    
