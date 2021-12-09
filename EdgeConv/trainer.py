import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import time
from EdgeConv.DataGeneratorGNN import DataGeneratorGNN
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


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,SAGEConv
from torch import nn
from torch_geometric.nn import MessagePassing

def cycle(loader):
    while True:
        for data in loader:
            yield data


def trainer(input_hdf5=None,
            input_trainset=None,
            output_name=None, 
            input_model = None,
            hparams_file = None,
            model_folder = None,
            input_dimention=(6000, 3),
            gmlp_blocks=5,
            gmlp_dim=32,
            seq_len = 375,
            activation = 'relu',            
            drop_rate=0.1,
            shuffle=True, 
            label_type='gaussian',
            normalization_mode='max',
            augmentation=True,
            add_event_r=0.6,
            shift_event_r=0.99,
            add_noise_r=0.3, 
            drop_channel_r=0.5,
            add_gap_r=0.2,
            coda_ratio=1.4,
            scale_amplitude_r=None,
            pre_emphasis=False,                
            loss_weights=[0.05, 0.40, 0.55],
            loss_types=F.binary_cross_entropy_with_logits,
            train_valid_test_split=[0.80, 0.20],
            mode='generator',
            batch_size=200,
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
    "output_name": output_name,
    "input_model": input_model,
    "hparams_file": hparams_file,
    "model_folder": model_folder,
    "input_dimention": input_dimention,
    "gmlp_blocks": gmlp_blocks,
    "gmlp_dim": gmlp_dim,
    "seq_len": seq_len,
    "activation": activation,
    "drop_rate": drop_rate,
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
    "loss_weights": loss_weights,
    "loss_types": loss_types,
    "train_valid_test_split": train_valid_test_split,
    "mode": mode,
    "batch_size": batch_size,
    "epochs": epochs,
    "monitor": monitor,
    "patience": patience                    
    }
    
    save_dir, save_models=_make_dir(args['output_name'])
    training, validation=_split(args,save_dir)
    
#     print(model)

    start_training = time.time()

    if args['mode'] == 'generator': 
        params_training = {'file_name': str(args['input_hdf5']), 
                          'dim': args['input_dimention'][0],
                          'batch_size': args['batch_size'],
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
                             'batch_size': args['batch_size'],
                             'n_channels': args['input_dimention'][-1],
                             'shuffle': False,  
                            'coda_ratio': args['coda_ratio'],
                             'norm_mode': args['normalization_mode'],
                            'label_type': args['label_type'],
                             'augmentation': False}         
        model = gMLPmodel.load_from_checkpoint(checkpoint_path=os.path.join(args['model_folder'],args['input_model']),hparams_file=os.path.join(args['model_folder'],args['hparams_file']))
    # change into eval mode
        
        model.eval() 
        args['pre_model']=model
        model_GNN=_build_model(args)

        
        training_generator = DataGeneratorGNN(list_IDs=training, **params_training)
        validation_generator = DataGeneratorGNN(list_IDs=validation, **params_validation) 
        
        checkpoint_callback = ModelCheckpoint(monitor=monitor,dirpath=save_models,save_top_k=3,verbose=True,save_last=True)
        early_stopping = EarlyStopping(monitor=monitor,patience=args['patience']) # patience=3
        tb_logger = pl_loggers.TensorBoardLogger(save_dir)
        
        trainer = pl.Trainer(precision=16, gpus=1, callbacks=[early_stopping, checkpoint_callback],check_val_every_n_epoch=1,profiler="simple",num_sanity_val_steps=0, logger =tb_logger)


        
                
        train_loader  = DataLoader(training_generator, batch_size = args['batch_size'], num_workers=8, pin_memory=True, prefetch_factor=5)
#         print('hello')
#         train_loader= iter(train_loader)
#         AAA=next(train_loader)
#         print(len(AAA),AAA[0].shape,AAA[1].shape )
#         return train_loader

        val_loader    = DataLoader(validation_generator, batch_size = args['batch_size'], num_workers=8, pin_memory=True, prefetch_factor=5)
# #         print(next(train_loader))

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
#         shutil.copyfile('gMLPhase/gMLP_torch.py',os.path.join(save_dir,'gMLP_torch.py'))
    return save_dir, save_models


def _build_model(args): 
    
    """ 
    
    Build and compile the model.
    Parameters
    ----------
    args: dic
        A dictionary containing all of the input parameters. 
               
    Returns
    -------   
    model: 
        Compiled model.
        
    """       
    model = Graphmodel(pre_model=args['pre_model'])



    return model  
    


def _split(args,save_dir):
    
    """ 
    
    Split the list of input data into training, validation, and test set.
    Parameters
    ----------
    args: dic
        A dictionary containing all of the input parameters. 
        
    save_dir: str
       Path to the output directory. 
              
    Returns
    -------   
    training: str
        List of trace names for the training set. 
    validation : str
        List of trace names for the validation set. 
                
    """       
    
    ev_list = np.load(args['input_trainset'],allow_pickle=True)
    
#     ev_list = df.trace_name.tolist()    
    np.random.shuffle(ev_list)     
    training = ev_list[:int(args['train_valid_test_split'][0]*len(ev_list))]
    validation =  ev_list[int(args['train_valid_test_split'][0]*len(ev_list)):
                            int(args['train_valid_test_split'][0]*len(ev_list) + args['train_valid_test_split'][1]*len(ev_list))]
#     test =  ev_list[ int(args['train_valid_test_split'][0]*len(ev_list) + args['train_valid_test_split'][1]*len(ev_list)):]
    np.save(os.path.join(save_dir,'training.npy'), training)  
    np.save(os.path.join(save_dir,'validation.npy'), validation)  

    return training, validation 



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
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='max',node_dim=-3) #  "Max" aggregation.
        activation= nn.GELU() 
        dropout=0.1
        self.conv1 = conv_block(in_channels*2, in_channels, 3, 1, 1, activation)
        self.conv2 = conv_block(in_channels, in_channels, 3, 1, 1, activation)
        self.conv3 = conv_block(in_channels, out_channels, 3, 1, 1, activation)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        tmp = torch.cat([x_i, x_j], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.conv3(self.conv2(self.conv1(tmp)))
    
    
class Graphmodel(pl.LightningModule):
    def __init__(self,
        pre_model
     ):
        super().__init__()
#         activation= nn.GELU() 
#         self.save_hyperparameters()
        self.edgeconv1 = EdgeConv(32, 32)
        
        # repeat pre_model module
        self.conv0 = pre_model.conv0
        self.conv1_0 = pre_model.conv1_0
        self.conv2_0 = pre_model.conv2_0
        self.conv3_0 = pre_model.conv3_0
        self.conv4_0 = pre_model.conv4_0
        self.conv5_0 = pre_model.conv5_0

        self.conv1_1 = pre_model.conv1_1
        self.conv2_1 = pre_model.conv2_1
        self.conv3_1 = pre_model.conv3_1
        self.conv4_1 = pre_model.conv4_1
        self.conv5_1 = pre_model.conv5_1

        self.deconv0_0 = pre_model.deconv0_0
        self.deconv1_0 = pre_model.deconv1_0
        self.deconv2_0 = pre_model.deconv2_0
        self.deconv3_0 = pre_model.deconv3_0
        self.deconv4_0 = pre_model.deconv4_0
        
        self.deconv0_1 = pre_model.deconv0_1
        self.deconv1_1 = pre_model.deconv1_1
        self.deconv2_1 = pre_model.deconv2_1
        self.deconv3_1 = pre_model.deconv3_1
        self.deconv4_1 = pre_model.deconv4_1
        self.deconv4_2 = pre_model.deconv4_2

        self.batch_norm0 = pre_model.batch_norm0
        self.batch_norm1 = pre_model.batch_norm1
        self.batch_norm2 = pre_model.batch_norm2
        self.batch_norm3 = pre_model.batch_norm3
        self.batch_norm4 = pre_model.batch_norm4
        self.batch_norm5 = pre_model.batch_norm5
        self.batch_norm6 = pre_model.batch_norm6
        self.batch_norm7 = pre_model.batch_norm7
        self.batch_norm8 = pre_model.batch_norm8
        self.batch_norm9 = pre_model.batch_norm9
        self.batch_norm10 = pre_model.batch_norm10

        
        self.gMLPlayers = pre_model.gMLPlayers
        
        self.criterion = pre_model.criterion
        self.loss_weights= pre_model.loss_weights
        
    def forward(self, data):
        
        x, edge_index = data[0], data[2]
        x = np.squeeze(x)
        edge_index=np.squeeze(edge_index)

        x0 = self.conv0(x)
        x0 = self.batch_norm0(x0)
        x1 = self.conv1_1(self.batch_norm1(self.conv1_0(x0)))
        x2 = self.conv2_1(self.batch_norm2(self.conv2_0(x1)))
        x3 = self.conv3_1(self.batch_norm3(self.conv3_0(x2)))
        x4 = self.conv4_1(self.batch_norm4(self.conv4_0(x3)))
        x5 = self.conv5_1(self.batch_norm5(self.conv5_0(x4)))

        x5.transpose_(1, 2)
        x5 = nn.Sequential(*self.gMLPlayers)(x5)
        x5.transpose_(1, 2)
        x5 = self.edgeconv1(x5,edge_index)
 
        x6 = torch.cat((self.batch_norm6(self.deconv0_0(x5)), x4), 1)
        x6 = self.deconv0_1(x6)
  
        x7 = torch.cat((self.batch_norm7(self.deconv1_0(x6)), x3), 1)
        x7 = self.deconv1_1(x7)
        
        x8 = torch.cat((self.batch_norm8(self.deconv2_0(x7)), x2), 1)
        x8 = self.deconv2_1(x8)
        
        x9 = torch.cat((self.batch_norm9(self.deconv3_0(x8)), x1), 1)
        x9 = self.deconv3_1(x9)
        
        x10 = torch.cat((self.batch_norm10(self.deconv4_0(x9)), x0), 1)
        x10 = self.deconv4_1(x10)
        x10 = self.deconv4_2(x10)
        
        
        
        return x10#F.log_softmax(x, dim=1)
    
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        y = batch[1][0]
        y = np.squeeze(y)

        y_hat = self.forward(batch)
#         y_hat2 = y_hat.view(-1,1)
#         y2 = y.view(-1,1)
#         loss = self.criterion(y_hat2, y2)

        y_hatD = y_hat[:,0,:].reshape(-1,1)
        yD = y[:,0,:].reshape(-1,1)
        lossD = self.criterion(y_hatD, yD)

        y_hatP = y_hat[:,1,:].reshape(-1,1)
        yP = y[:,1,:].reshape(-1,1)
        lossP = self.criterion(y_hatP, yP)

        y_hatS = y_hat[:,2,:].reshape(-1,1)
        yS = y[:,2,:].reshape(-1,1)
        lossS = self.criterion(y_hatS, yS)
        
        loss = self.loss_weights[0]*lossD+self.loss_weights[1]*lossP+self.loss_weights[2]*lossS
#         print(loss,lossD,lossP, lossS, (lossD+lossP+lossS)/3)
        # Logging to TensorBoard by default
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx): 
        y = batch[1][0]
        y = np.squeeze(y)

        y_hat = self.forward(batch)
#         y_hat2 = y_hat.view(-1,1)
#         y2 = y.view(-1,1)
#         loss = self.criterion(y_hat2, y2)
        
        y_hatD = y_hat[:,0,:].reshape(-1,1)
        yD = y[:,0,:].reshape(-1,1)
        lossD = self.criterion(y_hatD, yD)

        y_hatP = y_hat[:,1,:].reshape(-1,1)
        yP = y[:,1,:].reshape(-1,1)
        lossP = self.criterion(y_hatP, yP)

        y_hatS = y_hat[:,2,:].reshape(-1,1)
        yS = y[:,2,:].reshape(-1,1)
        lossS = self.criterion(y_hatS, yS)
        
        loss = self.loss_weights[0]*lossD+self.loss_weights[1]*lossP+self.loss_weights[2]*lossS
    
    
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        return {'val_loss': loss}
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "frequency": 1
                       },
        
        }
    