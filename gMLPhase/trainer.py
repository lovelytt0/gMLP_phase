import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import time
from gMLPhase.EqT_utils import DataGenerator
from torch import nn
from gMLPhase.gMLP_torch import gMLPmodel

from torch.utils.data import DataLoader, Dataset
import torch
import torch.optim as optim
import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn.functional as F

from pytorch_lightning import loggers as pl_loggers


def cycle(loader):
    while True:
        for data in loader:
            yield data


def trainer(input_hdf5=None,
            input_trainset=None,
            output_name=None,                
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
#     print(training,validation)
    model=_build_model(args)
#     model.cuda()
    print(model)

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
                          'drop_channe_r': args['drop_channel_r'],
                          'scale_amplitude_r': args['scale_amplitude_r'],
                          'pre_emphasis': args['pre_emphasis']}

        params_validation = {'file_name': str(args['input_hdf5']),  
                             'dim': args['input_dimention'][0],
                             'batch_size': args['batch_size'],
                             'n_channels': args['input_dimention'][-1],
                             'shuffle': False,  
                             'norm_mode': args['normalization_mode'],
                             'label_type': args['label_type'],
                             'augmentation': False,
                             'coda_ratio': args['coda_ratio']}         

        training_generator = DataGenerator(training, **params_training)
        validation_generator = DataGenerator(validation, **params_validation) 
        checkpoint_callback = ModelCheckpoint(monitor=monitor,dirpath=save_models,save_top_k=3,verbose=True,save_last=True)
        early_stopping = EarlyStopping(monitor=monitor,patience=args['patience']) # patience=3
        tb_logger = pl_loggers.TensorBoardLogger(save_dir)
        
        
        trainer = pl.Trainer(precision=16, gpus=1,callbacks=[early_stopping, checkpoint_callback],check_val_every_n_epoch=1,profiler="simple",num_sanity_val_steps=0, logger =tb_logger)


        
                
        train_loader  = DataLoader(training_generator, batch_size = 1, num_workers=8, pin_memory=True, prefetch_factor=5)
#         print(next(train_loader))
        val_loader    = DataLoader(validation_generator, batch_size =1 , num_workers=8, pin_memory=True, prefetch_factor=5)
#         print(next(train_loader))

        print('Started training in generator mode ...') 

        trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = val_loader)
                    
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
        shutil.copyfile('gMLPhase/gMLP_torch.py',os.path.join(save_dir,'gMLP_torch.py'))
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


    model = gMLPmodel(
#         input_dim = args['input_dimention'][-1],
#         dim = args['gmlp_dim'],
#         depth = args['gmlp_blocks'],
#         seq_len = args['seq_len'],
#         activation = args['activation'],
        dropout = args['drop_rate'],
        loss_types = args['loss_types'],
        loss_weights = args['loss_weights']
    )

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
    
    ev_list = np.load(args['input_trainset'])
#     ev_list = df.trace_name.tolist()    
    np.random.shuffle(ev_list)     
    training = ev_list[:int(args['train_valid_test_split'][0]*len(ev_list))]
    validation =  ev_list[int(args['train_valid_test_split'][0]*len(ev_list)):
                            int(args['train_valid_test_split'][0]*len(ev_list) + args['train_valid_test_split'][1]*len(ev_list))]
#     test =  ev_list[ int(args['train_valid_test_split'][0]*len(ev_list) + args['train_valid_test_split'][1]*len(ev_list)):]
    np.save(os.path.join(save_dir,'training.npy'), training)  
    np.save(os.path.join(save_dir,'validation.npy'), validation)  

    return training, validation 



