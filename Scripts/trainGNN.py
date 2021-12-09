from EdgeConv.trainer import trainer
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data

training_generator = trainer(input_hdf5='/home/tian_feng/UCLA/merge.hdf5',
        input_trainset= 'multitraces.npy',
        output_name='GNN/test3',      
        input_model='checkpoints/epoch=4-step=84539.ckpt',
        hparams_file = 'default/version_0/hparams.yaml', 
        model_folder='/home/tian_feng/UCLA/gMLP_phase/gMLP_phase/test_trainer/test15',
        augmentation = False,
        label_type='triangle',
        shift_event_r=None,
        mode='generator',
        loss_types = F.binary_cross_entropy_with_logits, 
        train_valid_test_split=[0.90, 0.10],
        batch_size = 1,
        epochs=50,
        monitor='val_loss',
        loss_weights=[0.05, 0.35, 0.60],
        patience=10)