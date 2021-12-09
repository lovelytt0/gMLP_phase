from gMLPhase.trainer import trainer
import torch.nn.functional as F
from torch import nn
# csv_file = "../../MLData/metadata_11_13_19.csv"
# file_name = "../../MLData/waveforms_11_13_19.hdf5"

# csv_file = 'EQTransformer/ModelsAndSampleData/100samples.hdf5'
# file_name = 'EQTransformer/ModelsAndSampleData/100samples.csv'
trainer(input_hdf5='/home/tian_feng/UCLA/merge.hdf5',
        input_trainset= 'train.npy',
        output_name='test_trainer/test15',                
        activation=nn.GELU(),
        drop_rate = 0.1,
        augmentation = True,
        label_type='triangle',
        add_event_r=0.3,
        add_noise_r=0.5, 
        shift_event_r=0.99,
        add_gap_r=0.2,
        drop_channel_r = 0.3,
        coda_ratio=1.4,
        normalization_mode = 'std',
        mode='generator',
        loss_types = F.binary_cross_entropy_with_logits, 
        train_valid_test_split=[0.95, 0.05],
        batch_size = 128,
        epochs=50,
        monitor='val_loss',
        loss_weights=[0.05, 0.35, 0.60],
        patience=10)

