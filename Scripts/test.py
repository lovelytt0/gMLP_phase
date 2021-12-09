from gMLPhase.tester import tester

tester(input_hdf5='/home/tian_feng/UCLA/merge.hdf5',
       input_testset='EQTransformer/ModelsAndSampleData/test.npy',
       input_model='checkpoints/last.ckpt',
       hparams_file = 'default/version_0/hparams.yaml', output_name='/home/tian_feng/UCLA/gMLP_phase/gMLP_phase/test_trainer/test15',
       detection_threshold=0.50,                
       P_threshold=0.3,
       S_threshold=0.3, 
       number_of_plots=128,
       estimate_uncertainty=False, 
       number_of_sampling=1,
       input_dimention=(6000, 3),
       normalization_mode='std',
       mode='generator',
       batch_size=64,
       gpuid=None,
       gpu_limit=None)