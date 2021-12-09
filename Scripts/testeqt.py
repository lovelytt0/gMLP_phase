from EQTransformer.core.tester import tester
tester(input_hdf5='/home/tian_feng/UCLA/merge.hdf5',
       input_testset='EQTransformer/ModelsAndSampleData/test.npy',
       input_model='EQTransformer/ModelsAndSampleData/EqT_model2.h5',
       output_name='eqt_results',
       detection_threshold=0.20,                
       P_threshold=0.1,
       S_threshold=0.1, 
       number_of_plots=100,
       estimate_uncertainty=True, 
       number_of_sampling=2,
       input_dimention=(6000, 3),
       normalization_mode='std',
       mode='generator',
       batch_size=10,
       gpuid=None,
       gpu_limit=None)