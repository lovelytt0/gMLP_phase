
# install PhaseNet
git clone --recurse-submodules   https://github.com/wayneweiqiang/PhaseNet
cd PhaseNet
echo "install phasenet env"
# conda env update -f=env.yml -n base
conda env create -f env.yml -n phasenet
conda activate phasenet
ipython kernel install --user --name=phasenet
conda deactivate

# install Eqtransformer
cd ..
git clone --recurse-submodules git://github.com/smousavi05/EQTransformer
cd EQTransformer
echo "install EQTransformer env"
python setup.py install
conda create -n eqt python=3.7
conda activate eqt
conda install -c smousavi05 eqtransformer
pip install torch, einops, pytorch_lightning, h5py, matplotlib, obspy, pandas, tensorflow
#conda install tensorflow-gpu==2.2.0 
ipython kernel install --user --name=eqt
conda deactivate

# Install pyg

# echo "install pyg env"
# conda create -n pyg python=3.7
# conda install pyg -c pyg -c conda-forge
# ipython kernel install --user --name=pyg
# conda deactivate

# pip install torch-scatter
# pip install torch-sparse
# pip install torch-geometric

# pip install torch-cluster
# pip install torch-spline-conv
