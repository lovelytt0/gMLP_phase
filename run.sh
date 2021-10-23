
# install PhaseNet
git clone --recurse-submodules   https://github.com/wayneweiqiang/PhaseNet
cd PhaseNet
echo "install phasenet env"
conda env update -f=env.yml -n base
conda env create -f env.yml
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
#conda install tensorflow-gpu==2.2.0 
ipython kernel install --user --name=eqtransformer
conda deactivate
