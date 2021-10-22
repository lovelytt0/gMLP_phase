
# install PhaseNet
git clone --recurse-submodules   https://github.com/wayneweiqiang/PhaseNet
cd PhaseNet
echo "install phasenet env"
conda env update -f=env.yml -n base
conda env create -f env.yml

# install Eqtransformer
cd ..
git clone --recurse-submodules git://github.com/smousavi05/EQTransformer
cd EQTransformer
echo "install EQTransformer env"
python setup.py install



