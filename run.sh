
# install PhaseNet
git clone --recurse-submodules   https://github.com/wayneweiqiang/PhaseNet
cd PhaseNet
conda env update -f=env.yml -n base
conda env create -f env.yml

# install Eqtransformer
git clone --recurse-submodules git://github.com/smousavi05/EQTransformer
python setup.py install



