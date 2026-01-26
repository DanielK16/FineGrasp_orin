# FineGrasp_orin
Try to see if FineGrasp works for detecting thin objects

# Environment
Python: 3.10
CUDA: 12.6

# Install FineGrasp
#Create Environment
conda create -n finegrasp python=3.10.12
conda activate finegrasp 

#Installtorch for correct CUDA Version
pip install -U torch torchvision torchaudio --index-url https://pypi.jetson-ai-lab.io/jp6/cu126

# Install spconv and cumm
#Issues that helped
https://github.com/traveller59/spconv/issues/760
https://github.com/traveller59/spconv/issues/726#issuecomment-2605339303

pip3 install pccm pybind11 ninja cmake

#Install cumm from source
git clone https://github.com/FindDefinition/cumm 
cd ./cumm
export CUMM_CUDA_ARCH_LIST="8.7"
export CUMM_DISABLE_JIT="1"
git checkout v0.7.11
pip install -e .

#Install spconv from source
git clone --depth 1 https://github.com/traveller59/spconv
cd spconv
export SPCONV_DISABLE_JIT="1"
python setup.py bdist_wheel
pip install dist/spconv-*.whl

# Check Installation
cumm 0.7.11
spconv 2.3.8
python3 -c "import cumm; print(cumm.__version__)"
python3 -c "import spconv; print(spconv.__version__)"
