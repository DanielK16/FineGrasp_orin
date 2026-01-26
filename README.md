# 🦾 FineGrasp Jetson Orin (CUDA 12.6)

This repository provides a comprehensive guide for installing and configuring **FineGrasp** on an NVIDIA Jetson Orin running **JetPack 6**. This setup is specifically optimized for detecting thin objects (such as chopsticks) using **CUDA 12.6** and **Python 3.10**.

---

## 📋 System Specifications
* **OS:** Ubuntu 22.04 (JetPack 6)
* **Python:** 3.10.12
* **CUDA:** 12.6
* **Architecture:** `aarch64` (Compute Capability 8.7)
* **Core Libraries:** `cumm 0.7.11`, `spconv 2.3.8`, `MinkowskiEngine 0.5.4`

---

## 1. Environment Setup
```bash
# Create Conda Environment
conda create -n finegrasp python=3.10.12 -y
conda activate finegrasp 

# Install PyTorch for JetPack 6 / CUDA 12.6
pip install -U torch torchvision torchaudio --index-url [https://pypi.jetson-ai-lab.io/jp6/cu126](https://pypi.jetson-ai-lab.io/jp6/cu126)
```

## 2. Cumm & Spconv
These libaries must be build from source 
```bash
pip3 install pccm pybind11 ninja cmake

# Install cumm (v0.7.11)
git clone [https://github.com/FindDefinition/cumm](https://github.com/FindDefinition/cumm) && cd cumm
git checkout v0.7.11
export CUMM_CUDA_ARCH_LIST="8.7"
export CUMM_DISABLE_JIT="1"
pip install -e .
cd ..

# Install spconv (v2.3.8)
git clone --depth 1 [https://github.com/traveller59/spconv](https://github.com/traveller59/spconv) && cd spconv
export SPCONV_DISABLE_JIT="1"
python setup.py bdist_wheel
pip install dist/spconv-*.whl
cd ..
```

## 3. FineGrasp & GraspNet API
```bash
# Install Robo Orchard Lab (FineGrasp)
git clone [https://github.com/HorizonRobotics/robo_orchard_lab.git](https://github.com/HorizonRobotics/robo_orchard_lab.git)
cd robo_orchard_lab && make version
pip3 install .[finegrasp]
pip install transformers
cd ..

# Install GraspNet API
git clone [https://github.com/graspnet/graspnetAPI.git](https://github.com/graspnet/graspnetAPI.git) && cd graspnetAPI
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
pip3 install .
pip3 install transforms3d==0.4.2 numpy==1.26.4 IPython

# Libstdc++ Fix for Conda Environment
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

## 4. MinkowskiEngine
Following files was patched inside MinkowskiEngine replace these files:
- src/3rdparty/concurrent_unordered_map.cuh
- src/convolution_kernel.cuh
- src/coordinate_map_gpu.cu
- setup.py
```bash
git clone [https://github.com/NVIDIA/MinkowskiEngine.git](https://github.com/NVIDIA/MinkowskiEngine.git) && cd MinkowskiEngine

# [OVERWRITE THE FILES ABOVE WITH YOUR PATCHED VERSIONS NOW]

sudo apt-get install -y libopenblas-dev
export CC=gcc CXX=g++ MAX_JOBS=1 # MAX_JOBS=1 prevents RAM exhaustion on Jetson
export TORCH_CUDA_ARCH_LIST="8.7"
export BLAS=openblas
export CUDA_HOME="/usr/local/cuda"
export NVCC_PREPEND_FLAGS='-std=c++17 -DNVTX_DISABLE -DCUB_SKIP_NVTX_CHECK'
export CXXFLAGS='-std=c++17 -DNVTX_DISABLE'

python3 setup.py install --blas=openblas --force_cuda
cd ..
```

## 5. Install PointNet2 and KNN
```bash
git clone [https://github.com/mahaoxiang822/Scale-Balanced-Grasp](https://github.com/mahaoxiang822/Scale-Balanced-Grasp) && cd Scale-Balanced-Grasp

# 1. Pointnet2 Installation
cd pointnet2
export MAX_JOBS=2
export TORCH_CUDA_ARCH_LIST="8.7"
export CUDA_HOME="/usr/local/cuda"
export CC=gcc CXX=g++ NVCC_CCBIN=g++
python setup.py install --verbose
cd ..

# 2. KNN Installation
# IMPORTANT: Replace 'src/cuda/vision.h' and 'src/knn.h' with your patched versions!
cd knn
python setup.py install --verbose
cd ../..
```
