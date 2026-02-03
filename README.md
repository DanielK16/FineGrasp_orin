#  FineGrasp Jetson Orin (CUDA 12.6)
This repository documents the specific build process for FineGrasp on NVIDIA Jetson Orin hardware. It addresses the compatibility challenges of compiling spconv, cumm, and MinkowskiEngine on aarch64 architecture with CUDA 12.6.
---

##  System Specifications
* **OS:** Ubuntu 22.04 (JetPack 6)
* **Python:** 3.10.12
* **CUDA:** 12.6
* **Architecture:** `aarch64` (Compute Capability 8.7)
* **Core Libraries:** `cumm 0.7.11`, `spconv 2.3.8`, `MinkowskiEngine 0.5.4`

## Credits Paper
https://arxiv.org/pdf/2507.05978

---

## 0. Final folder structure:
<img width="968" height="539" alt="image" src="https://github.com/user-attachments/assets/33fcf311-6847-4990-9e70-099d87c51128" />


## 1. Environment Setup
```bash
# Create Conda Environment
conda create -n finegrasp python=3.10.12 -y
conda activate finegrasp 

# Install PyTorch for JetPack 6 / CUDA 12.6
pip install -U torch torchvision torchaudio --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

## 2. Cumm & Spconv
These libaries must be build from source:
- https://github.com/traveller59/spconv
- https://github.com/FindDefinition/cumm
```bash
pip3 install pccm pybind11 ninja cmake

# Install cumm (v0.7.11)
git clone https://github.com/FindDefinition/cumm
cd cumm
git checkout v0.7.11
export CUMM_CUDA_ARCH_LIST="8.7"
export CUMM_DISABLE_JIT="1"
pip install -e .
cd ..

# Install spconv (v2.3.8)
git clone --depth 1 https://github.com/traveller59/spconv
cd spconv
export SPCONV_DISABLE_JIT="1"
python setup.py bdist_wheel
pip install dist/spconv-*.whl
cd ..
```

## 3. FineGrasp & GraspNet API
- https://github.com/HorizonRobotics/RoboOrchardLab/tree/master/projects/finegrasp_graspnet1b
- https://github.com/graspnet/graspnetAPI
```bash
# Install Robo Orchard Lab (FineGrasp)
git clone https://github.com/HorizonRobotics/robo_orchard_lab.git
cd robo_orchard_lab && make version
pip3 install .[finegrasp]
pip install transformers
cd ..

# Install GraspNet API
git clone https://github.com/graspnet/graspnetAPI.git && cd graspnetAPI
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
# Libstdc++ Fix to prevent errors
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
pip3 install .
pip3 install transforms3d==0.4.2 numpy==1.26.4 IPython
```

## 4. MinkowskiEngine
Following files was patched inside MinkowskiEngine replace these files:
- src/3rdparty/concurrent_unordered_map.cuh
- src/convolution_kernel.cuh
- src/coordinate_map_gpu.cu
- setup.py
```bash
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
# [OVERWRITE THE FILES ABOVE WITH PATCHED VERSIONS]
sudo sed -i 's/\bauto __raw = __to_address(__r.get());/auto __raw = std::__to_address(__r.get());/' /usr/include/c++/11/bits/shared_ptr_base.h
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

## 5. Install PointNet2 and KNN (Scale-Balanced-Grasp)
Following files was patched inside Scale-Balanced-Grasp replace these files:
- src/cuda/vision.h
- src/knn.h
```bash
git clone https://github.com/mahaoxiang822/Scale-Balanced-Grasp
cd Scale-Balanced-Grasp/pointnet2
# [OVERWRITE THE FILES ABOVE WITH PATCHED VERSIONS]

# Install PointNet2
cd pointnet2
export MAX_JOBS=2
export TORCH_CUDA_ARCH_LIST="8.7"
export CUDA_HOME="/usr/local/cuda"
export CC=gcc CXX=g++ NVCC_CCBIN=g++
python setup.py install --verbose
cd ..

# Install knn
python setup.py install --verbose
cd ../..
```

## Helpful Ressources & Issues for Instalation
- https://github.com/traveller59/spconv/issues/760
- https://github.com/traveller59/spconv/issues/726#issuecomment-2605339303
- https://github.com/NVIDIA/MinkowskiEngine/issues/543

## Download checkpoint 
https://huggingface.co/HorizonRobotics/FineGrasp
Place Weights: Move the file to scripts/config/model.safetensors.

## Results using my_infer.py
<img width="902" height="327" alt="Screenshot from 2026-01-26 15-54-49" src="https://github.com/user-attachments/assets/9ba81126-c9dc-44b4-bdd0-2a0909c6efe6" />

<img width="1035" height="674" alt="Screenshot from 2026-01-26 15-55-49" src="https://github.com/user-attachments/assets/8916f275-e97f-4478-a5db-2ec2e46a4205" />

<img width="1223" height="705" alt="Screenshot from 2026-01-26 15-57-19" src="https://github.com/user-attachments/assets/d2a81cd6-a546-4f37-a520-1a220cb939bb" />


