source "/itet-stor/${USER}/net_scratch/conda/etc/profile.d/conda.sh"
source "/itet-stor/${USER}/net_scratch/conda/etc/profile.d/mamba.sh"

# GPU="titan_x"
GPU="2080_ti"

# https://github.com/pytorch/pytorch/issues/123097#issuecomment-2055236551
# downgrade mkl to 2023.1
if [[ "$GPU" == "titan_x" ]]; then
    mamba create -y -n neilfpp python=3.10 mkl=2023.1 pytorch=2.0.1 torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
    mamba activate neilfpp
    export TCNN_CUDA_ARCHITECTURES=52
else
    mamba create -y -n neilfpp2080 python=3.10 mkl=2023.1 pytorch=2.0.1 torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
    mamba activate neilfpp2080
    export TCNN_CUDA_ARCHITECTURES=75
fi
mamba install nvidia/label/cuda-11.7.0::cuda-nvcc -y
mamba install cudatoolkit=11.7 cudatoolkit-dev=11.7 -c conda-forge -y
mamba install -y matplotlib numpy pybind11 openexr-python -c conda-forge
pip install --no-input lpips opencv-python open3d tqdm imageio scikit-image scikit-learn trimesh pyexr einops wandb

# tcnn v1.6, please also refer to the official installation guide
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# Install optix SDK
# https://developer.nvidia.com/designworks/optix/downloads/legacy
# sh NVIDIA-OptiX-SDK-7.3.0-linux64-x86_64.sh --include-subdir --skip-license 
export OptiX_INSTALL_DIR=/cluster/${USER}/apps/optix/NVIDIA-OptiX-SDK-7.3.0-linux64-x86_64
export Torch_DIR=$CONDA_PREFIX/lib/python3.10/site-packages/torch/share/cmake/Torch

mamba install -c conda-forge cudnn -y
mamba install cxx-compiler gcc_linux-64=10 -c conda-forge -y

cd code/pyrt
mkdir -p build
cd build
cmake ..
make pyrt

# Need to install optix and rtcore drivers because they are not included in debian by default
# if you run `strace executable`, strace will tell you which drivers it is looking for and where
# https://forums.developer.nvidia.com/t/libnvoptix-so-1-not-installed-by-the-driver/221704/5
# https://wiki.debian.org/NvidiaGraphicsDrivers#OptiX
# Optix 7.3 release notes: 465+ driver req
mamba install nvidia/label/cuda-11.7.0::cuda-tools -y
# cd ~/scratch/apps/optix
# libnvoptix-so
# apt-get download libnvidia-tesla-470-nvoptix1
# libnvidia-rtcore.so
# apt-get download libnvidia-tesla-470-rtcore
# dpkg-deb -x libnvidia-tesla-470-nvoptix1_470.141.03-1~deb11u1~bpo10+1_amd64.deb .
# dpkg-deb -x libnvidia-tesla-470-rtcore_470.141.03-1~deb11u1~bpo10+1_amd64.deb .

# Need to include optix and rtcore in LD_LIBRARY_PATH when running ray tracing code
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cluster/${USER}/apps/optix/usr/lib/x86_64-linux-gnu/nvidia/tesla-470
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cluster/${USER}/apps/optix/usr/lib/x86_64-linux-gnu

