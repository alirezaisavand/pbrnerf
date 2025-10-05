#!/usr/bin/env bash 
# GPU="titan_x" 
# GPU="2080_ti" 
GPU="4090" 
sh NVIDIA-OptiX-SDK-7.3.0-linux64-x86_64.sh --include-subdir --skip-license 
# sh NVIDIA-OptiX-SDK-7.6.0-linux64-x86_64-31894579.sh --include-subdir --skip-license 
module load eth_proxy cmake/3.27.7 cudnn/8.9.7.29-12 
python -m venv --system-site-packages neilfpp_env 
source neilfpp_env/bin/activate 
# export TCNN_CUDA_ARCHITECTURES=75 
export TCNN_CUDA_ARCHITECTURES=80 
pip3 install --upgrade pip 
pip3 install mkl==2023.1.0 
pip3 install --no-input lpips opencv-python open3d tqdm imageio scikit-image scikit-learn trimesh pyexr einops wandb OpenEXR matplotlib numpy pybind11 
pip3 install ninja git+https://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@github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch 
export OptiX_INSTALL_DIR=/cluster/home/${USER}/apps/optix/NVIDIA-OptiX-SDK-7.3.0-linux64-x86_64 
# export OptiX_INSTALL_DIR=/cluster/home/${USER}/apps/optix/NVIDIA-OptiX-SDK-7.6.0-linux64-x86_64 
export Torch_DIR=/cluster/software/stacks/2024-06/spack/opt/spack/linux-ubuntu22.04-x86_64_v3/gcc-12.2.0/python-3.11.6-ukhwpjnwzzzizek3pgr75zkbhxros5fq/lib/python3.11/site-packages/torch/share/cmake/Torch 
export OPENCV_IO_ENABLE_OPENEXR=True 
# Add the following lines in /cluster/home/${USER}/ml-neilfpp/code/pyrt/CMakeLists.txt file 
# set(pybind11_DIR /cluster/home/${USER}/.local/lib/python3.11/site-packages/pybind11/share/cmake/pybind11) 
# find_package(pybind11 REQUIRED) 
# sed -i '100s/^/#/' code/pyrt/CMakeLists.txt 
# echo "find_package(pybind11 REQUIRED)" >> code/pyrt/CMakeLists.txt 
sed -i "100i\set(pybind11_DIR /cluster/home/${USER}/ml-neilfpp/neilfpp_env/lib/python3.11/site-packages/pybind11/share/cmake/pybind11)" code/pyrt/CMakeLists.txt 
# comment out the below line in file: code/pyrt/common/gdt/gdt/gdt.h (line 137) 
# using ::saturate; 
sed -i '137s/^/\/\//' code/pyrt/common/gdt/gdt/gdt.h 
# mkdir -p code/pyrt/build 
# cd code/pyrt/build 
# cmake .. 
# make pyrt 

###############
# ---------- Build repo-local C++ ext (code/pyrt) ----------
if [ -d "code/pyrt" ]; then
  echo "==> Building code/pyrt ..."

  # Map env var name to what CMake expects, and validate headers exist
  if [ -z "${OptiX_INSTALL_DIR:-}" ] && [ -n "${OPTIX_INSTALL_DIR:-}" ]; then
    export OptiX_INSTALL_DIR="${OPTIX_INSTALL_DIR}"
  fi
  if [ ! -f "${OptiX_INSTALL_DIR}/include/optix.h" ]; then
    echo "ERROR: optix.h not found at \$OptiX_INSTALL_DIR/include"
    echo "       Set OPTIX_INSTALL_DIR to your OptiX SDK root (the dir that contains include/optix.h)."
    exit 1
  fi

  # Locate Torch_DIR and pybind11_DIR from the current Python env
  Torch_DIR="$("$PY_BIN" - <<'PY'
import torch, pathlib; print(pathlib.Path(torch.__file__).parent / "share/cmake/Torch")
PY
)"
  pybind11_DIR="$("$PY_BIN" - <<'PY'
import pybind11; print(pybind11.get_cmake_dir())
PY
)"

  # Clean build dir to avoid stale cache that ignores new -D flags
  rm -rf code/pyrt/build
  cmake -S code/pyrt -B code/pyrt/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DTorch_DIR="$Torch_DIR" \
        -Dpybind11_DIR="$pybind11_DIR" \
        -DOptiX_INSTALL_DIR="$OptiX_INSTALL_DIR" \
        -DOPTIX_INSTALL_DIR="$OptiX_INSTALL_DIR" \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5

  cmake --build code/pyrt/build -j "${MAX_JOBS:-4}"
else
  echo "==> No code/pyrt directory; skipping C++ build."
fi

###############


# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cluster/home/${USER}/apps/optix/usr/lib/x86_64-linux-gnu/nvidia/tesla-470 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cluster/home/${USER}/apps/optix/usr/lib/x86_64-linux-gnu