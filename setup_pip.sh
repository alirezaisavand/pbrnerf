#!/usr/bin/env bash
set -euo pipefail

# -------- defaults (define BEFORE first use) --------
PY_BIN="${PY_BIN:-python3}"
PIP_CONSTRAINT="${PIP_CONSTRAINT:-/tmp/constraints.txt}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"   # A100 = sm_80
TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES:-80}"
CMAKE_ARGS="${CMAKE_ARGS:-"-DTCNN_CUDA_ARCHITECTURES=80 -DCMAKE_CUDA_ARCHITECTURES=80"}"
MAX_JOBS="${MAX_JOBS:-4}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"
OPTIX_INSTALL_DIR="${OPTIX_INSTALL_DIR:-}"

export TORCH_CUDA_ARCH_LIST TCNN_CUDA_ARCHITECTURES CMAKE_ARGS MAX_JOBS CUDA_HOME CMAKE_BUILD_PARALLEL_LEVEL

echo "==> PBR-NeRF setup (Docker, CUDA 11.8, A100)"
echo "==> Using Python: $("$PY_BIN" -c 'import sys; print(sys.executable)')"

# -------- enforce constraints globally (numpy<2 etc.) --------
if [ ! -f "$PIP_CONSTRAINT" ]; then
  printf "numpy<2\nsetuptools<81\npybind11>=2.12\n" > "$PIP_CONSTRAINT"
fi
export PIP_CONSTRAINT

# ---- show torch info ----
"$PY_BIN" - <<'PY'
import sys, torch
print("==> Python version:", sys.version.split()[0])
print("==> Torch version:", torch.__version__)
print("==> CUDA available:", torch.cuda.is_available())
PY

# ---- toolchain (respects constraints) ----
"$PY_BIN" -m pip install -U pip wheel ninja cmake packaging "setuptools<81" "numpy<2" "pybind11>=2.12"

# ---- common python deps ----
"$PY_BIN" -m pip install -U \
  tqdm opencv-python-headless matplotlib scikit-image scipy \
  einops loguru yacs tensorboard imageio imageio-ffmpeg \
  plyfile trimesh lpips

# ---- locate CMake packages for Torch / pybind11 ----
Torch_DIR="$("$PY_BIN" - <<'PY'
import torch, pathlib; print(pathlib.Path(torch.__file__).parent / "share/cmake/Torch")
PY
)"
pybind11_DIR="$("$PY_BIN" - <<'PY'
import pybind11; print(pybind11.get_cmake_dir())
PY
)"
export Torch_DIR pybind11_DIR
echo "==> Torch_DIR=$Torch_DIR"
echo "==> pybind11_DIR=$pybind11_DIR"

# ---- tiny-cuda-nn (PyTorch bindings) for A100 (sm_80) ----
echo "==> Installing tiny-cuda-nn (PyTorch bindings) from recursive clone..."
set -e
TCNN_DIR="/opt/tcnn"
rm -rf "$TCNN_DIR"
git clone --recursive https://github.com/NVlabs/tiny-cuda-nn.git "$TCNN_DIR"
( cd "$TCNN_DIR" && git checkout v1.6 || true && git submodule update --init --recursive )

# Build with proper arches and pinned toolchain
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export CMAKE_ARGS="${CMAKE_ARGS:-"-DTCNN_CUDA_ARCHITECTURES=80 -DCMAKE_CUDA_ARCHITECTURES=80"}"
export MAX_JOBS="${MAX_JOBS:-4}"
"$PY_BIN" -m pip install -v --no-build-isolation "$TCNN_DIR/bindings/torch"


# smoke test (and verify numpy stayed <2)
"$PY_BIN" - <<'PY'
import numpy, tinycudann as tcnn
print("==> NumPy:", numpy.__version__)
print("==> tinycudann @", tcnn.__file__)
PY

# ---- OptiX (optional) ----
if [ -n "$OPTIX_INSTALL_DIR" ] && [ -d "$OPTIX_INSTALL_DIR" ]; then
  export LD_LIBRARY_PATH="${OPTIX_INSTALL_DIR}/lib64:${LD_LIBRARY_PATH:-}"
  echo "==> OptiX at $OPTIX_INSTALL_DIR"
else
  echo "!! OPTIX_INSTALL_DIR not set/found; continuing without it."
fi

# ---- build local C++ extension (code/pyrt) if present ----
if [ -d "code/pyrt" ]; then
  echo "==> Preparing to build code/pyrt ..."
  if [ -f "code/pyrt/common/gdt/gdt/gdt.h" ]; then
    sed -i 's/^[[:space:]]*using[[:space:]]*::saturate;/\/\/ using ::saturate;/' code/pyrt/common/gdt/gdt/gdt.h || true
  fi

  if [ -f "code/pyrt/setup.py" ] || [ -f "code/pyrt/pyproject.toml" ]; then
    (cd code/pyrt && "$PY_BIN" -m pip install -v -e .)
  elif [ -f "code/pyrt/CMakeLists.txt" ]; then
    mkdir -p code/pyrt/build && cd code/pyrt/build
    cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DTorch_DIR="$Torch_DIR" -Dpybind11_DIR="$pybind11_DIR"
    cmake --build . -j "$MAX_JOBS"
    cd ../../..
  else
    echo "==> code/pyrt not recognized; skipping."
  fi
else
  echo "==> No code/pyrt directory; skipping C++ local build."
fi

# ---- final sanity ----
"$PY_BIN" - <<'PY'
import torch
print("==> CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU-only")
print("==> Setup complete.")
PY
