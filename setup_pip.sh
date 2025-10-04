#!/usr/bin/env bash
set -euo pipefail

# Enforce the same constraints during setup
: "${PIP_CONSTRAINT:=/tmp/constraints.txt}"
if [ ! -f "$PIP_CONSTRAINT" ]; then
  echo -e "numpy<2\nsetuptools<81\npybind11>=2.12" > "$PIP_CONSTRAINT"
fi
export PIP_CONSTRAINT

# Toolchain + pin BEFORE any native builds
$PY_BIN -m pip install -U "setuptools<81" "numpy<2" "pybind11>=2.12" wheel ninja cmake packaging

echo "==> PBR-NeRF setup (Docker, CUDA 11.8, A100)"

PY_BIN="${PY_BIN:-python3}"
echo "==> Python: $($PY_BIN -c 'import sys; print(sys.executable)')"

$PY_BIN - <<'PY'
import sys, torch
print("==> Python version:", sys.version.split()[0])
print("==> Torch version:", torch.__version__)
print("==> CUDA available:", torch.cuda.is_available())
PY

# ---- Build env: A100 (sm_80) ----
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES:-80}"
export CMAKE_ARGS="${CMAKE_ARGS:-"-DTCNN_CUDA_ARCHITECTURES=80 -DCMAKE_CUDA_ARCHITECTURES=80"}"
export MAX_JOBS="${MAX_JOBS:-4}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"

echo "==> Build settings:"
echo "    TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "    TCNN_CUDA_ARCHITECTURES=$TCNN_CUDA_ARCHITECTURES"
echo "    CMAKE_ARGS=$CMAKE_ARGS"

# ---- Pin numpy < 2 and ensure pybind11 >= 2.12 BEFORE any native builds ----
$PY_BIN -m pip install -U "setuptools<81" "numpy<2" "pybind11>=2.12" wheel ninja cmake packaging

# Core python deps (these won't bump numpy above <2)
$PY_BIN -m pip install -U \
  tqdm opencv-python-headless matplotlib scikit-image scipy \
  einops loguru yacs tensorboard imageio imageio-ffmpeg \
  plyfile trimesh lpips

# (Optional) Open3D if you need it:
# $PY_BIN -m pip install -U open3d==0.18.0 || echo "Open3D skipped."

# Locate CMake packages
export Torch_DIR="$($PY_BIN - <<'PY'
import torch, pathlib
print(pathlib.Path(torch.__file__).parent / "share/cmake/Torch")
PY
)"
export pybind11_DIR="$($PY_BIN - <<'PY'
import pybind11
print(pybind11.get_cmake_dir())
PY
)"
echo "==> Torch_DIR=$Torch_DIR"
echo "==> pybind11_DIR=$pybind11_DIR"

# ---- tiny-cuda-nn (PyTorch bindings) ----
echo "==> Installing tiny-cuda-nn (PyTorch bindings) for sm_80..."
set +e
$PY_BIN -m pip install -v --no-build-isolation \
  "git+https://github.com/NVlabs/tiny-cuda-nn.git@v1.7#subdirectory=bindings/torch"
TCNN_RC=$?
set -e
if [ $TCNN_RC -ne 0 ]; then
  echo "!! git+pip failed; retry via tarball..."
  TMP_TAR="$(mktemp -u /tmp/tcnn.XXXX).tar.gz"
  curl -L https://codeload.github.com/NVlabs/tiny-cuda-nn/tar.gz/refs/heads/master -o "$TMP_TAR"
  mkdir -p /tmp/tcnn && tar -xzf "$TMP_TAR" --strip-components=1 -C /tmp/tcnn
  $PY_BIN -m pip install -v --no-build-isolation /tmp/tcnn/bindings/torch
  rm -rf "$TMP_TAR" /tmp/tcnn
fi

# Smoke test
$PY_BIN - <<'PY'
import numpy, tinycudann as tcnn
print("==> NumPy:", numpy.__version__)
print("==> tinycudann OK @", tcnn.__file__)
PY

# ---- OptiX (optional) ----
if [ -n "${OPTIX_INSTALL_DIR:-}" ] && [ -d "$OPTIX_INSTALL_DIR" ]; then
  export LD_LIBRARY_PATH="${OPTIX_INSTALL_DIR}/lib64:${LD_LIBRARY_PATH:-}"
  echo "==> OptiX at $OPTIX_INSTALL_DIR"
else
  echo "!! OPTIX_INSTALL_DIR not set/found; continue."
fi

# ---- Build local C++ ext (code/pyrt) if present ----
if [ -d "code/pyrt" ]; then
  echo "==> Preparing to build code/pyrt ..."
  if [ -f "code/pyrt/common/gdt/gdt/gdt.h" ]; then
    sed -i 's/^[[:space:]]*using[[:space:]]*::saturate;/\/\/ using ::saturate;/' code/pyrt/common/gdt/gdt/gdt.h || true
  fi

  if [ -f "code/pyrt/setup.py" ] || [ -f "code/pyrt/pyproject.toml" ]; then
    (cd code/pyrt && $PY_BIN -m pip install -v -e .)
  elif [ -f "code/pyrt/CMakeLists.txt" ]; then
    mkdir -p code/pyrt/build && cd code/pyrt/build
    cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DTorch_DIR="$Torch_DIR" -Dpybind11_DIR="$pybind11_DIR"
    cmake --build . -j "${MAX_JOBS}"
    cd ../../..
  else
    echo "==> No recognizable build system in code/pyrt; skipping."
  fi
else
  echo "==> No code/pyrt directory; skipping."
fi

$PY_BIN - <<'PY'
import torch
print("==> CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU-only")
print("==> Setup complete.")
PY