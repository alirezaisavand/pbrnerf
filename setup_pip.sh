#!/usr/bin/env bash
# Robust, idempotent setup for PBR-NeRF inside your CUDA 11.8 + PyTorch 2.2 container.
# - No conda/venv: installs into the container's active Python.
# - Targets A100 (sm_80) for CUDA/C++ extensions (tiny-cuda-nn etc.)
# - Works whether or not OptiX is mounted/baked. Warns if missing.

set -euo pipefail

echo "==> PBR-NeRF setup (Docker, CUDA 11.8, A100) starting..."

# -------------------------------
# 0) Sanity: show Python & Torch
# -------------------------------
PY_BIN="${PY_BIN:-python3}"
echo "==> Python executable: $($PY_BIN -c 'import sys; print(sys.executable)')"
$PY_BIN - <<'PY'
import sys, torch
print("==> Python version:", sys.version.split()[0])
print("==> Torch version:", torch.__version__)
print("==> CUDA available:", torch.cuda.is_available())
PY

# ---------------------------------------------
# 1) CUDA build env for A100 + compile settings
# ---------------------------------------------
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"    # A100 (sm_80)
export TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES:-80}"
export CMAKE_ARGS="${CMAKE_ARGS:-"-DTCNN_CUDA_ARCHITECTURES=80 -DCMAKE_CUDA_ARCHITECTURES=80"}"
export MAX_JOBS="${MAX_JOBS:-4}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"

echo "==> Build settings:"
echo "    TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "    TCNN_CUDA_ARCHITECTURES=$TCNN_CUDA_ARCHITECTURES"
echo "    CMAKE_ARGS=$CMAKE_ARGS"
echo "    MAX_JOBS=$MAX_JOBS"
echo "    CUDA_HOME=$CUDA_HOME"

# ------------------------------------------------------
# 2) Core Python tooling & libs (safe if rerun multiple)
# ------------------------------------------------------
$PY_BIN -m pip install -U pip wheel setuptools
$PY_BIN -m pip install -U ninja cmake pybind11 packaging
$PY_BIN -m pip install -U \
  tqdm opencv-python-headless matplotlib scikit-image scipy \
  einops loguru yacs tensorboard imageio imageio-ffmpeg \
  plyfile trimesh lpips

# (Optional) Open3D — uncomment if you need it:
# apt libs are already in Dockerfile; this often works headless.
# $PY_BIN -m pip install -U open3d==0.18.0 || echo "Open3D install skipped/failed; continue."

# --------------------------------------------
# 3) Locate Torch/pybind11 CMake config paths
# --------------------------------------------
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

# ----------------------------------------------------
# 4) tiny-cuda-nn (PyTorch bindings) for A100 (sm_80)
# ----------------------------------------------------
echo "==> Installing tiny-cuda-nn (PyTorch bindings) for sm_80..."
set +e
$PY_BIN -m pip install -v --no-build-isolation \
  "git+https://github.com/NVlabs/tiny-cuda-nn.git@v1.7#subdirectory=bindings/torch"
TCNN_RC=$?
set -e

if [ $TCNN_RC -ne 0 ]; then
  echo "!! git+pip install failed; falling back to tarball of master ..."
  TMP_TAR="$(mktemp -u /tmp/tcnn.XXXX).tar.gz"
  curl -L https://codeload.github.com/NVlabs/tiny-cuda-nn/tar.gz/refs/heads/master -o "$TMP_TAR"
  mkdir -p /tmp/tcnn && tar -xzf "$TMP_TAR" --strip-components=1 -C /tmp/tcnn
  $PY_BIN -m pip install -v --no-build-isolation /tmp/tcnn/bindings/torch
  rm -rf "$TMP_TAR" /tmp/tcnn
fi

# Smoke test tinycudann
$PY_BIN - <<'PY'
import tinycudann as tcnn
print("==> tinycudann OK @", tcnn.__file__)
PY

# --------------------------------------------
# 5) (Optional) OptiX presence check & warning
# --------------------------------------------
if [ -z "${OPTIX_INSTALL_DIR:-}" ] || [ ! -d "${OPTIX_INSTALL_DIR:-/nonexistent}" ]; then
  echo "!! OPTIX_INSTALL_DIR is not set or missing. If any module needs OptiX, mount or bake it."
else
  echo "==> Found OptiX at: $OPTIX_INSTALL_DIR"
  export LD_LIBRARY_PATH="${OPTIX_INSTALL_DIR}/lib64:${LD_LIBRARY_PATH:-}"
fi

# --------------------------------------------------------
# 6) Build repo-local C++/CUDA (e.g., code/pyrt) if exists
# --------------------------------------------------------
if [ -d "code/pyrt" ]; then
  echo "==> Preparing to build code/pyrt ..."
  # Some forks require commenting out 'using ::saturate;' in gdt.h (harmless if absent).
  if [ -f "code/pyrt/common/gdt/gdt/gdt.h" ]; then
    sed -i 's/^[[:space:]]*using[[:space:]]*::saturate;/\/\/ using ::saturate;/' code/pyrt/common/gdt/gdt/gdt.h || true
  fi

  if [ -f "code/pyrt/setup.py" ] || [ -f "code/pyrt/pyproject.toml" ]; then
    echo "==> Installing code/pyrt via pip (editable) ..."
    (cd code/pyrt && $PY_BIN -m pip install -v -e .)
  elif [ -f "code/pyrt/CMakeLists.txt" ]; then
    echo "==> Building code/pyrt via CMake + Ninja ..."
    mkdir -p code/pyrt/build && cd code/pyrt/build
    cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DTorch_DIR="$Torch_DIR" -Dpybind11_DIR="$pybind11_DIR"
    cmake --build . -j "${MAX_JOBS}"
    cd ../../..
  else
    echo "!! code/pyrt not recognized (no setup.py/pyproject/CMakeLists). Skipping."
  fi
else
  echo "==> No code/pyrt directory; skipping C++ local build."
fi

# --------------------------------
# 7) Final smoke tests & summary
# --------------------------------
$PY_BIN - <<'PY'
import torch, tinycudann
print("==> CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU-only")
print("==> All good. Ready to run PBR-NeRF.")
PY

echo "==> Setup finished."
