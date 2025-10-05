#!/usr/bin/env bash
set -euo pipefail

# ---------- defaults ----------
PY_BIN="${PY_BIN:-python3}"
PIP_CONSTRAINT="${PIP_CONSTRAINT:-/tmp/constraints.txt}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"           # A100 (sm_80)
TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES:-80}"
CMAKE_ARGS="${CMAKE_ARGS:-"-DCMAKE_CXX_STANDARD=17 -DCMAKE_CUDA_STANDARD=17 -DTCNN_CUDA_ARCHITECTURES=80 -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_POLICY_VERSION_MINIMUM=3.5"}"
MAX_JOBS="${MAX_JOBS:-4}"
OPTIX_INSTALL_DIR="${OPTIX_INSTALL_DIR:-}"                     # set via docker run if you mount OptiX
export TORCH_CUDA_ARCH_LIST TCNN_CUDA_ARCHITECTURES CMAKE_ARGS MAX_JOBS OPTIX_INSTALL_DIR

# Enforce constraints (NumPy < 2 etc.)
if [ ! -f "$PIP_CONSTRAINT" ]; then
  printf "numpy<2\nsetuptools<81\npybind11>=2.12\n" > "$PIP_CONSTRAINT"
fi
export PIP_CONSTRAINT

echo "==> Python: $("$PY_BIN" -c 'import sys; print(sys.executable)')"
"$PY_BIN" - <<'PY'
import sys, torch
print("==> Python:", sys.version.split()[0])
print("==> Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("==> CUDA available:", torch.cuda.is_available())
PY

# ---------- base Python deps ----------
"$PY_BIN" -m pip install -U pip wheel ninja cmake packaging \
  tqdm opencv-python-headless imageio imageio-ffmpeg scikit-image scikit-learn \
  matplotlib einops loguru yacs tensorboard plyfile trimesh lpips

# If you truly need EXR via OpenCV:
export OPENCV_IO_ENABLE_OPENEXR=True
# (Optional) EXR libs:
# "$PY_BIN" -m pip install pyexr  # Avoid 'OpenEXR' unless you add system libs.

# ---------- tiny-cuda-nn (PyTorch bindings), with submodules ----------
echo "==> Installing tiny-cuda-nn (PyTorch bindings) for sm_80 ..."
TCNN_DIR="/opt/tcnn"
rm -rf "$TCNN_DIR"
git clone --recursive https://github.com/NVlabs/tiny-cuda-nn.git "$TCNN_DIR"
( cd "$TCNN_DIR" && git checkout v1.6 || true && git submodule update --init --recursive )
"$PY_BIN" -m pip install -v --no-build-isolation "$TCNN_DIR/bindings/torch"

# ---------- locate CMake packages for Torch / pybind11 ----------
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

# ---------- OptiX (optional) ----------
if [ -n "$OPTIX_INSTALL_DIR" ] && [ -d "$OPTIX_INSTALL_DIR" ]; then
  export LD_LIBRARY_PATH="${OPTIX_INSTALL_DIR}/lib64:${LD_LIBRARY_PATH:-}"
  echo "==> OptiX at $OPTIX_INSTALL_DIR"
else
  echo "!! OPTIX_INSTALL_DIR not set/found; continuing without OptiX."
fi

# ---------- Build repo-local C++ ext (code/pyrt) ----------
if [ -d "code/pyrt" ]; then
  echo "==> Preparing to build code/pyrt ..."
  # Some forks require commenting out 'using ::saturate;' in gdt.h; do it safely if present:
  if [ -f "code/pyrt/common/gdt/gdt/gdt.h" ]; then
    sed -i 's/^[[:space:]]*using[[:space:]]*::saturate;/\/\/ using ::saturate;/' code/pyrt/common/gdt/gdt/gdt.h || true
  fi

  # Prefer pip editable; if it fails, fall back to CMake+Ninja
  (cd code/pyrt && "$PY_BIN" -m pip install -v -e .) || {
    mkdir -p code/pyrt/build && cd code/pyrt/build
    cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DTorch_DIR="$Torch_DIR" -Dpybind11_DIR="$pybind11_DIR" \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    cmake --build . -j "$MAX_JOBS"
    cd ../../..
  }
else
  echo "==> No code/pyrt directory; skipping C++ build."
fi

# ---------- Smoke tests ----------
"$PY_BIN" - <<'PY'
import numpy, torch, tinycudann as tcnn
print("==> NumPy:", numpy.__version__)
print("==> Torch CUDA:", torch.version.cuda)
print("==> tinycudann:", tcnn.__file__)
PY

echo "==> setup_pip.sh done."
