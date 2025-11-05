#!/usr/bin/env bash
# Set the default scene
SCENE="cube"
TAGS="debug"
CONS_LOSS=0.0
SPEC_LOSS=0.0

# Override with the first command-line argument, if provided
if [ -n "$1" ]; then
  SCENE="$1"
fi

if [ -n "$2" ]; then
  TAGS="$2"
fi

if [ -n "$3" ]; then
  CONS_LOSS="$3"
fi

if [ -n "$4" ]; then
  SPEC_LOSS="$4"
fi

echo "Using scene: $SCENE"

cd code
OPENCV_IO_ENABLE_OPENEXR=1 python3 training/train.py \
  /workspace/datasets/nerf_synthetic/$SCENE \
  outputs \
  --name pbrnerf_nerfsynthetic_$SCENE \
  --tags $TAGS \
  --override_cons_weighting $CONS_LOSS \
  --override_spec_weighting $SPEC_LOSS \
  --config_path configs/config_nerfsynthetic_pbrnerf.json

