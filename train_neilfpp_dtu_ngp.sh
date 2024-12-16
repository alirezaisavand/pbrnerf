#!/usr/bin/env bash
# Set the default scene
SCENE="DTU_scan37"
TAGS="debug"

# Override with the first command-line argument, if provided
if [ -n "$1" ]; then
  SCENE="$1"
fi

if [ -n "$2" ]; then
  TAGS="$2"
fi

echo "Using scene: $SCENE"

cd code
python training/train.py \
  ~/scratch/datasets/data_dtu/$SCENE \
  outputs \
  --name neilfpp_dtu_ngp_$SCENE \
  --tags $TAGS \
  --config_path configs/config_dtu_volsdf_ngp.json

