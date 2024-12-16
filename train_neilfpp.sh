#!/usr/bin/env bash
# Set the default scene
SCENE="city"
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
  ~/scratch/datasets/neilfpp_synthetic/synthetic_$SCENE \
  outputs \
  --name neilfpp_$SCENE \
  --tags $TAGS \
  --config_path configs/config_synthetic_data_neilfpp.json

