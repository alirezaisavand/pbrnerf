#!/usr/bin/env bash
# Set the default scene
SCENE="city"
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
python training/train.py \
  ~/scratch/datasets/neilfpp_synthetic/synthetic_$SCENE \
  outputs \
  --name pbrnerf_neilfpp_$SCENE \
  --tags $TAGS \
  --override_cons_weighting $CONS_LOSS \
  --override_spec_weighting $SPEC_LOSS \
  --config_path configs/config_synthetic_data_pbrnerf_neilfpp.json

