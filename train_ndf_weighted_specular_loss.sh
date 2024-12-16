#!/usr/bin/env bash
cd code
python training/train.py \
  ~/scratch/datasets/neilfpp_synthetic/synthetic_city \
  outputs \
  --name pbrnerf_brdf_weighted_specular_loss \
  --config_path configs/config_synthetic_data_brdf_weighted_specular_loss.json

