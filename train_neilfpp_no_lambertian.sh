#!/usr/bin/env bash
cd code
python training/train.py \
  ~/scratch/datasets/neilfpp_synthetic/synthetic_city \
  outputs \
  --name neilfpp_no_lambertian \
  --pixel_sampling_weights_path ~/scratch/datasets/neilfpp_synthetic/synthetic_city/inputs/fov_hemisphere_check_pixel_sampling_weights.npy \
  --config_path configs/config_synthetic_data_neilfpp_no_lambertian.json

# python evaluation/evaluate.py ~/scratch/datasets/neilfpp_synthetic/synthetic_city outputs --config_path configs/config_synthetic_data_neilfpp_no_lambertian.json --eval_nvs --eval_brdf --export_nvs --export_lighting

