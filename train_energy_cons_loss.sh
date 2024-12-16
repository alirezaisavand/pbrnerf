#!/usr/bin/env bash
cd code
python training/train.py \
  ~/scratch/datasets/neilfpp_synthetic/synthetic_city \
  outputs \
  --name pbrnerf_energy_cons_loss \
  --config_path configs/config_synthetic_data_energy_cons_loss.json

