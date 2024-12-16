#!/usr/bin/env bash
cd code
python training/train.py \
  ~/scratch/datasets/data_dtu/DTU_scan1 \
  outputs \
  --name neilfpp_dtu \
  --config_path configs/config_dtu_volsdf.json

