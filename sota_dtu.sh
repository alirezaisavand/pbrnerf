#!/usr/bin/env bash
SCENES=(
    "DTU_scan24"
    "DTU_scan37"
    "DTU_scan97"
    "DTU_scan40"
    "DTU_scan55"
    "DTU_scan63"
    "DTU_scan65"
    "DTU_scan69"
    "DTU_scan83"
    "DTU_scan105"
    "DTU_scan106"
    "DTU_scan110"
    "DTU_scan114"
    "DTU_scan118"
    "DTU_scan122"
)

TAGS="avg"

for scene in "${SCENES[@]}"; do
    echo "Processing scene: $scene"

    ./train_neilfpp_dtu_ngp.sh $scene $TAGS
    ./train_pbrnerf_dtu.sh $scene $TAGS
done
