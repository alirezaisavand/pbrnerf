#!/usr/bin/env bash
SCENES=(
  "city"
  "studio"
  "castel"
  "city_mix"
  "studio_mix"
  "castel_mix"
)

TAGS="S_L_256"

for scene in "${SCENES[@]}"; do
    echo "Processing scene: $scene"

    ./train_neilfpp.sh $scene $TAGS
    ./train_pbrnerf_neilfpp.sh $scene $TAGS
done
