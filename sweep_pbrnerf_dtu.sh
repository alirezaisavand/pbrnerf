#!/usr/bin/env bash
CONS_LOSS_WEIGHTINGS=(0.1 0.01 0.001 0.0001 0.00002)
SPEC_LOSS_WEIGHTINGS=(0.1 0.01 0.001 0.0001 0.00002)

SCENE="DTU_scan37"
TAGS="sweep_cons_loss,sweep_spec_loss"

for cons_weighting in "${CONS_LOSS_WEIGHTINGS[@]}"; do
    for spec_weighting in "${SPEC_LOSS_WEIGHTINGS[@]}"; do
        echo "Processing scene: $SCENE"
        echo "Running with CONS_LOSS_WEIGHTING=$cons_weighting and SPEC_LOSS_WEIGHTINGS=$spec_weighting"

        ./train_pbrnerf_dtu.sh $SCENE $TAGS $cons_weighting $spec_weighting
    done
done
