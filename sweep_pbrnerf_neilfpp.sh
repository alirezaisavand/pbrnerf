#!/usr/bin/env bash
CONS_LOSS_WEIGHTINGS=(1.5 1.0 0.5 0.1 0.05 0.01 0.005)
SPEC_LOSS_WEIGHTINGS=(1.5 1.0 0.5 0.1 0.05 0.01 0.005)

SCENE="city"
TAGS="sweep_cons_loss,sweep_spec_loss"

for cons_weighting in "${CONS_LOSS_WEIGHTINGS[@]}"; do
    for spec_weighting in "${SPEC_LOSS_WEIGHTINGS[@]}"; do
        echo "Processing scene: $SCENE"
        echo "Running with CONS_LOSS_WEIGHTING=$cons_weighting and SPEC_LOSS_WEIGHTINGS=$spec_weighting"

        ./train_pbrnerf_neilfpp.sh $SCENE $TAGS $cons_weighting $spec_weighting
    done
done
