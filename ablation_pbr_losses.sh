#!/usr/bin/env bash
./train_neilfpp.sh
./train_neilfpp_no_lambertian.sh
./train_energy_cons_loss.sh
./train_ndf_weighted_specular_loss.sh

