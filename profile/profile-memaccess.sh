#!/bin/bash
set -xe

# Usage
# ./profile-memaccess.sh
# argv[0]: this script

REPO_ROOT=$(git rev-parse --show-toplevel)
cd $REPO_ROOT

METRICS="l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct"

ncu -f -o opt --csv --metrics "${METRICS}" ./build-opt/gen_cuda
ncu --csv --import opt.ncu-rep > profile/data/memaccess-opt.csv

ncu -f -o noopt --csv --metrics "${METRICS}" ./build-noopt/gen_cuda
ncu --csv --import noopt.ncu-rep > profile/data/memaccess-noopt.csv

python3 ./profile/plot-memaccess.py