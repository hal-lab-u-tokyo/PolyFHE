#!/bin/bash
set -xe

# Usage
# ./profile-cachehit.sh
# argv[0]: this script

REPO_ROOT=$(git rev-parse --show-toplevel)
cd $REPO_ROOT

# l1tex__t_sector_hit_rate.pct: L1 Cache Hit Rate
# lts__t_sector_hit_rate.pct: LTS Cache Hit Rate
METRICS="l1tex__t_sector_hit_rate,lts__t_sector_hit_rate"

ncu -f -o cachehit-opt --csv --metrics "${METRICS}" ./build-opt/bench
ncu --csv --import cachehit-opt.ncu-rep > profile/data/cachehit-opt.csv

ncu -f -o cachehit-noopt --csv --metrics "${METRICS}" ./build-noopt/bench
ncu --csv --import cachehit-noopt.ncu-rep > profile/data/cachehit-noopt.csv

ncu -f -o cachehit-phantom --csv --metrics "${METRICS}" ./thirdparty/phantom-fhe/build-for-eval/ckks_hmult_logn16_L6
ncu --csv --import cachehit-phantom.ncu-rep > profile/data/cachehit-phantom.csv

python3 ./profile/plot-cachehit.py