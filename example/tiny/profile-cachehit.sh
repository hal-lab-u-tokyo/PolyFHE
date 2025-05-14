#!/bin/bash
set -xe

# Usage
# ./profile-cachehit.sh
# argv[0]: this script

REPO_ROOT=$(git rev-parse --show-toplevel)
cd ${REPO_ROOT}/example/tiny

# l1tex__t_sector_hit_rate.pct: L1 Cache Hit Rate
# lts__t_sector_hit_rate.pct: LTS Cache Hit Rate
METRICS="l1tex__t_sector_hit_rate,lts__t_sector_hit_rate"
PARAM_SIZE=large

mkdir -p data
mkdir -p data/${PARAM_SIZE}


ncu -f -o cachehit --csv --metrics "${METRICS}" ./example.out 2
ncu --csv --import cachehit.ncu-rep > data/${PARAM_SIZE}/cachehit-reg.csv

ncu -f -o cachehit --csv --metrics "${METRICS}" ./example.out 1
ncu --csv --import cachehit.ncu-rep > data/${PARAM_SIZE}/cachehit-l2.csv

ncu -f -o cachehit --csv --metrics "${METRICS}" ./example.out 0
ncu --csv --import cachehit.ncu-rep > data/${PARAM_SIZE}/cachehit-noopt.csv

uv run ./plot-cachehit.py