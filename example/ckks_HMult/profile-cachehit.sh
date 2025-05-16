#!/bin/bash
set -xe

# Usage
# ./profile-cachehit.sh
# argv[0]: this script

REPO_ROOT=$(git rev-parse --show-toplevel)
cd ${REPO_ROOT}/example/ckks_HMult

# l1tex__t_sector_hit_rate.pct: L1 Cache Hit Rate
# lts__t_sector_hit_rate.pct: LTS Cache Hit Rate
METRICS="l1tex__t_sector_hit_rate,lts__t_sector_hit_rate,dram__bytes_read.sum,dram__bytes_write.sum"
PARAM_SIZE=large

mkdir -p data
NUM_DIVIDES_LIST=(1 2 4 6 9 12 18 36)

for NUM_DIVIDES in ${NUM_DIVIDES_LIST[@]}; do
    echo "NUM_DIVIDES=${NUM_DIVIDES}"
    ncu -f -o cachehit --nvtx --nvtx-include "compute/" --csv --metrics "${METRICS}" ./build/example.out ${NUM_DIVIDES}
    ncu --csv --import cachehit.ncu-rep > data/ncu-ntt-${NUM_DIVIDES}.csv
done
