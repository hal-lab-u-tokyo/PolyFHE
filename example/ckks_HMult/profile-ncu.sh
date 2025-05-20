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
# METRICS="smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct,smsp__warp_issue_stalled_drain_per_warp_active.pct,smsp__warp_issue_stalled_imc_miss_per_warp_active.pct,smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,smsp__warp_issue_stalled_membar_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,smsp__warp_issue_stalled_misc_per_warp_active.pct,smsp__warp_issue_stalled_no_instruction_per_warp_active.pct,smsp__warp_issue_stalled_not_selected_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_sleeping_per_warp_active.pct,smsp__warp_issue_stalled_tex_throttle_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct,smsp__warp_issue_stalled_selected_per_warp_active.pct"

mkdir -p data
# NUM_DIVIDES_LIST=(1 2 4 6 9 12 18 36)
NUM_DIVIDES_LIST=(0 2)

for NUM_DIVIDES in ${NUM_DIVIDES_LIST[@]}; do
    echo "NUM_DIVIDES=${NUM_DIVIDES}"
    rm -f cachehit.ncu-rep
    ncu -f -o cachehit --nvtx --nvtx-include "compute/" --csv --metrics "${METRICS}" ./build/example.out ${NUM_DIVIDES}
    ncu --csv --import cachehit.ncu-rep > data/ncu-memory-accum-v7-${NUM_DIVIDES}.csv
done
