#!/bin/bash
set -xe

# Usage
# ./profile-stallreason.sh <output file name>
# argv[0]: this script
# argv[1]: output file name

REPO_ROOT=$(git rev-parse --show-toplevel)
cd $REPO_ROOT

OUTPUT_FILE=$1
METRICS="smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct,smsp__warp_issue_stalled_drain_per_warp_active.pct,smsp__warp_issue_stalled_imc_miss_per_warp_active.pct,smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,smsp__warp_issue_stalled_membar_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,smsp__warp_issue_stalled_misc_per_warp_active.pct,smsp__warp_issue_stalled_no_instruction_per_warp_active.pct,smsp__warp_issue_stalled_not_selected_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_sleeping_per_warp_active.pct,smsp__warp_issue_stalled_tex_throttle_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct,smsp__warp_issue_stalled_selected_per_warp_active.pct"

ncu -f -o ${OUTPUT_FILE} --csv --metrics "${METRICS}" ./build/gen_cuda

ncu --csv --import ${OUTPUT_FILE}.ncu-rep > profile/data/${OUTPUT_FILE}.csv

python3 ./profile/plot-stallreason.py ${OUTPUT_FILE}