#!/bin/bash
set -xe

REPO_ROOT=$(git rev-parse --show-toplevel)
cd $REPO_ROOT

METRICS="smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct,smsp__warp_issue_stalled_drain_per_warp_active.pct,smsp__warp_issue_stalled_imc_miss_per_warp_active.pct,smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,smsp__warp_issue_stalled_membar_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,smsp__warp_issue_stalled_misc_per_warp_active.pct,smsp__warp_issue_stalled_no_instruction_per_warp_active.pct,smsp__warp_issue_stalled_not_selected_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_sleeping_per_warp_active.pct,smsp__warp_issue_stalled_tex_throttle_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct,smsp__warp_issue_stalled_selected_per_warp_active.pct"

ncu -f -o stallreason-noopt --csv --metrics "${METRICS}" ./build-noopt/bench
ncu --csv --import stallreason-noopt.ncu-rep > profile/data/stallreason-noopt.csv

ncu -f -o stallreason-opt --csv --metrics "${METRICS}" ./build-opt/bench
ncu --csv --import stallreason-opt.ncu-rep > profile/data/stallreason-opt.csv

ncu -f -o stallreason-phantom --csv --metrics "${METRICS}" ./thirdparty/phantom-fhe/build-for-eval/ckks_hmult_logn16_L6
ncu --csv --import stallreason-phantom.ncu-rep > profile/data/stallreason-phantom.csv

python3 ./profile/plot-stallreason.py