#!/bin/bash
set -xe

REPO_ROOT=$(git rev-parse --show-toplevel)
cd $REPO_ROOT

BIN="thirdparty/phantom-fhe/build-for-eval/ckks_hmult_logn16_L36"
METRICS="smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct,smsp__warp_issue_stalled_drain_per_warp_active.pct,smsp__warp_issue_stalled_imc_miss_per_warp_active.pct,smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,smsp__warp_issue_stalled_membar_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,smsp__warp_issue_stalled_misc_per_warp_active.pct,smsp__warp_issue_stalled_no_instruction_per_warp_active.pct,smsp__warp_issue_stalled_not_selected_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_sleeping_per_warp_active.pct,smsp__warp_issue_stalled_tex_throttle_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct,smsp__warp_issue_stalled_selected_per_warp_active.pct"

ncu -f -o motivative-stallreason --profile-from-start off --csv --metrics "${METRICS}" "${BIN}"
ncu --csv --import motivative-stallreason.ncu-rep > profile/data/phantom/phantom-L36-stallreason.csv

#nsys start -c cudaProfilerApi
#nsys launch -w true thirdparty/phantom-fhe/build-for-eval/ckks_hmult_logn16_L36
#nsys stats --report cuda_gpu_kern_sum --format csv ./report22.nsys-rep > profile/data/phantom/phantom-L36-exectime.csv