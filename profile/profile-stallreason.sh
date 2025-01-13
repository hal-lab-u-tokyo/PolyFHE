#!/bin/bash
set -xe

REPO_ROOT=$(git rev-parse --show-toplevel)
cd $REPO_ROOT

METRICS="smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct,smsp__warp_issue_stalled_drain_per_warp_active.pct,smsp__warp_issue_stalled_imc_miss_per_warp_active.pct,smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,smsp__warp_issue_stalled_membar_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,smsp__warp_issue_stalled_misc_per_warp_active.pct,smsp__warp_issue_stalled_no_instruction_per_warp_active.pct,smsp__warp_issue_stalled_not_selected_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_sleeping_per_warp_active.pct,smsp__warp_issue_stalled_tex_throttle_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct,smsp__warp_issue_stalled_selected_per_warp_active.pct"

SNAME="evalstall-stallreason-noopt"
TNAME="evalstall-exectime-noopt"
ncu -f -o "${SNAME}" --csv --metrics "${METRICS}" ./build-noopt/bench
ncu --csv --import "${SNAME}".ncu-rep > profile/data/"${SNAME}".csv
nsys profile -w true -o "${TNAME}" ./build-noopt/bench
nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true ./"${TNAME}".nsys-rep > profile/data/"${TNAME}".csv

SNAME="evalstall-stallreason-opt"
TNAME="evalstall-exectime-opt"
ncu -f -o "${SNAME}" --csv --metrics "${METRICS}" ./build-opt/bench
ncu --csv --import "${SNAME}".ncu-rep > profile/data/"${SNAME}".csv
nsys profile -w true -o "${TNAME}" ./build-opt/bench
nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true ./"${TNAME}".nsys-rep > profile/data/"${TNAME}".csv

SNAME="evalstall-stallreason-phantom"
TNAME="evalstall-exectime-phantom"
BIN="./thirdparty/phantom-fhe/build-for-eval/ckks_hmult_logn16_L6"
ncu -f -o "${SNAME}" --profile-from-start off --csv --metrics "${METRICS}" "${BIN}"
ncu --csv --import "${SNAME}".ncu-rep > profile/data/"${SNAME}".csv
nsys profile -w true -o "${TNAME}" "${BIN}"
nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true ./"${TNAME}".nsys-rep > profile/data/"${TNAME}".csv

python3 ./profile/plot-stallreason.py