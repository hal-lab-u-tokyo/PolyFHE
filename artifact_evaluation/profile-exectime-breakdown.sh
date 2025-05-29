#!/bin/bash
set -xe

REPO_ROOT=$(git rev-parse --show-toplevel)
cd $REPO_ROOT

PROFILE_NAME="exectime-breakdown"
BIN="${REPO_ROOT}/artifact_evaluation/polyfhe-noopt/build/example.out"
OUTPUT_DIR="${REPO_ROOT}/artifact_evaluation/data"
# METRICS="smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct,smsp__warp_issue_stalled_drain_per_warp_active.pct,smsp__warp_issue_stalled_imc_miss_per_warp_active.pct,smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,smsp__warp_issue_stalled_membar_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,smsp__warp_issue_stalled_misc_per_warp_active.pct,smsp__warp_issue_stalled_no_instruction_per_warp_active.pct,smsp__warp_issue_stalled_not_selected_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_sleeping_per_warp_active.pct,smsp__warp_issue_stalled_tex_throttle_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct,smsp__warp_issue_stalled_selected_per_warp_active.pct"

# ncu -f -o motivative-stallreason --profile-from-start off --csv --metrics "${METRICS}" "${BIN}"
# ncu --csv --import motivative-stallreason.ncu-rep > profile/data/phantom/phantom-L36-stallreason.csv

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/figure"
# nsys start -c cudaProfilerApi
# nsys profile -w true --force-overwrite true -o "${OUTPUT_DIR}/${PROFILE_NAME}" --capture-range=cudaProfilerApi "${BIN}" 1
nsys stats --report cuda_gpu_kern_sum --format csv "${OUTPUT_DIR}/${PROFILE_NAME}".nsys-rep > "${OUTPUT_DIR}/${PROFILE_NAME}".csv

uv run ./artifact_evaluation/plot-exectime-breakdown.py --fname "${PROFILE_NAME}"
# python3 ./profile/plot-motivative-ex-stallreason.py
