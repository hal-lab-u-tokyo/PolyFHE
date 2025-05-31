#!/bin/bash
set -xe

REPO_ROOT=$(git rev-parse --show-toplevel)
cd ${REPO_ROOT}

OUTPUT_DIR="${REPO_ROOT}/artifact_evaluation/data"
PARAMSET=("setB" "setC")
OPTLEVEL=("noopt" "reg" "l2")
METRICS="smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct,smsp__warp_issue_stalled_drain_per_warp_active.pct,smsp__warp_issue_stalled_imc_miss_per_warp_active.pct,smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,smsp__warp_issue_stalled_membar_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,smsp__warp_issue_stalled_misc_per_warp_active.pct,smsp__warp_issue_stalled_no_instruction_per_warp_active.pct,smsp__warp_issue_stalled_not_selected_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_sleeping_per_warp_active.pct,smsp__warp_issue_stalled_tex_throttle_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct,smsp__warp_issue_stalled_selected_per_warp_active.pct"

mkdir -p "${OUTPUT_DIR}"

for param in "${PARAMSET[@]}"; do
  for opt in "${OPTLEVEL[@]}"; do
    PROFILE_NAME="stallreason-${param}-${opt}-A4090"
    BIN="${REPO_ROOT}/example/ckks_HMult/final-${param}/build/example-${opt}.out"
    echo "Profiling ${PROFILE_NAME} with binary ${BIN}"
    
    ncu -f -o "${OUTPUT_DIR}/${PROFILE_NAME}" --profile-from-start off --nvtx --nvtx-include "compute/" --csv --metrics "${METRICS}" "${BIN}" 1
    ncu --csv --import "${OUTPUT_DIR}/${PROFILE_NAME}".ncu-rep > "${OUTPUT_DIR}/${PROFILE_NAME}".csv
    
    # uv run ./artifact_evaluation/plot-memaccess.py --fname "${PROFILE_NAME}"
  done
done


# cd "${REPO_ROOT}/example/ckks_HMult"
# ncu -f -o "${OUTPUT_DIR}/${PROFILE_NAME}" --profile-from-start off --nvtx --nvtx-include "compute/" --csv --metrics "${METRICS}" "${BIN}" 2
# ncu --csv --import "${OUTPUT_DIR}/${PROFILE_NAME}".ncu-rep > "${OUTPUT_DIR}/${PROFILE_NAME}".csv

# uv run ./artifact_evaluation/plot-memaccess.py --fname "${PROFILE_NAME}"