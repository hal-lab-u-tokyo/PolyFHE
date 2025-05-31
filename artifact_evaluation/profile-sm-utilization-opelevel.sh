#!/bin/bash
set -xe

REPO_ROOT=$(git rev-parse --show-toplevel)
cd ${REPO_ROOT}

# PROFILE_NAME="memaccess-RTX4090"
# BIN="${REPO_ROOT}/artifact_evaluation/polyfhe-noopt/build/example.out"
# PROFILE_NAME="memaccess-l2opt"
# BIN="${REPO_ROOT}/example/ckks_HMult/build/example.out"
OUTPUT_DIR="${REPO_ROOT}/artifact_evaluation/data"

PARAMSET=("setB" "setC")
OPTLEVEL=("noopt" "reg" "l2")
# OPTLEVEL=("noopt")

METRICS="sm__warps_active.avg.pct"
# METRICS+=",sm__throughput.avg.pct"

mkdir -p "${OUTPUT_DIR}"

for param in "${PARAMSET[@]}"; do
  for opt in "${OPTLEVEL[@]}"; do
    PROFILE_NAME="smutil-${param}-${opt}-A4090"
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