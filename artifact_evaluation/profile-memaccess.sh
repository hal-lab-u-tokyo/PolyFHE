#!/bin/bash
set -xe

REPO_ROOT=$(git rev-parse --show-toplevel)
cd ${REPO_ROOT}

PROFILE_NAME="memaccess-noopt-A100"
BIN="${REPO_ROOT}/artifact_evaluation/polyfhe-noopt/build/example.out"
# PROFILE_NAME="memaccess-l2opt"
# BIN="${REPO_ROOT}/example/ckks_HMult/build/example.out"
OUTPUT_DIR="${REPO_ROOT}/artifact_evaluation/data"

METRICS="l1tex__t_sector_hit_rate.pct"
METRICS+=",lts__t_sector_hit_rate.pct"
METRICS+=",dram__bytes_read.sum"
METRICS+=",dram__bytes_write.sum"
# METRICS+=",lts__t_sectors_op_write.sum"
# METRICS+=",lts__t_sectors_op_read.sum"
# METRICS+=",lts__t_sector_op_read_hit_rate.pct"
# METRICS+=",lts__t_sector_op_write_hit_rate.pct"
# METRICS+=",lts__t_sector_hit_rate"

mkdir -p "${OUTPUT_DIR}"

ncu -f -o "${OUTPUT_DIR}/${PROFILE_NAME}" --profile-from-start off --nvtx --nvtx-include "compute/" --csv --metrics "${METRICS}" "${BIN}" 2
ncu --csv --import "${OUTPUT_DIR}/${PROFILE_NAME}".ncu-rep > "${OUTPUT_DIR}/${PROFILE_NAME}".csv

uv run ./artifact_evaluation/plot-memaccess.py --fname "${PROFILE_NAME}"