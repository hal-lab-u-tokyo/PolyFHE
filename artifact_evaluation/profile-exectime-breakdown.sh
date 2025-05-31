#!/bin/bash
set -xe

REPO_ROOT=$(git rev-parse --show-toplevel)
cd $REPO_ROOT

PROFILE_NAME="exectime-breakdown-l2opt"
# BIN="${REPO_ROOT}/artifact_evaluation/polyfhe-noopt/build/example.out"
BIN="${REPO_ROOT}/example/ckks_HMult/build/example.out"
OUTPUT_DIR="${REPO_ROOT}/artifact_evaluation/data"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/figure"
# nsys start -c cudaProfilerApi
nsys profile -w true --force-overwrite true -o "${OUTPUT_DIR}/${PROFILE_NAME}" --capture-range=cudaProfilerApi "${BIN}" 2
nsys stats --report cuda_gpu_kern_sum --format csv "${OUTPUT_DIR}/${PROFILE_NAME}".nsys-rep > "${OUTPUT_DIR}/${PROFILE_NAME}".csv

uv run ./artifact_evaluation/plot-exectime-breakdown.py --fname "${PROFILE_NAME}"
# python3 ./profile/plot-motivative-ex-stallreason.py
