#!/bin/bash
set -xe

# Usage
# ./profile-memtransfer.sh
# argv[0]: this script

REPO_ROOT=$(git rev-parse --show-toplevel)
cd $REPO_ROOT

BIN="build-opt/cc-hifive"
BIN_RUNTIME="build/bench"
TARGET="data/graph_fhe.dot"
SRC_RUNTIME="hifive/kernel/device_context.cu hifive/kernel/polynomial.cpp hifive/utils.cpp"
CXXFLAGS_RUNTIME="-g -std=c++17 -I./  --relocatable-device-code true"

# Run the program 10 times with different configurations
mkdir -p build
for i in {1..10}
do
CONFIG=config/config-${i}0KB.csv
echo "Running with config file: $CONFIG"
./${BIN} -i ${TARGET} -c ${CONFIG}
nvcc -o ${BIN_RUNTIME} build/generated.cu ${SRC_RUNTIME} ${CXXFLAGS_RUNTIME}
./${BIN_RUNTIME} > profile/data/smem_limit/brisket/${i}0KB.txt	
done

python3 ./profile/plot-smemlimit-gpus.py

#python3 ./profile/plot-memtransfer.py