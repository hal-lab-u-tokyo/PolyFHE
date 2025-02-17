#!/bin/bash
set -xe

# Usage
# ./profile-memtransfer.sh
# argv[0]: this script

REPO_ROOT=$(git rev-parse --show-toplevel)
cd $REPO_ROOT

BIN="build-opt/cc-polyfhe"
BIN_RUNTIME="build/bench"
TARGET="data/graph_fhe.dot"
SRC_RUNTIME="polyfhe/kernel/device_context.cu polyfhe/kernel/polynomial.cpp polyfhe/utils.cpp"
CXXFLAGS_RUNTIME="-g -std=c++17 -I./  --relocatable-device-code true"

logN=16
L=6
dnum=3

# Run the program 10 times with different configurations
mkdir -p build
for i in $(seq 11 18); do
CONFIG_FILE="./config/config-${i}0KB.csv"
SharedMemKB="${i}0"
echo "logN,$logN" > $CONFIG_FILE
echo "L,$L" >> $CONFIG_FILE
echo "dnum,$dnum" >> $CONFIG_FILE
echo "SharedMemKB,$SharedMemKB" >> $CONFIG_FILE                
CONFIG=config/config-${i}0KB.csv
echo "Running with config file: $CONFIG"
./${BIN} -i ${TARGET} -c ${CONFIG}
nvcc -o ${BIN_RUNTIME} build/generated.cu ${SRC_RUNTIME} ${CXXFLAGS_RUNTIME}
./${BIN_RUNTIME} > profile/data/smem_limit/rump/${i}0KB.txt	
done

python3 ./profile/plot-smemlimit-gpus.py

#python3 ./profile/plot-memtransfer.py