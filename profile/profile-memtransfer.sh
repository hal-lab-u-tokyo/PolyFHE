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

METRICS="dram__bytes_read.sum"

mkdir -p build

for i in {1,2}
do
CONFIG=config/config-set${i}.csv
./${BIN} -i ${TARGET} -c ${CONFIG}
nvcc -o build-opt/bench build/generated.cu ${SRC_RUNTIME} ${CXXFLAGS_RUNTIME}
./${BIN} -i ${TARGET} -c ${CONFIG} --noopt
nvcc -o build/bench build/generated.cu ${SRC_RUNTIME} ${CXXFLAGS_RUNTIME}
ncu -f -o memtransfer-opt --csv --metrics "${METRICS}" ./build-opt/bench
ncu --csv --import memtransfer-opt.ncu-rep > profile/data/memtransfer-opt.csv
ncu -f -o memtransfer-noopt --csv --metrics "${METRICS}" ./build-noopt/bench
ncu --csv --import memtransfer-noopt.ncu-rep > profile/data/memtransfer-noopt.csv
ncu -f -o memtransfer-phantom --profile-from-start off --csv --metrics "${METRICS}" ./thirdparty/phantom-fhe/build-for-eval/ckks_set${i}
ncu --csv --import memtransfer-phantom.ncu-rep > profile/data/memtransfer-phantom.csv
python3 ./profile/plot-memtransfer.py set${i}
done