#!/bin/bash
set -xe

# Usage
# ./profile-exectime.sh
# argv[0]: this script

REPO_ROOT=$(git rev-parse --show-toplevel)
cd $REPO_ROOT

BIN="build-opt/cc-polyfhe"
BIN_RUNTIME="build/bench"
TARGET="data/graph_fhe.dot"
SRC_RUNTIME="polyfhe/kernel/device_context.cu polyfhe/kernel/polynomial.cpp polyfhe/utils.cpp"
CXXFLAGS_RUNTIME="-g -std=c++17 -I./  --relocatable-device-code true"

mkdir -p build
for logN in {15,16}; do
    for L in {6,12,18,24,30,36}; do
        for dnum in {2,3,6,18}; do
            # if dnum > L, skip
            if [ $dnum -gt $L ]; then
                continue
            fi
            for SharedMemKB in {20,40,60,80}; do
                NAME="logN${logN}_L${L}_dnum${dnum}_SMemKB${SharedMemKB}"
                CONFIG_FILE="./config/config-${NAME}.csv"
                echo "logN,$logN" > $CONFIG_FILE
                echo "L,$L" >> $CONFIG_FILE
                echo "dnum,$dnum" >> $CONFIG_FILE
                echo "SharedMemKB,$SharedMemKB" >> $CONFIG_FILE
                OUTPUT_FILE="profile/data/exectime/exectime-${NAME}.txt"

                ./${BIN} -i ${TARGET} -c ${CONFIG_FILE}
                nvcc -o ${BIN_RUNTIME} build/generated.cu ${SRC_RUNTIME} ${CXXFLAGS_RUNTIME}
                ./${BIN_RUNTIME} > $OUTPUT_FILE

                echo "Executed with logN=$logN, L=$L, dnum=$dnum, SharedMemKB=$SharedMemKB"
            done
        done
    done
done