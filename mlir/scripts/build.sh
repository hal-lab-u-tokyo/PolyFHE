#!/bin/sh
set -ex

REPO_ROOT=$(git rev-parse --show-toplevel)
cd $REPO_ROOT

BUILD_SYSTEM="Ninja"
BUILD_DIR="build"

#rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
cd $BUILD_DIR

LLVM_BUILD_DIR=thirdparty/llvm-project/build

cmake -G $BUILD_SYSTEM .. \
    -DLLVM_DIR="$LLVM_BUILD_DIR/lib/cmake/llvm" \
    -DMLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir" \
    -DBUILD_DEPS="ON" \
    -DBUILD_SHARED_LIBS="OFF" \
    -DCMAKE_BUILD_TYPE=Debug

cd $REPO_ROOT
cmake --build $BUILD_DIR --target hifive-opt