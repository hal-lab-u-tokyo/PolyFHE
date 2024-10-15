!#/bin/bash
set -ex

REPO_ROOT=$(git rev-parse --show-toplevel)
cd $REPO_ROOT

cd hifive/kernel/FullRNS-HEAAN/lib
make clean
make -j 32