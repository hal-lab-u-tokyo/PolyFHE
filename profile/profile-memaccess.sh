#!/bin/bash
set -xe

# Usage
# ./profile-memaccess.sh
# argv[0]: this script

REPO_ROOT=$(git rev-parse --show-toplevel)
cd $REPO_ROOT

METRICS="l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"

ncu -f -o opt --csv --metrics "${METRICS}" ./build-opt/bench
ncu --csv --import opt.ncu-rep > profile/data/memaccess-opt.csv

ncu -f -o noopt --csv --metrics "${METRICS}" ./build-noopt/bench
ncu --csv --import noopt.ncu-rep > profile/data/memaccess-noopt.csv

python3 ./profile/plot-memaccess.py