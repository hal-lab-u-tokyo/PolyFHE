!#/bin/bash
set -ex

OPT_LEVEL=$1
if [ -z "$OPT_LEVEL" ]; then
  echo "Usage: $0 <opt_level> <n_opt>"
  echo "  opt_level: 0(noopt), 1(reg), 2(reg + L2)"
  echo "  n_opt: 1, 2, 3.."
  exit 1
fi

N_OPT=$2
if [ -z "$N_OPT" ]; then
  echo "Usage: $0 <opt_level> <n_opt>"
  echo "  opt_level: 0(noopt), 1(reg), 2(reg + L2)"
  echo "  n_opt: 1, 2, 3.."
  exit 1
fi

REPO_ROOT=$(git rev-parse --show-toplevel)
cd $REPO_ROOT/example/ckks_HMult
uv run ./example.py
make clean
make OPT_LEVEL=$OPT_LEVEL N_OPT=$N_OPT
make dot