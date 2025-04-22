!#/bin/bash
set -ex

OPT_LEVEL=$1
if [ -z "$OPT_LEVEL" ]; then
  echo "Usage: $0 <opt_level>"
  echo "Example: $0 noopt"
  exit 1
fi

REPO_ROOT=$(git rev-parse --show-toplevel)
cd $REPO_ROOT/example/ckks_HMult
uv run ./example.py
make clean
make OPT_LEVEL=--$OPT_LEVEL
make dot