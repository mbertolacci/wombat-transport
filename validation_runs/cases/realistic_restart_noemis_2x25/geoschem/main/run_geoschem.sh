#!/usr/bin/env bash

set -euo pipefail
set -x

mkdir -p OutputDir Restarts
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OMP_STACKSIZE="${OMP_STACKSIZE:-1000M}"

./gcclassic 2>&1 | tee gcclassic.log
