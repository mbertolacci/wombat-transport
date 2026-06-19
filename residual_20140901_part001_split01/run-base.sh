#!/bin/bash

set -e
set -x
set -o pipefail

export OMP_NUM_THREADS=6
export OMP_STACKSIZE=1000M

./gcclassic | tee OutputDir/gcclassic.log
