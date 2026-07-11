#!/usr/bin/env bash

set -euo pipefail
set -x

if [[ ! -e CodeDir ]]; then
  echo "CodeDir is required; create a symlink to the GEOS-Chem Classic source before compiling." >&2
  exit 2
fi

mkdir -p build
cd build
cmake ../CodeDir -DRUNDIR=..
make -j "${GC_BUILD_JOBS:-$(nproc)}"
make install
