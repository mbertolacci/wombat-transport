#!/usr/bin/env bash

set -euo pipefail
set -x

reuse_dir="${GC_REUSE_RUN_DIR:-../../window1/geoschem}"
if [[ ! -x "${reuse_dir}/gcclassic" ]]; then
  echo "Expected compiled executable at ${reuse_dir}/gcclassic; compile the previous stage first or set GC_REUSE_RUN_DIR." >&2
  exit 2
fi

ln -sfn "${reuse_dir}/gcclassic" gcclassic
if [[ -x "${reuse_dir}/kpp_standalone" ]]; then
  ln -sfn "${reuse_dir}/kpp_standalone" kpp_standalone
fi
