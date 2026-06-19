#!/bin/bash

set -e
set -x

cd build
cmake ../CodeDir -DRUNDIR=..
make -j
make install
