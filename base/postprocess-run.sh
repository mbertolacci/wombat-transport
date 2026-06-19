#!/bin/bash

set -ex

for month in ; do
    rm -f OutputDir/HEMCO_diagnostics.${month}*
done

if ls OutputDir | grep -q HEMCO_diagnostics; then
    cdo -L -z zip_6 \
        -splityearmon \
        -cat OutputDir/HEMCO_diagnostics.\?\?\?\?\?\?\?\?\?\?\?\?.nc \
        OutputDir/HEMCO_diagnostics.
fi

find OutputDir/ \
    -name "HEMCO_diagnostics.????????????.nc" \
    -delete

gzip -f OutputDir/HEMCO.log OutputDir/gcclassic.log
