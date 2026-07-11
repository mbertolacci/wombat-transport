# Continuous No-Emissions Four-Day Run

This case is the uninterrupted counterpart to `restart_chain_noemis`.

It starts from the same GEOS-Chem CO2 restart on `2014-09-01 00:00` and runs
without emissions to `2014-09-05 00:00`. The purpose is to distinguish
restart-chain drift from drift that appears in a continuous integration of the
same transport-only configuration.
