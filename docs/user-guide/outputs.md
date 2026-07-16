# Outputs

Wombat writes GEOS-Chem HISTORY-like concentration and restart collections.
Output collection paths are controlled by `outputs.collections`, relative to
the run YAML directory unless absolute.

The top-level `output_dir` field is currently required for configuration
compatibility and inspection, but it does not redirect collection files.

## Concentration collections

A concentration collection must use `mode: time-averaged` and the
`SpeciesConcVV_?ADV?` field token.

```yaml
outputs:
  expid: OutputDir/Wombat
  collections:
    SpeciesConcThreeHourly:
      template: "%y4%m2%d2_%h2%n2z.nc4"
      frequency: 00000000 030000
      duration: 00000000 030000
      mode: time-averaged
      fields:
        - SpeciesConcVV_?ADV?
```

With this example, files are named
`OutputDir/Wombat.SpeciesConcThreeHourly.<timestamp>.nc4`.

## Restart collections

A restart collection must use `mode: instantaneous`, include
`SpeciesRst_?ALL?`, and have a nonzero frequency.

```yaml
Restart:
  filename: ./Restarts/GEOSChem.Restart.%y4%m2%d2_%h2%n2z.nc4
  frequency: 00000001 000000
  duration: 00000001 000000
  mode: instantaneous
  fields:
    - SpeciesRst_?ALL?
    - Met_DELPDRY
    - Met_PS1WET
    - Met_PS1DRY
    - Met_SPHU1
    - Met_TMPU1
```

The listed meteorology fields are the supported restart companions.

## Intervals and filename templates

HISTORY intervals use `YYYYMMDD hhmmss`. Years are converted to months, and
`duration` defaults to `frequency` when omitted.

Collection filename templates expand `%y4`, `%m2`, `%d2`, `%h2`, and `%n2`.
Each collection requires either:

- `filename`, which is the complete path template; or
- `template`, which is appended to `<expid>.<collection-name>.`.

## Storage

```yaml
outputs:
  dtype: float32
  compression:
    enabled: true
    level: 1
    shuffle: true
  chunking:
    rank1: [512]
    rank2:
    rank3: [1, 91, 144]
    rank4: [1, 1, 91, 144]
```

`dtype` may be `float32` or `float64`. Compression level must be from zero to
nine. A null chunk entry lets netCDF choose; otherwise each rank-specific
array must contain that number of positive dimensions. Chunk sizes should be
adjusted for 4x5 dimensions.

## Synchronous and threaded writing

`outputs.writer` may be `sync` or `threaded` and defaults to `sync`. Threaded
mode moves completed time-average NetCDF work behind the simulation thread;
restart collections remain durable before shutdown returns. Threaded output
can increase memory use and is not guaranteed to improve wall time when
transport dominates or storage is already fast.

ObsOperator has its own output and restart files. It does not use the HISTORY
writer setting; see [ObsOperator](obsoperator.md).
