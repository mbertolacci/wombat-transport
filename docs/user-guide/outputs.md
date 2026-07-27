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
    algorithm: zlib
    level: 1
    shuffle: true
  chunking:
    rank1: [512]
    rank2:
    rank3: [1, 91, 144]
    rank4: [1, 1, 91, 144]
```

`dtype` may be `float32` or `float64`. Compression algorithms are `zlib`,
`zstd`, `blosc_lz4`, and `blosc_zstd`; compression level must be from zero to
nine. `zlib` is the portable default. The other algorithms require matching
HDF5 filter plugins in every reader. A null chunk entry lets netCDF choose;
otherwise each rank-specific array must contain that number of positive
dimensions. Chunk sizes should be adjusted for 4x5 dimensions.

Collections inherit these settings and may override `dtype`, `compression`,
or `chunking` directly in their collection mapping. For example, large
HISTORY fields can use threaded Blosc-Zstd while restart collections retain
portable zlib:

```yaml
outputs:
  compression:
    algorithm: zlib
  collections:
    SpeciesConcThreeHourly:
      compression:
        algorithm: blosc_zstd
        level: 1
        shuffle: true
      chunking:
        rank4: [1, 8, 91, 144]
```

Set `BLOSC_NTHREADS` before starting Wombat to control Blosc's internal worker
count. Coordinate and time variables remain on zlib because the Blosc HDF5
filter does not accept every very small metadata buffer.

## Synchronous writing

HISTORY accumulation and NetCDF writing are synchronous. The optional
`outputs.writer: sync` setting is accepted for compatibility; `threaded` is
rejected. NetCDF calls remain sequential.

ObsOperator has its own output and restart files. It does not use the HISTORY
writer setting; see [ObsOperator](obsoperator.md).
