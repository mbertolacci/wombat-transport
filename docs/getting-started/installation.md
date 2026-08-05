# Installation

Wombat requires Python 3.10 or newer and is currently installed from its Git
repository.

## Create an environment

```bash
git clone https://github.com/mbertolacci/wombat-transport.git
cd wombat-transport
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

The runtime dependencies install NumPy, netCDF4, and the YAML 1.2 parser.
Wombat supports NumPy 1.26 and NumPy 2.x through 2.3.

## Install CPU acceleration

Numba is optional for correctness but strongly recommended for production.
Without it, Wombat uses much slower native NumPy/Python operator paths and
emits a major performance warning.

```bash
.venv/bin/python -m pip install -e '.[cpu]'
```

Confirm that the command-line module is available:

```bash
.venv/bin/python -m wombat_transport.run --help
```

## Install CUDA development dependencies

The CUDA 12 extra installs CuPy as the device-array, runtime, and raw CUDA C++
kernel layer. Install it alongside the CPU extra when developing or evaluating
CUDA kernels:

```bash
.venv/bin/python -m pip install -e '.[cpu,cuda]'
```

This requires a CUDA 12 toolkit and a compatible NVIDIA driver. The
correctness-first CUDA runner can then be selected explicitly:

```bash
WOMBAT_BACKEND=cuda \
WOMBAT_CUDA_DTYPE=float64 \
WOMBAT_NUMBA_THREADS=8 \
.venv/bin/python -m wombat_transport.run run.yml
```

`WOMBAT_CUDA_DTYPE` accepts `float32` or `float64`, and
`WOMBAT_CUDA_DEVICE` selects the device index. The CPU extra remains required:
the current CUDA implementation prepares tracer-independent transport plans
with Numba on the CPU before uploading them each step. Tracer storage, HISTORY
accumulation, and ObsOperator sampling remain device-resident between their
explicit I/O boundaries.

## Development and documentation extras

Install the test or documentation dependencies only when needed:

```bash
.venv/bin/python -m pip install -e '.[dev,cpu]'
.venv/bin/python -m pip install -e '.[docs]'
```

Preview this documentation with:

```bash
.venv/bin/zensical serve
```

The development site is then available at `http://localhost:8000/`.
