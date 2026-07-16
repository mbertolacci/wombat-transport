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

## Install Numba

Numba is optional for correctness but strongly recommended for production.
Without it, Wombat uses much slower native NumPy/Python operator paths and
emits a major performance warning.

```bash
.venv/bin/python -m pip install numba
```

Confirm that the command-line module is available:

```bash
.venv/bin/python -m wombat_transport.run --help
```

## Development and documentation extras

Install the test or documentation dependencies only when needed:

```bash
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pip install -e '.[docs]'
```

Preview this documentation with:

```bash
.venv/bin/zensical serve
```

The development site is then available at `http://localhost:8000/`.
