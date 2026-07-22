#!/usr/bin/env python3
"""Generate a PJC module copy with deterministic north-pole geometry."""

from __future__ import annotations

import argparse
from pathlib import Path


OLD = """    ! SIN of lat edge at North Pole
    SINE_FV(J2_GL+1) = 1.e+0_fp
"""
NEW = """    ! SIN and COS of lat edge at North Pole.  Upstream initializes
    ! SINE_FV but leaves COSE_FV undefined; zero is the exact polar value and
    ! matches the established standalone PJC oracle.
    SINE_FV(J2_GL+1) = 1.e+0_fp
    COSE_FV(J2_GL+1) = 0.e+0_fp
"""


def generate(source: Path, output: Path) -> None:
    text = source.read_text(encoding="utf-8")
    count = text.count(OLD)
    if count != 1:
        raise SystemExit(f"north-pole PJC anchor matched {count} times, expected 1")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text.replace(OLD, NEW), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    generate(args.source, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
