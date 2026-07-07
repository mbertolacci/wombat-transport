#!/usr/bin/env python3
"""Generate a GEOS-Chem convection module copy with DO_CLOUD_CONVECTION exposed."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    text = args.input.read_text(encoding="utf-8")
    text = text.replace("  PRIVATE :: DO_CLOUD_CONVECTION\n", "  PUBLIC  :: DO_CLOUD_CONVECTION\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
