#!/usr/bin/env python3
"""Run the physical RGB recording pipeline in an isolated perception environment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from rcr.perception.challenge.physical_run import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
