#!/usr/bin/env python
"""Entry point for f-LENNS (kept at this path for backwards compatibility).

This is a thin wrapper around ``flenns.cli``; it adds the repository root to
sys.path so that the package can be used straight from a git checkout, without
installing anything. This is what ``run_flenns.sh`` calls with ``fspython``.
"""

import os
import sys

basepath = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if basepath not in sys.path:
    sys.path.insert(0, basepath)

from flenns.cli import main

if __name__ == "__main__":
    sys.exit(main())
