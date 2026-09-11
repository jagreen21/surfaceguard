"""Entry point for both `python -m surfaceguard` and the frozen app."""

import multiprocessing
import sys

from surfaceguard.app import main

if __name__ == "__main__":
    # Without this a frozen app re-executes itself for every worker process.
    multiprocessing.freeze_support()
    sys.exit(main())
