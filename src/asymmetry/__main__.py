"""Entry point for ``python -m asymmetry``."""

import multiprocessing as mp

from asymmetry.cli import main

if __name__ == "__main__":
    mp.freeze_support()
    main()
