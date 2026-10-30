#!/usr/bin/env python3
"""Run the modern component-only GPU CTest suite.

This is a thin launcher: it forwards directly to CTest for the ``gpu_solver``
label and returns the child exit status. It does not configure or build, delete
directories, run a solver or benchmark, and it makes no CPU/GPU trajectory or
alignment claim. Native numerical, caller and physical qualification is a
separate supervisor task.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build",
        type=Path,
        default=Path(__file__).resolve().parent / "build",
        help="GPU build directory (default: gpu/build)",
    )
    parser.add_argument(
        "--ctest",
        default=None,
        help="CTest executable (default: discovered ctest)",
    )
    args = parser.parse_args()

    ctest = args.ctest or shutil.which("ctest")
    if not ctest:
        raise SystemExit("ctest executable not found; pass --ctest EXECUTABLE")

    argv = [
        ctest,
        "--test-dir",
        str(args.build.absolute()),
        "--output-on-failure",
        "--no-tests=error",
        "--parallel",
        "1",
        "-L",
        "gpu_solver",
    ]
    return subprocess.run(argv).returncode


if __name__ == "__main__":
    sys.exit(main())
