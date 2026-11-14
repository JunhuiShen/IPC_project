#!/usr/bin/env python3
"""Run the primary Newton/Warp GPU validation suite through CTest.

Forward the ``gpu_solver`` label and return CTest's status. Configure and build
the primary gpu/ project first. Its tests cover equations, model/scene mapping,
captured execution and the requirement for a real CUDA runtime.
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
