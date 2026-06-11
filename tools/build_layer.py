"""Build the Lambda dependency layer (numpy/scipy/onnxruntime) for the
Python 3.12 Lambda runtime WITHOUT Docker, by downloading manylinux wheels.

Output: amplify/layers/pydeps/python/<packages>
Run with any local Python:  python tools/build_layer.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

PACKAGES = ["numpy", "scipy", "onnxruntime"]
PY_VERSION = "3.12"
PLATFORMS = ["manylinux2014_x86_64", "manylinux_2_28_x86_64"]
TARGET = Path("amplify/layers/pydeps/python")

# directories/files to prune to stay under Lambda's 250 MB unzipped limit
PRUNE_DIR_NAMES = {"__pycache__", "tests", "test"}
PRUNE_SUFFIXES = {".pyc", ".pyi"}


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)


def build() -> None:
    if TARGET.exists():
        shutil.rmtree(TARGET)
    TARGET.mkdir(parents=True)

    cmd = [
        sys.executable, "-m", "pip", "install",
        "--target", str(TARGET),
        "--only-binary=:all:",
        "--python-version", PY_VERSION,
        "--implementation", "cp",
    ]
    for p in PLATFORMS:
        cmd += ["--platform", p]
    cmd += PACKAGES
    run(cmd)


def prune() -> None:
    removed = 0
    for path in sorted(TARGET.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and path.name in PRUNE_DIR_NAMES:
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
        elif path.is_file() and path.suffix in PRUNE_SUFFIXES:
            path.unlink(missing_ok=True)
            removed += 1
    print(f"pruned {removed} items")


def size_mb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def main() -> None:
    build()
    prune()
    total = size_mb(TARGET)
    print(f"\nlayer unzipped size = {total:.1f} MB (Lambda limit: function+layers <= 250 MB)")
    for child in sorted(TARGET.iterdir()):
        if child.is_dir():
            print(f"  {child.name:28s} {size_mb(child):7.1f} MB")
    if total > 235:
        print("WARNING: close to / over the 250 MB limit; further pruning needed.")


if __name__ == "__main__":
    main()
