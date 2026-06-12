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

# Pinned to the verified-working set (the exact versions the sandbox layer uses).
# IMPORTANT: target manylinux_2_28 ONLY. Lambda's Python 3.12 runtime is Amazon
# Linux 2023 (glibc 2.34), so 2_28 wheels load fine. Keeping manylinux2014 made
# pip backtrack to numpy 2.2.6 (the newest numpy that still ships a 2014 wheel),
# whose import then breaks once prune() removes the bundled tests/ dirs
# (Runtime.ImportModuleError: No module named 'numpy._core.tests'). 2.4.6 is fine.
PACKAGES = ["numpy==2.4.6", "scipy==1.17.1", "onnxruntime==1.26.0"]
PY_VERSION = "3.12"
PLATFORMS = ["manylinux_2_28_x86_64"]
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
        # Cross-build for the Lambda 3.12 runtime: the wheels selected are cp312,
        # so the machine running pip can be any version. Old pip (e.g. 23.0.1 on
        # the Amplify build image = Python 3.10) checks a package's Requires-Python
        # against the HOST interpreter and would reject numpy 2.4.6 (needs >=3.11).
        # Skip that host gate — the downloaded cp312 wheels are what matters.
        "--ignore-requires-python",
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
