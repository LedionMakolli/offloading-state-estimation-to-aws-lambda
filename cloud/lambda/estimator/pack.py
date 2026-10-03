"""Build a deployable Lambda zip.

Single source of truth: eskf.py and rotations.py are copied from vimu-gen at
package time rather than duplicated in this directory, so the code that runs in
Lambda is provably the same code that E2b validated on the host.

numpy is NOT bundled. Two supported options, both documented in README.md:
  * attach an AWS-provided numpy/scipy layer (simplest), or
  * pip install --platform manylinux2014_x86_64 into the package.
Bundling a Windows numpy build would produce a zip that fails at import time in
Lambda, which is a slow and confusing failure to debug.

Usage:  python pack.py [--out estimator.zip] [--with-numpy]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
VIMU = REPO / "vimu-gen"

SHARED = ["eskf.py", "rotations.py"]
LOCAL = ["handler.py"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=HERE / "estimator.zip")
    ap.add_argument("--with-numpy", action="store_true",
                    help="vendor a manylinux numpy into the zip (needs pip)")
    ap.add_argument("--arch", default="manylinux2014_x86_64",
                    help="use manylinux2014_aarch64 for Graviton/arm64 Lambdas")
    args = ap.parse_args()

    missing = [f for f in SHARED if not (VIMU / f).exists()]
    if missing:
        print(f"ERROR: missing in {VIMU}: {', '.join(missing)}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as td:
        stage = Path(td) / "pkg"
        stage.mkdir()

        for f in SHARED:
            shutil.copy2(VIMU / f, stage / f)
        for f in LOCAL:
            shutil.copy2(HERE / f, stage / f)

        if args.with_numpy:
            print(f"vendoring numpy for {args.arch} ...")
            rc = subprocess.call([
                sys.executable, "-m", "pip", "install",
                "--platform", args.arch,
                "--target", str(stage),
                "--implementation", "cp",
                "--only-binary=:all:",
                "--upgrade", "numpy",
            ])
            if rc != 0:
                print("ERROR: numpy vendoring failed", file=sys.stderr)
                return 1

        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.out.exists():
            args.out.unlink()
        with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    z.write(path, path.relative_to(stage))

    size_kb = args.out.stat().st_size / 1024
    print(f"wrote {args.out}  ({size_kb:.1f} kB)")
    if not args.with_numpy:
        print("numpy not bundled - attach a numpy layer, or re-run with --with-numpy")
    print("handler = handler.lambda_handler")
    return 0


if __name__ == "__main__":
    sys.exit(main())
