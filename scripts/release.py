#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIST_DIR = REPO_ROOT / "dist"
BUILD_DIR = REPO_ROOT / "build"
FORMAT_TARGETS = ("scripts", "setup.py", "trade_assist", "tests")


class ReleaseToolError(RuntimeError):
    """User-facing setup or execution issue for the release helper."""


def _run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink()


def _clean_build_artifacts(dist_dir: Path) -> None:
    _remove_path(BUILD_DIR)
    _remove_path(dist_dir)
    for egg_info_dir in REPO_ROOT.glob("*.egg-info"):
        _remove_path(egg_info_dir)


def _require_module(module_name: str, purpose: str) -> None:
    if importlib.util.find_spec(module_name) is not None:
        return
    raise ReleaseToolError(
        f"Missing required package '{module_name}' for {purpose}. "
        'Run `make dev-setup` or `pip install -e ".[dev]"` in your virtualenv '
        "and try again."
    )


def run_release(
    *,
    dist_dir: Path,
    skip_format: bool,
    skip_tests: bool,
) -> None:
    if not skip_format:
        _require_module("black", "formatting")
        _run([sys.executable, "-m", "black", *FORMAT_TARGETS])

    if not skip_tests:
        _require_module("pytest", "testing")
        _run([sys.executable, "-m", "pytest", "-q"])

    _require_module("build", "building release artifacts")
    _require_module("setuptools", "building release artifacts")
    _require_module("wheel", "building release artifacts")
    _clean_build_artifacts(dist_dir)
    dist_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(dist_dir),
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="release.py",
        description="Format, test, and build release artifacts for trade_assist.",
    )
    parser.add_argument(
        "--dist-dir",
        default=str(DEFAULT_DIST_DIR),
        help="Directory to write release artifacts into.",
    )
    parser.add_argument(
        "--skip-format",
        action="store_true",
        help="Skip auto-formatting before building.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip tests before building.",
    )
    args = parser.parse_args(argv)

    try:
        run_release(
            dist_dir=Path(args.dist_dir).resolve(),
            skip_format=bool(args.skip_format),
            skip_tests=bool(args.skip_tests),
        )
        return 0
    except ReleaseToolError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as exc:
        return int(exc.returncode or 1)


if __name__ == "__main__":
    raise SystemExit(main())
