"""Helper script — start MLflow tracking server locally.

Usage:
    cd backend
    python scripts/start_mlflow.py

    # Custom host/port:
    python scripts/start_mlflow.py --port 5001

    # Use S3 for artifact storage:
    python scripts/start_mlflow.py --artifact-root s3://my-bucket/mlflow

Then open: http://localhost:5000
"""
from __future__ import annotations

import os
import sys

# MLflow >= 3.15 starts uvicorn as a subprocess, so patches applied in this
# process do not reach the server child. Export the sitecustomize-based ssl fix
# (corrupt Windows cert store) to it via PYTHONPATH, and apply it here too --
# the mlflow.cli import below also pulls in aiohttp.
_PATCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_win_cert_patch")
_PYTHONPATH = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = _PATCH_DIR + os.pathsep + _PYTHONPATH if _PYTHONPATH else _PATCH_DIR
os.environ["MLFLOW_WIN_CERT_PATCH"] = "1"

sys.path.insert(0, _PATCH_DIR)
import _cert_fix

_cert_fix.apply()

import argparse
from mlflow.cli import cli


def main() -> None:
    parser = argparse.ArgumentParser(description="Start MLflow tracking UI")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Port (default: 5000)")
    parser.add_argument(
        "--backend-store",
        default="mlruns",
        help="Backend store URI (default: mlruns/ — local file). "
             "Use sqlite:///mlflow.db for SQLite.",
    )
    parser.add_argument(
        "--artifact-root",
        default="mlartifacts",
        help="Artifact root URI (default: mlartifacts/ — local). "
             "Use s3://bucket/path for S3.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Starting MLflow UI")
    print(f"  URL      : http://{args.host}:{args.port}")
    print(f"  Backend  : {args.backend_store}")
    print(f"  Artifacts: {args.artifact_root}")
    print("=" * 60)
    print("\n  Press Ctrl+C to stop.\n")

    sys.argv = [
        "mlflow", "ui",
        "--host", args.host,
        "--port", str(args.port),
        "--backend-store-uri", args.backend_store,
        "--default-artifact-root", args.artifact_root,
    ]
    
    try:
        sys.exit(cli())
    except KeyboardInterrupt:
        print("\nMLflow UI stopped.")


if __name__ == "__main__":
    main()
