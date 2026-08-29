"""Applies the Windows cert-store SSL fix to the MLflow server subprocess.

MLflow >= 3.15 launches uvicorn via subprocess, so a monkeypatch applied in
start_mlflow.py never reaches the server. This directory is prepended to
PYTHONPATH for that child, and Python imports sitecustomize automatically at
interpreter startup — before uvicorn imports aiohttp.

Guarded by MLFLOW_WIN_CERT_PATCH so unrelated Python processes that happen to
inherit this PYTHONPATH are unaffected. See _cert_fix.py for the rationale.
"""

import os

if os.environ.get("MLFLOW_WIN_CERT_PATCH") == "1":
    import _cert_fix

    _cert_fix.apply()
