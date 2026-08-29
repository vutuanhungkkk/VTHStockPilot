"""Model promotion script — transition a run's model to Production stage.

After training, models land in 'Staging'. Use this script to promote
the best run to 'Production' so that the forecast node can load it.

Usage:
    cd backend
    python scripts/promote_model.py --run-id <RUN_ID>

    # Or promote by version number:
    python scripts/promote_model.py --version 3

    # Inspect before promoting:
    python scripts/promote_model.py --list
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote MLflow model to Production")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run-id", help="MLflow run ID (promotes the model from that run)")
    group.add_argument("--version", type=int, help="Model Registry version number to promote")
    group.add_argument("--list", action="store_true", help="List all registered model versions")
    parser.add_argument(
        "--target-stage",
        default="Production",
        choices=["Staging", "Production", "Archived"],
        help="Target stage (default: Production)",
    )
    args = parser.parse_args()

    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        from app.core.config import get_settings
    except ImportError as exc:
        print(f"Import error: {exc}\nInstall with: pip install mlflow")
        sys.exit(1)

    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()
    model_name = settings.model_name

    # ── List mode ──────────────────────────────────────────────────────────────
    if args.list:
        try:
            versions = client.search_model_versions(f"name='{model_name}'")
        except Exception as exc:
            print(f"Could not list model versions: {exc}")
            print("Make sure MLflow server is running: python scripts/start_mlflow.py")
            sys.exit(1)

        print(f"\n{'='*70}")
        print(f"  Registered versions for: {model_name}")
        print(f"{'='*70}")
        print(f"  {'Ver':>4}  {'Stage':<14} {'Run ID':<36} {'Created'}")
        print(f"  {'-'*4}  {'-'*14} {'-'*36} {'-'*20}")
        for v in sorted(versions, key=lambda x: int(x.version)):
            print(f"  {v.version:>4}  {v.current_stage:<14} {v.run_id:<36} {v.creation_timestamp}")
        print(f"{'='*70}\n")
        return

    # ── Resolve target version ─────────────────────────────────────────────────
    target_version: str | None = None

    if args.version:
        target_version = str(args.version)

    elif args.run_id:
        versions = client.search_model_versions(f"name='{model_name}'")
        matches = [v for v in versions if v.run_id == args.run_id]
        if not matches:
            print(f"No registered model found for run_id={args.run_id}")
            sys.exit(1)
        target_version = matches[0].version

    else:
        parser.print_help()
        sys.exit(0)

    # ── Archive existing Production models ────────────────────────────────────
    if args.target_stage == "Production":
        existing_prod = client.search_model_versions(
            filter_string=f"name='{model_name}'"
        )
        existing_prod = [v for v in existing_prod if v.current_stage == 'Production']
        for v in existing_prod:
            if v.version != target_version:
                client.transition_model_version_stage(
                    name=model_name, version=v.version, stage="Archived"
                )
                print(f"  Archived previous Production model v{v.version}")

    # ── Promote ───────────────────────────────────────────────────────────────
    client.transition_model_version_stage(
        name=model_name,
        version=target_version,
        stage=args.target_stage,
    )
    print(f"\n✓ Model '{model_name}' v{target_version} promoted to {args.target_stage}")
    print(f"\n  The forecast node will now use this model.")
    print(f"  Make sure STOCK_USE_MLFLOW_MODEL=true in your .env\n")


if __name__ == "__main__":
    main()
