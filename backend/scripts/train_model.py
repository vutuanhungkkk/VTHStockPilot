"""CLI script — Step 2: Train ranking models and register in MLflow.

Loads the training dataset produced by collect_training_data.py,
trains HistGBM + LightGBM models via walk-forward cross-validation,
logs all metrics to MLflow, and registers the best model.

Usage:
    cd backend
    python scripts/train_model.py

    # With custom options:
    python scripts/train_model.py \\
        --processed-dir data/processed \\
        --mlflow-uri http://localhost:5000 \\
        --demo-fallback

    # Start MLflow UI to view results:
    python scripts/start_mlflow.py

Requirements:
    pip install mlflow scikit-learn lightgbm scipy shap

MLflow experiment: stock-cross-sectional-ranking
Model registry  : stock-ranking-model (configurable via STOCK_MODEL_NAME env var)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# ── Make the backend/app package importable ────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train stock ranking model and register in MLflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Directory containing training_dataset.parquet (default: data/processed).",
    )
    parser.add_argument(
        "--mlflow-uri",
        default=None,
        help=(
            "MLflow tracking URI (default: STOCK_MLFLOW_TRACKING_URI from config, "
            "or http://localhost:5000). Use 'mlruns' for file-based local tracking."
        ),
    )
    parser.add_argument(
        "--demo-fallback",
        action="store_true",
        help="If real training data is missing, use synthetic demo data instead of failing.",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Path to a specific training parquet file (overrides --processed-dir).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)

    # Allow --mlflow-uri to override the setting before importing config
    if args.mlflow_uri:
        os.environ["STOCK_MLFLOW_TRACKING_URI"] = args.mlflow_uri

    try:
        from app.core.config import get_settings
        settings = get_settings()
    except Exception as exc:
        logger.error("Failed to load settings: %s", exc)
        sys.exit(1)

    print(f"\n{'='*65}")
    print(f"  VTHStockPilot — Model Training")
    print(f"{'='*65}")
    print(f"  MLflow URI   : {settings.mlflow_tracking_uri}")
    print(f"  Experiment   : stock-cross-sectional-ranking")
    print(f"  Model name   : {settings.model_name}")
    print(f"  Data dir     : {processed_dir.resolve()}")
    print(f"  Demo fallback: {args.demo_fallback}")
    print(f"{'='*65}\n")

    # Load a specific data file if provided
    data = None
    if args.data_path:
        import pandas as pd
        path = Path(args.data_path)
        if not path.exists():
            logger.error("Data file not found: %s", path)
            sys.exit(1)
        logger.info("Loading data from explicit path: %s", path)
        data = pd.read_parquet(path)

    t0 = time.time()
    try:
        from app.ml.train import train
        run_id = train(
            data=data,
            processed_dir=processed_dir,
            use_demo_fallback=args.demo_fallback,
        )
        elapsed = time.time() - t0
        print(f"\n{'='*65}")
        print(f"  ✓ Training complete! ({elapsed:.1f}s)")
        print(f"  Run ID  : {run_id}")
        print(f"  View UI : {settings.mlflow_tracking_uri}")
        print(f"\n  To promote to Production, run:")
        print(f"    python scripts/promote_model.py --run-id {run_id}")
        print(f"{'='*65}\n")
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        logger.error(
            "Tip: run 'python scripts/collect_training_data.py' first, "
            "or use --demo-fallback for synthetic data."
        )
        sys.exit(1)
    except Exception as exc:
        logger.error("Training failed: %s", exc)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
