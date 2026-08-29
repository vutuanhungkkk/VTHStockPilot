"""CLI script — Step 1: Collect historical HOSE data for model training.

Downloads multi-year OHLCV price history + quarterly financial ratios
from vnstock3, then builds the training dataset with forward return labels.

Usage:
    cd backend
    python scripts/collect_training_data.py

    # With options:
    python scripts/collect_training_data.py \\
        --tickers VCB,BID,MBB,TCB,VPB,FPT,HPG,VNM \\
        --years 3 \\
        --horizon 21 \\
        --raw-dir data/raw \\
        --processed-dir data/processed \\
        --skip-existing

Requirements:
    pip install vnstock3 pandas pyarrow

Output files:
    data/raw/
        <TICKER>_ohlcv.parquet    — daily OHLCV per ticker
        <TICKER>_ratios.parquet   — quarterly fundamental ratios
        vnindex_ohlcv.parquet     — VN-Index benchmark
        _manifest.json            — fetch summary
    data/processed/
        training_dataset.parquet  — panel dataset ready for training

Next step:
    python scripts/train_model.py
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ── Make sure the backend/app package is importable ───────────────────────────
_HERE = Path(__file__).resolve().parent        # backend/scripts/
_BACKEND = _HERE.parent                         # backend/
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("collect_training_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Vietnamese market data for model training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tickers",
        default=None,
        help=(
            "Comma-separated HOSE tickers. "
            "Defaults to STOCK_VNSTOCK_TICKERS in config / .env"
        ),
    )
    parser.add_argument(
        "--years",
        type=float,
        default=3.0,
        help="Years of price history to download (default: 3). vnstock3 supports ~5 years.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=21,
        help="Forward return horizon in trading days for label (default: 21 ≈ 1 month).",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Directory to save raw parquet files (default: data/raw).",
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Directory to save processed training dataset (default: data/processed).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip downloading tickers whose parquet files already exist.",
    )
    parser.add_argument(
        "--no-build-labels",
        action="store_true",
        help="Only download raw data; skip building the training dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Resolve tickers ────────────────────────────────────────────────────────
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        try:
            from app.core.config import get_settings
            settings = get_settings()
            tickers = [t.strip() for t in settings.vnstock_tickers.split(",") if t.strip()]
            logger.info("Tickers from config: %s", ", ".join(tickers))
        except Exception:
            tickers = "VCB,BID,MBB,TCB,VPB,ACB,FPT,VIC,VHM,HPG,VNM,SAB,GAS,MSN,REE".split(",")
            logger.info("Using default ticker list: %s", ", ".join(tickers))

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)

    print(f"\n{'='*65}")
    print(f"  VTHStockPilot — Training Data Collection")
    print(f"{'='*65}")
    print(f"  Tickers      : {', '.join(tickers)}")
    print(f"  History      : {args.years:.1f} years")
    print(f"  Label horizon: {args.horizon} trading days (~{args.horizon//21} month(s))")
    print(f"  Raw output   : {raw_dir.resolve()}")
    print(f"  Processed    : {processed_dir.resolve()}")
    print(f"{'='*65}\n")

    t0 = time.time()

    # ── Step 1: Download raw data ─────────────────────────────────────────────
    logger.info("Step 1/2 — Downloading raw OHLCV + financial ratios ...")
    try:
        from app.etl.historical_data import collect_all
        manifest = collect_all(
            tickers=tickers,
            years=args.years,
            raw_dir=raw_dir,
            skip_existing=args.skip_existing,
        )
        success_count = sum(
            1 for r in manifest["results"].values()
            if r.get("ohlcv", {}).get("rows", 0) > 0
        )
        logger.info(
            "Download complete: %d/%d tickers OK (%.1fs)",
            success_count, len(tickers), time.time() - t0,
        )
    except ImportError as exc:
        logger.error("vnstock3 not installed: %s", exc)
        logger.error("Install with:  pip install vnstock3")
        sys.exit(1)
    except Exception as exc:
        logger.error("Data collection failed: %s", exc)
        sys.exit(1)

    if args.no_build_labels:
        logger.info("--no-build-labels set; skipping label construction.")
        return

    # ── Step 2: Build training dataset with labels ─────────────────────────────
    logger.info("Step 2/2 — Building training dataset with forward return labels ...")
    try:
        from app.etl.label_builder import build_training_dataset
        dataset = build_training_dataset(
            tickers=tickers,
            horizon_days=args.horizon,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
        )
        elapsed = time.time() - t0
        print(f"\n{'='*65}")
        print(f"  ✓ Training dataset ready!")
        print(f"  Rows     : {len(dataset):,}")
        print(f"  Tickers  : {dataset['ticker'].nunique()}")
        print(f"  Path     : {(processed_dir / 'training_dataset.parquet').resolve()}")
        print(f"  Time     : {elapsed:.1f}s")
        print(f"\n  Next step:")
        print(f"    python scripts/train_model.py")
        print(f"{'='*65}\n")
    except Exception as exc:
        logger.error("Label building failed: %s", exc)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
