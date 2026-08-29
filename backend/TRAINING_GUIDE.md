# TRAINING_GUIDE.md — VTHStockPilot Real-Data Training

End-to-end guide to train the model on real data from the Vietnam stock market (HOSE).

---

## Data Pipeline Overview

```
[vnstock3 / HOSE]
       ↓
 Step 1: collect_training_data.py
       ↓
 data/raw/
   VCB_ohlcv.parquet      ← Daily OHLCV price
   VCB_ratios.parquet     ← P/E, P/B, ROE, D/E quarterly
   ...
   vnindex_ohlcv.parquet  ← VN-Index Benchmark
       ↓
 data/processed/
   training_dataset.parquet
   (date, ticker, momentum, quality, ..., future_excess_return)
       ↓
 Step 2: train_model.py
       ↓
 MLflow Experiment: stock-cross-sectional-ranking
   Run: HistGBM + LightGBM  → metrics: NDCG, RankIC, Sharpe
       ↓
 Step 3: promote_model.py
       ↓
 MLflow Model Registry: stock-ranking-model/Production
       ↓
 Forecast Node (LangGraph Pipeline)
   → expected_excess_return (model prediction)
```

---

## Install Dependencies

```bash
# In backend/ directory
pip install vnstock3          # HOSE real-time + historical data
pip install mlflow            # Experiment tracking + model registry
pip install lightgbm          # Challenger model (better than HistGBM)
pip install scikit-learn      # Champion model + metrics
pip install shap              # Feature importance (optional)
pip install pyarrow           # Parquet I/O
```

Or install everything:
```bash
pip install -r backend/requirements-fastapi.txt
```

---

## Find Data Sources

### 1. vnstock3 (Recommended — Free, integrated)

```python
from vnstock import Vnstock

# VCB historical prices
stock = Vnstock().stock(symbol="VCB", source="VCI")
hist = stock.quote.history(start="2022-01-01", end="2025-01-01", interval="1D")
print(hist.head())
# → columns: time, open, high, low, close, volume

# Financial ratios
ratio = stock.finance.ratio(period="quarter", lang="en")
print(ratio.head())
# → columns: yearReport, lengthReport, price_to_earning, roe, ...

# VN-Index benchmark
vnindex = Vnstock().stock(symbol="VNINDEX", source="VCI").quote.history(
    start="2022-01-01", end="2025-01-01", interval="1D"
)
```

**vnstock3 Limitations:**
- Maximum history ~3-5 years (depends on VCI/TCBS source)
- 15 tickers × 3 years ≈ 11,000 price rows + 15 × 12 quarters ≈ 180 ratio rows
- Rate limit: needs 1-2s delay between requests

### 2. CAFEF.vn (Scraping — Free, more data)

CAFEF website provides financial data in HTML. You can use BeautifulSoup:

```python
import requests
from bs4 import BeautifulSoup

url = "https://cafef.vn/tai-chinh/VCB/ket-qua-kinh-doanh.chn"
resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
soup = BeautifulSoup(resp.text, "html.parser")
# parse table...
```

> **Note:** Must add delay between requests, do not scrape too fast.

### 3. VietstockFinance API (Paid — Most in-depth)

- URL: https://finance.vietstock.vn/
- 10+ years history, intraday, all stocks
- Offers a trial plan

### 4. SSI Research API (Paid)

- URL: https://iboard.ssi.com.vn/
- High-quality real-time + historical data

---

## Step 1 — Collect Historical Data

```bash
cd backend

# Collect 3 years of history for all tickers in config
python scripts/collect_training_data.py

# Collect specific tickers and 2 years of history
python scripts/collect_training_data.py \
    --tickers VCB,BID,MBB,TCB,VPB,FPT,HPG,VNM \
    --years 2

# Skip already downloaded tickers (for incremental update)
python scripts/collect_training_data.py --skip-existing

# Only download data, do not build labels
python scripts/collect_training_data.py --no-build-labels
```

**Output:**
```
data/raw/
├── VCB_ohlcv.parquet    (e.g., 760 rows × 6 cols)
├── VCB_ratios.parquet   (e.g., 12 rows × 15 cols)
├── BID_ohlcv.parquet
├── BID_ratios.parquet
├── ...
├── vnindex_ohlcv.parquet
└── _manifest.json       (fetch results summary)

data/processed/
└── training_dataset.parquet  (full panel dataset with labels)
```

**Verify dataset:**
```python
import pandas as pd
df = pd.read_parquet("data/processed/training_dataset.parquet")
print(df.info())
print(df["future_excess_return"].describe())
print(df.groupby("ticker").size())
```

---

## Step 2 — Train Model with MLflow

### 2a. Start MLflow server (separate terminal)

```bash
cd backend
python scripts/start_mlflow.py
# → Open http://localhost:5000 in browser
```

Or use file-based (no server needed):
```bash
mlflow ui --backend-store-uri mlruns/
```

### 2b. Train model

```bash
cd backend
python scripts/train_model.py
```

**Sample Output:**
```
================================================================
  MLflow Run ID : a3b7c1d2e4f5...
  Model         : LightGBM
  Data source   : vnstock_real
  NDCG (avg)    : 0.7821
  RankIC (avg)  : 0.3456
  Sharpe (avg)  : 1.2340
  Registered    : stock-ranking-model v3
================================================================
```

### 2c. View Results in MLflow UI

Open http://localhost:5000 → Experiment "stock-cross-sectional-ranking"

Metrics to check:

| Metric | Meaning | Good Target |
|--------|---------|------------|
| `lgb_ndcg` | Ranking quality | > 0.75 |
| `lgb_rank_ic` | Spearman correlation | > 0.15 |
| `lgb_pseudo_sharpe` | Risk-adjusted return | > 0.8 |
| `lgb_precision_at_k` | Top-K accuracy | > 0.4 |

---

## Step 3 — Promote Model to Production

```bash
cd backend

# List trained versions
python scripts/promote_model.py --list

# Promote a specific run to Production
python scripts/promote_model.py --run-id a3b7c1d2e4f5...

# Promote by version number
python scripts/promote_model.py --version 3
```

---

## Step 4 — Enable MLflow Model in Pipeline

In the `.env` file (or `backend/.env`):

```env
STOCK_USE_MLFLOW_MODEL=true
STOCK_MLFLOW_MODEL_STAGE=Production
STOCK_MLFLOW_TRACKING_URI=http://localhost:5000
```

Restart backend after changes:
```bash
uvicorn app.main:app --reload
```

**Check if the model is being used:**
Call the API and check `pipeline_stages` in the response:
```json
{
  "pipeline_stages": [
    ...,
    {
      "node": "forecast",
      "forecast_method": "mlflow_model",
      "assets_forecast": 15,
      "mean_excess_return": 0.0432
    }
  ]
}
```

If `forecast_method = "linear"`, check logs for model load errors.

---

## New File Structure

```
backend/
├── app/
│   ├── etl/
│   │   ├── historical_data.py    ← NEW: Download OHLCV + ratios
│   │   ├── label_builder.py      ← NEW: Calculate true future_excess_return
│   │   └── feature_eng.py        ← EDITED: Added build_training_dataframe()
│   ├── ml/
│   │   └── train.py              ← EDITED: Use real data, added Sharpe metric
│   ├── workflows/nodes/
│   │   └── forecast.py           ← EDITED: MLflow model inference
│   └── core/
│       └── config.py             ← EDITED: use_mlflow_model, mlflow_model_stage
├── scripts/
│   ├── collect_training_data.py  ← NEW: CLI step 1
│   ├── train_model.py            ← NEW: CLI step 2
│   ├── promote_model.py          ← NEW: CLI step 3
│   └── start_mlflow.py           ← NEW: Start MLflow UI
└── .env.example                  ← EDITED: Added new variables
```

---

## Troubleshooting

### "vnstock3 not installed"
```bash
pip install vnstock3
```

### "Training dataset not found"
Run step 1 first:
```bash
cd backend
python scripts/collect_training_data.py
```

### "MLflow model load failed"
Check:
1. MLflow server is running: `python scripts/start_mlflow.py`
2. Model is trained: `python scripts/train_model.py`
3. Promoted: `python scripts/promote_model.py --list`
4. Config is correct: `STOCK_USE_MLFLOW_MODEL=true`

### Low NDCG (< 0.65)
- Increase `--years` for more data
- Add tickers to `STOCK_VNSTOCK_TICKERS`
- Try `--horizon 63` (3 months, less noise)
- Check if data has enough rows (min 500+ rows)

### Demo mode (no internet or vnstock)
```bash
cd backend
python scripts/train_model.py --demo-fallback
```
Model will train on synthetic data — enough to test the pipeline but no real predictive value.
