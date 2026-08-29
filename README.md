# VTH-StockPilot — Vietnam Portfolio Recommendation Platform

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-7C3AED?style=for-the-badge&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-02A88E?style=for-the-badge&logo=python&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-232F3E?style=for-the-badge&logo=amazon%20aws&logoColor=white)
![GitHub Actions](https://github.com/Nam-gu/Stock-platform/actions/workflows/test-build.yml/badge.svg)

Personalized stock portfolio recommendation platform for the Vietnamese stock market (HOSE/HNX/UPCOM).

---

## About The Project

### Overview

* Collects and analyzes Vietnamese stock data from HOSE via the **vnstock3** library, providing multi-dimensional information for investors.
* Implements a **LangGraph** 9-node agentic pipeline for explainable portfolio recommendations: data quality → market regime → fundamental analysis → forecast → customer preference → risk compliance → ranking → portfolio optimizer → explanation.
* Uses **LightGBM** cross-sectional ranking with SHAP values to explain the reasoning behind each stock selection (Explainable AI).
* Tracks and compares ML models via **MLflow** experiment tracking.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       VTH-StockPilot                        │
│                    Vietnam Market Edition                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────▼──────────────┐
        │   Frontend (Vanilla JS)      │
        │   HOSE universe · VND units  │
        └──────────────┬──────────────┘
                       │ REST / WebSocket
        ┌──────────────▼──────────────┐
        │   Backend (FastAPI)          │
        │   LangGraph 9-node workflow  │
        │   LightGBM + SHAP ranking    │
        └──────┬───────────────┬──────┘
               │               │
  ┌────────────▼─────┐  ┌──────▼───────────┐
  │ Database (SQLite/ │  │  MLflow Tracking  │
  │ PostgreSQL prod)  │  │  (Experiment log) │
  └────────────┬─────┘  └──────────────────┘
               │
  ┌────────────▼─────────────────────────────┐
  │  vnstock3 — Vietnamese Market Data        │
  │  HOSE · HNX · UPCOM · VN30 · VN100       │
  └──────────────────────────────────────────┘
```

---

## System Configuration

| Component  | Tools                        | Description |
|:----------:|:----------------------------:|-------------|
| Frontend   | Vanilla JS / CSS / Chart.js  | 5 views: Recommendations, Portfolio, Backtest, Model Performance, Preferences. Displays portfolio in VND units (bil/mil). |
| Backend    | FastAPI + LangGraph          | 9-node agentic pipeline. 5 REST routes + WebSocket streaming. Cross-sectional ML ranking (LightGBM + SHAP). |
| Market Data | vnstock3                    | HOSE real-time data. Demo mode: 15 VN30 blue-cap stocks (VCB, FPT, HPG, VIC, VNM...). |
| Database   | SQLite (dev) / PostgreSQL (prod) | Saves portfolio states, recommendation history. SQLAlchemy async + Alembic migrations. |
| ML         | LightGBM / scikit-learn / SHAP | Walk-forward cross-sectional ranking. SHAP explainability for each stock. |
| Experiments | MLflow                      | Tracking champion/challenger models. Metrics: NDCG, RankIC, Precision@K, Sharpe. |
| CI/CD      | GitHub Actions / AWS CodeDeploy | Automated testing and deployment. |

---

## Vietnamese Market Universe (Demo)

| Symbol | Company | Exchange | Sector |
|--------|---------|-----|-------|
| VCB    | Vietcombank | HOSE | Banking |
| BID    | BIDV | HOSE | Banking |
| MBB    | MB Bank | HOSE | Banking |
| TCB    | Techcombank | HOSE | Banking |
| VPB    | VPBank | HOSE | Banking |
| ACB    | Asia Commercial Bank | HOSE | Banking |
| FPT    | FPT Corporation | HOSE | Technology |
| VIC    | Vingroup | HOSE | Real Estate |
| VHM    | Vinhomes | HOSE | Real Estate |
| HPG    | Hoa Phat Group | HOSE | Materials |
| VNM    | Vinamilk | HOSE | Consumer |
| SAB    | Sabeco | HOSE | Consumer |
| GAS    | PetroVietnam Gas | HOSE | Energy |
| MSN    | Masan Group | HOSE | Consumer |
| REE    | REE Corporation | HOSE | Industrials |

---

## LangGraph Workflow (9 Nodes)

```
data_quality
    │
    ▼
market_regime       ← Classifies bull / bear / high_volatility / neutral
    │
    ▼
fundamental         ← Scores ROE, P/E, P/B, revenue growth
    │
    ▼
forecast            ← Predicts excess return using cross-sectional ranking
    │
    ▼
customer_preference ← Adjusts based on preferred sectors, ESG
    │
    ▼
risk_compliance     ← Filters by volatility, D/E, suitability
    │
    ▼
ranking             ← LightGBM + SHAP scoring
    │
    ▼
portfolio_optimizer ← Mean-variance optimization, sets position limits
    │
    ▼
explanation         ← Synthesizes key drivers, risk flags, SHAP contributions
```

---

## Platform Features

### 1. Recommendations View
* Input investor profile: capital (VND), risk appetite, horizon, preferred sectors.
* Click **Generate recommendation** → 9-node pipeline runs, showing real-time progress via WebSocket.
* Results: ranked stocks + portfolio allocation + SHAP explanation.

### 2. Portfolio Builder
* Customize portfolio weights manually or follow recommendations.
* Displays Value-at-Risk (VaR 95%), Sortino ratio, and correlation matrix.

### 3. Backtest
* Walk-forward simulation with transaction costs (bps).
* Benchmark: **VNINDEX**.
* Charts: equity curve, underwater curve, rolling Sharpe.

### 4. Model Performance
* View MLflow experiments history: NDCG, RankIC, Precision@K.
* Compare champion (LightGBM) vs challenger (HistGBM).

### 5. Preferences
* Record stock feedback to personalize future recommendations.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend API** | FastAPI 0.115, Uvicorn, Pydantic v2 |
| **Agentic Workflow** | LangGraph 0.2, LangChain Core |
| **ML** | LightGBM 4.5, scikit-learn, SHAP |
| **Experiment Tracking** | MLflow 2.16 |
| **Database** | SQLAlchemy 2.0 (async), Alembic, PostgreSQL / SQLite |
| **Cache** | Redis 5.1 |
| **Market Data** | vnstock3 (HOSE/HNX/UPCOM) |
| **Frontend** | Vanilla HTML/CSS/JS, Chart.js |
| **Scheduling** | APScheduler, Celery (optional heavy workers) |
| **Infra** | Docker, AWS EC2 + CodeDeploy, GitHub Actions |

---

## Quick Start

### Requirements
```bash
pip install -r backend/requirements-fastapi.txt
# Optional: live market data
pip install vnstock3
```

### Run (Development)
```bash
# Backend
cd backend
uvicorn app.main:app --reload --port 8000

# Or with Docker Compose
docker compose up
```

### Environment Variables
```env
STOCK_DATABASE_URL=sqlite+aiosqlite:///./vth_stockpilot.db
STOCK_USE_REAL_MARKET_DATA=false          # true → vnstock live
STOCK_MARKET_DATA_SOURCE=demo             # demo | vnstock
STOCK_VNSTOCK_MARKET=HOSE
STOCK_RISK_FREE_RATE=0.045                # 4.5% — 5-year VN Govt Bond
STOCK_LLM_PROVIDER=template              # template | openai | gemini
STOCK_OPENAI_API_KEY=sk-...
```

---

## Related

* **ML Model Repository**: https://github.com/Nam-gu/Personalized-stock-portfolio-model
* **vnstock3 docs**: https://docs.vnstock.site
