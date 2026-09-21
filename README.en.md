# Attention Meta-Learner for MOEX Si Futures

Bachelor's thesis project (2026) on short-term price-direction forecasting for the Si futures contract traded on the Moscow Exchange.

The project studies an adaptive ensemble in which an attention-based meta-model changes the contribution of several base predictors depending on market conditions.

## Scope

- minute OHLCV data for the Si futures contract;
- feature engineering for returns, volatility, momentum and volume;
- CatBoost, LightGBM and experimental neural-network proxy models;
- attention-based meta-learning;
- time-aware holdout and purged cross-validation;
- vectorised research backtest;
- Streamlit dashboard for reviewing results.

## Dataset

The experiment uses 219,529 minute bars covering May 2025 to April 2026. The final holdout contains 7,550 observations.

## Repository structure

```text
.
├── app.py
├── assets/
├── data/
├── docs/
│   ├── thesis.pdf
│   └── presentation.pdf
├── notebooks/
│   └── attention_meta_learner.ipynb
├── tests/
├── requirements.txt
└── requirements-research.txt
```

## Dashboard

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate

python -m pip install -r requirements.txt
streamlit run app.py
```

## Research notes

The reported performance is based on a historical research backtest with project-specific assumptions. It should not be interpreted as verified live-trading performance or investment advice. The backtest does not fully model slippage, latency and liquidity, and some market-microstructure variables are approximated from OHLCV data.

[Thesis PDF](docs/thesis.pdf) · [Presentation](docs/presentation.pdf) · [Research notebook](notebooks/attention_meta_learner.ipynb)
