"""Interactive dashboard for the Attention Meta-Learner thesis results."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "backtest_results.csv"
BARS_PER_YEAR = 840 * 252

REQUIRED_COLUMNS = {
    "Datetime", "Close", "Ensemble_Signal", "CB_Signal",
    "Equity_Ensemble", "Equity_CB", "Equity_BH",
    "Weight_CB", "Weight_KAN", "Weight_Mamba",
    "Ensemble_PnL", "CB_PnL",
}

st.set_page_config(
    page_title="Attention Meta-Learner | MOEX Si",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp { background: #0d1117; }
        [data-testid="stMetric"] {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 10px;
            padding: 14px;
        }
        .block-container { padding-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    """Load and validate exported holdout backtest results."""
    frame = pd.read_csv(path, parse_dates=["Datetime"])
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"В файле данных отсутствуют столбцы: {sorted(missing)}")
    return frame.sort_values("Datetime").set_index("Datetime")


def annualized_sharpe(series: pd.Series) -> float:
    """Annualized Sharpe ratio under the notebook's bar-frequency assumption."""
    if len(series) < 2 or np.isclose(series.std(), 0):
        return float("nan")
    return float(np.sqrt(BARS_PER_YEAR) * series.mean() / series.std())


def max_drawdown(series: pd.Series) -> float:
    """Maximum drawdown of a cumulative PnL series."""
    equity = series.cumsum()
    return float((equity - equity.cummax()).min())


try:
    full = load_data(DATA_PATH)
except (FileNotFoundError, ValueError) as error:
    st.error(f"Не удалось загрузить результаты: {error}")
    st.stop()

st.title("📈 Attention Meta-Learner · MOEX Si")
st.markdown("**ВКР · Финансовый университет · Прикладная математика и информатика**")
st.caption(
    "Интерактивный анализ holdout-бэктеста адаптивного ансамбля "
    "CatBoost + KAN-proxy + Mamba-proxy."
)

st.sidebar.header("Параметры просмотра")
minimum = full.index.min().to_pydatetime()
maximum = full.index.max().to_pydatetime()
date_range = st.sidebar.slider(
    "Временной диапазон",
    min_value=minimum,
    max_value=maximum,
    value=(minimum, maximum),
    format="DD.MM.YY",
)
st.sidebar.info(
    "Сигналы и PnL рассчитаны заранее в исследовательском ноутбуке "
    "при пороге вероятности 0,55."
)

data = full.loc[(full.index >= date_range[0]) & (full.index <= date_range[1])].copy()
if data.empty:
    st.warning("В выбранном диапазоне нет наблюдений.")
    st.stop()

ensemble_pnl = data["Ensemble_PnL"]
catboost_pnl = data["CB_PnL"]
ensemble_sharpe = annualized_sharpe(ensemble_pnl)
catboost_sharpe = annualized_sharpe(catboost_pnl)

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric(
    "Доходность ансамбля",
    f"{ensemble_pnl.sum() * 100:.2f}%",
    f"CatBoost: {catboost_pnl.sum() * 100:.2f}%",
)
metric_2.metric(
    "Sharpe ансамбля",
    f"{ensemble_sharpe:.2f}",
    f"Δ к CatBoost: {ensemble_sharpe - catboost_sharpe:+.2f}",
)
metric_3.metric("Макс. просадка", f"{max_drawdown(ensemble_pnl) * 100:.2f}%")
metric_4.metric("Наблюдений", f"{len(data):,}".replace(",", " "))

overview_tab, signals_tab, weights_tab, about_tab = st.tabs(
    ["Результат", "Сигналы", "Веса моделей", "О проекте"]
)

with overview_tab:
    st.subheader("Нормализованные кривые капитала")
    equity_chart = go.Figure()
    for column, label, color, width in [
        ("Equity_Ensemble", "Attention Ensemble", "#00ff9d", 2.5),
        ("Equity_CB", "CatBoost baseline", "#ff9900", 1.5),
        ("Equity_BH", "Buy & Hold", "#8b949e", 1.3),
    ]:
        normalized = data[column] - data[column].iloc[0]
        equity_chart.add_trace(go.Scatter(
            x=data.index, y=normalized * 100, mode="lines", name=label,
            line={"color": color, "width": width},
        ))
    equity_chart.update_layout(
        template="plotly_dark", height=520, hovermode="x unified",
        xaxis_title=None, yaxis_title="Накопленная доходность, %",
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        legend={"orientation": "h", "y": 1.08},
    )
    st.plotly_chart(equity_chart, use_container_width=True)

with signals_tab:
    st.subheader("Цена фьючерса и сигналы ансамбля")
    signals_chart = go.Figure()
    signals_chart.add_trace(go.Scatter(
        x=data.index, y=data["Close"], mode="lines", name="Цена Si",
        line={"color": "#8b9dc3", "width": 1},
    ))
    longs = data[data["Ensemble_Signal"] == 1]
    shorts = data[data["Ensemble_Signal"] == -1]
    signals_chart.add_trace(go.Scatter(
        x=longs.index, y=longs["Close"], mode="markers", name="Long",
        marker={"color": "#00ff9d", "size": 7, "symbol": "triangle-up"},
    ))
    signals_chart.add_trace(go.Scatter(
        x=shorts.index, y=shorts["Close"], mode="markers", name="Short",
        marker={"color": "#ff4d8d", "size": 7, "symbol": "triangle-down"},
    ))
    signals_chart.update_layout(
        template="plotly_dark", height=560, hovermode="x unified",
        xaxis_title=None, yaxis_title="Цена",
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        legend={"orientation": "h", "y": 1.08},
    )
    st.plotly_chart(signals_chart, use_container_width=True)

with weights_tab:
    st.subheader("Динамическое распределение весов")
    weights_chart = go.Figure()
    for column, label, color in [
        ("Weight_CB", "CatBoost", "#ff9900"),
        ("Weight_KAN", "KAN-proxy", "#ff4d8d"),
        ("Weight_Mamba", "Mamba-proxy", "#4499ff"),
    ]:
        weights_chart.add_trace(go.Scatter(
            x=data.index,
            y=data[column].rolling(30, min_periods=1).mean() * 100,
            mode="lines", stackgroup="ensemble", name=label,
            line={"color": color, "width": 0.5},
        ))
    weights_chart.update_layout(
        template="plotly_dark", height=520, hovermode="x unified",
        xaxis_title=None, yaxis={"title": "Вес, %", "range": [0, 100]},
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        legend={"orientation": "h", "y": 1.08},
    )
    st.plotly_chart(weights_chart, use_container_width=True)

with about_tab:
    st.subheader("Что показывает приложение")
    st.markdown(
        """
        Дашборд визуализирует уже рассчитанный holdout-бэктест выпускной
        квалификационной работы. Он помогает сравнить Attention Ensemble
        с CatBoost и Buy & Hold, просмотреть торговые сигналы и увидеть,
        как механизм внимания меняет вклад базовых моделей во времени.

        **Важно:** это исследовательский прототип, а не торговый терминал
        и не инвестиционная рекомендация. Результаты исторического
        тестирования не гарантируют будущую доходность.
        """
    )

st.caption(
    f"Источник: {DATA_PATH.name} · "
    f"{full.index.min():%d.%m.%Y}–{full.index.max():%d.%m.%Y}"
)
