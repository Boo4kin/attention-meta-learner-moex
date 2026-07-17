"""Smoke tests for the public thesis repository."""

import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BACKTEST = ROOT / "data" / "backtest_results.csv"
RAW_DATA = ROOT / "data" / "Si_1min_2025-05-02_2026-04-30.csv"


class RepositoryDataTests(unittest.TestCase):
    def test_expected_files_are_present(self) -> None:
        self.assertTrue(BACKTEST.is_file())
        self.assertTrue(RAW_DATA.is_file())

    def test_backtest_schema_and_dates(self) -> None:
        frame = pd.read_csv(BACKTEST, parse_dates=["Datetime"])
        required = {
            "Datetime", "Close", "Ensemble_Signal", "CB_Signal",
            "Equity_Ensemble", "Equity_CB", "Equity_BH",
            "Weight_CB", "Weight_KAN", "Weight_Mamba",
            "Ensemble_PnL", "CB_PnL",
        }
        self.assertTrue(required.issubset(frame.columns))
        self.assertEqual(len(frame), 7550)
        self.assertTrue(frame["Datetime"].is_monotonic_increasing)
        self.assertFalse(frame[list(required)].isnull().any().any())

    def test_weights_sum_to_one(self) -> None:
        frame = pd.read_csv(BACKTEST)
        total = frame[["Weight_CB", "Weight_KAN", "Weight_Mamba"]].sum(axis=1)
        self.assertTrue(((total - 1.0).abs() < 1e-5).all())


if __name__ == "__main__":
    unittest.main()
