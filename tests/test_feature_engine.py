"""Offline safety tests: run `python -m unittest discover -s tests -v`."""
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feature_engine import compute_roll_spread, compute_corwin_schultz_spread, engineer_features, FEATURE_COLUMNS


def synthetic_bars(n=90):
    dates = pd.date_range('2026-01-01', periods=n, freq='B')
    rng = np.random.default_rng(7)
    prices = 120.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, n)))
    high = prices * 1.015
    low = prices * 0.985
    return pd.DataFrame({'Date': dates, 'Open': prices, 'High': high,
                         'Low': low, 'Close': prices, 'Volume': 250_000})


class FeatureEngineeringTests(unittest.TestCase):
    def test_positive_covariance_no_warning(self):
        # Increasing returns -> positive serial covariance, which used to pass
        # negative numbers to sqrt() through eager np.where evaluation.
        ret = np.linspace(0.001, 0.03, 80)
        close = pd.Series(100.0 * np.cumprod(1.0 + ret))
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            spread = compute_roll_spread(close)
        self.assertTrue(np.isfinite(spread).all())
        self.assertTrue((spread >= 0).all())
        self.assertTrue((spread <= 0.25).all())

    def test_negative_covariance_calculates_positive_spread(self):
        ret = np.resize(np.array([0.02, -0.02]), 90)
        close = pd.Series(100.0 * np.cumprod(1 + ret))
        spread = compute_roll_spread(close)
        self.assertGreater(spread.iloc[-1], 0)

    def test_corwin_spread_no_nan_sqrt_or_inf(self):
        bars = synthetic_bars()
        out = compute_corwin_schultz_spread(bars)
        self.assertEqual(len(out), len(bars))
        self.assertTrue(out.iloc[1:].notna().all())
        self.assertTrue(out.iloc[1:].between(0, 0.4).all())

    def test_invalid_hl_is_not_invented(self):
        bars = synthetic_bars()
        bars.loc[8, 'Low'] = 0
        result = compute_corwin_schultz_spread(bars)
        self.assertTrue(np.isnan(result.iloc[8]))
        self.assertTrue(np.isnan(result.iloc[9]))

    def test_latest_bar_retained_for_live_prediction(self):
        bars = synthetic_bars()
        out = engineer_features(bars)
        self.assertFalse(out.empty)
        self.assertEqual(out.iloc[-1]['Date'], bars.iloc[-1]['Date'])
        self.assertTrue(all(x in out.columns for x in FEATURE_COLUMNS))
        self.assertTrue(out.iloc[-1][list(FEATURE_COLUMNS)].notna().all())

    def test_unknown_forward_labels_are_nan(self):
        out = engineer_features(synthetic_bars())
        self.assertTrue(out['target_breakout_5d'].iloc[-5:].isna().all())
        self.assertTrue(out['target_breakout_5d'].iloc[:-5].notna().all())

    def test_normalized_pence_turnover(self):
        out = engineer_features(synthetic_bars())
        row = out.iloc[-1]
        self.assertAlmostEqual(row['turnover_gbp'], row['Close'] * row['Volume'] / 100, places=6)

    def test_bad_bar_rejected(self):
        bars = synthetic_bars()
        bars.loc[60, 'High'] = 0
        out = engineer_features(bars)
        self.assertTrue((out['High'] > 0).all())
        self.assertNotIn(bars.loc[60, 'Date'], set(out['Date']))

    def test_all_constant_prices_no_warnings(self):
        bars = synthetic_bars()
        bars.loc[:, ['Open','Close']] = 100.0
        bars.loc[:, 'High'] = 101.0
        bars.loc[:, 'Low'] = 99.0
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            out = engineer_features(bars)
        self.assertTrue(out.iloc[-1][list(FEATURE_COLUMNS)].notna().all())

    def test_short_input_empty_not_crash(self):
        out = engineer_features(synthetic_bars(18))
        self.assertTrue(out.empty)


if __name__ == '__main__':
    unittest.main()
