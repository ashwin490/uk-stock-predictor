# ALPHA-LSE: feature-engineering and dependency hotfix

## What changed

- `feature_engine.py`: fixes the confirmed `sqrt(-positive_covariance)` warning from eagerly evaluating both branches of `np.where`; validates prices; computes the Corwin–Schultz spread without overflowing; and preserves *all latest feature rows* for real-time inference.
- `target_breakout_5d`: the last 5 outcomes are now `NaN` (unknown), **not false losses**. Live `app.py` uses named predictor columns and can still score the latest row. If you have a separate supervised model-training script, filter missing targets: `train = features.dropna(subset=['target_breakout_5d'])` before model fitting.
- `requirements.txt`: pins `scikit-learn==1.7.2` to match the `LabelEncoder` version reported in your logs. Pins `yfinance==1.8.0`, which includes updates addressing the NumPy timedelta compatibility warnings. Adds `pyarrow` for offline Parquet processing.

## Installation

1. **Back up your current repository and do NOT delete your trained model or historical database.**
2. Put `feature_engine.py` and `requirements.txt` beside the existing `app.py` in your project (replace those two files only).
3. In VS Code terminal, run: `python -m pip install -r requirements.txt`. For an isolated virtual environment, use the environment's Python.
4. In Streamlit Cloud, commit/push **both** files to GitHub; Cloud will reinstall the revised requirements as part of the redeploy.
5. Reboot the app; check whether predictions appear and the Python logs no longer contain the recurring invalid-sqrt warnings.
6. Verify one sample ticker's Yahoo quote currency and unit conversions before trusting its simulated capital or return calculations. Yahoo data access can change.

These are *offline-tested fixes*, **not a guarantee** that all live integrations are compatible. In particular, the user's trained model file was not supplied, so its full compatibility is not tested; `LightGBM` and `CatBoost` model library versions may also matter.

## Tests

`python -m unittest discover -s tests -v`

The tests use synthetic data and do not contact market-data services.

## Notes

- The feature calculations assume that LSE OHLC quotes have been verified and normalized to pence (GBp) by the latest app.py. Do not pass raw GBP quotes directly.
- A normal upgrade of Python dependencies cannot guarantee a 10-year maintenance-free system. Keep backups and periodic health checks.
- If `scikit-learn==1.7.2` fails to load the *complete* model, restore the previous working environment and inspect the model's library versions. Never overwrite the only copy.
- If you run on Streamlit Community Cloud, Python's version is a deployment setting; editing `requirements.txt` does not change Python itself.
