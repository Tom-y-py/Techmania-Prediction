import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
APP_DIR = PROJECT_ROOT / "app"

for path in (SRC_DIR, APP_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


import predict as predict_module  # noqa: E402
import app as app_module  # noqa: E402


def test_feature_contract_fails_on_missing_columns():
    df_pred = pd.DataFrame([{"a": 1.0}])
    with pytest.raises(ValueError, match="Chybí požadované featury"):
        predict_module._build_feature_matrix(df_pred, ["a", "b"])


def test_feature_contract_fails_on_nan_values():
    df_pred = pd.DataFrame([{"a": 1.0, "b": None}])
    with pytest.raises(ValueError, match="NaN"):
        predict_module._build_feature_matrix(df_pred, ["a", "b"])


def test_normalize_historical_mae_accepts_segmented_dict():
    normalized = app_module.normalize_historical_mae({"weekday": 10.0, "weekend": 12.0})
    assert normalized["weekday"] == 10.0
    assert normalized["weekend"] == 12.0
    assert "overall" in normalized


def test_normalize_historical_mae_rejects_invalid_input():
    with pytest.raises(ValueError):
        app_module.normalize_historical_mae({"unexpected": 5})
