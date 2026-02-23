import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


import app as app_module  # noqa: E402


def test_events_endpoint_uses_db_helper_name_without_recursion(monkeypatch):
    fake_event = SimpleNamespace(
        id=1,
        event_date=SimpleNamespace(isoformat=lambda: "2026-01-10"),
        title="Test Event",
        description="Desc",
        venue="Techmania",
        category="education",
        expected_attendance=123,
        source="test",
        source_url="http://example.test",
        impact_level=2,
        is_active=True,
        created_at=SimpleNamespace(isoformat=lambda: "2026-01-01T12:00:00"),
    )

    monkeypatch.setattr(app_module, "DATABASE_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "db_get_events_for_date", lambda db, date: [fake_event])
    app_module.app.dependency_overrides[app_module.get_db] = lambda: iter([object()])

    client = TestClient(app_module.app)
    response = client.get("/events/2026-01-10")

    app_module.app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert data["events"][0]["title"] == "Test Event"


def test_predict_endpoint_maps_real_metadata(monkeypatch):
    fake_predict_module = ModuleType("predict")

    def fake_predict_single_date(pred_date, models_dict, historical_df=None):
        return {
            "ensemble_prediction": 321,
            "confidence_interval": (300, 350),
            "ensemble_type": "weighted",
            "individual_predictions": {"lightgbm": 320, "xgboost": 322, "catboost": 321},
            "model_weights": {"lightgbm": 0.5, "xgboost": 0.5, "catboost": 0.0},
            "weather": {
                "description": "sunny",
                "temperature": 22.0,
                "precipitation": 0.0,
                "rain": 0.0,
                "snowfall": 0.0,
                "wind_speed_max": 4.2,
                "is_nice_weather": True,
            },
        }

    fake_predict_module.predict_single_date = fake_predict_single_date
    sys.modules["predict"] = fake_predict_module

    monkeypatch.setattr(app_module, "models", {"lightgbm": object(), "xgboost": object(), "catboost": object()}, raising=False)
    monkeypatch.setattr(app_module, "ensemble_weights", {"LightGBM": 0.5, "XGBoost": 0.5, "CatBoost": 0.0}, raising=False)
    monkeypatch.setattr(app_module, "feature_columns", ["is_weekend", "is_holiday"], raising=False)
    monkeypatch.setattr(app_module, "ensemble_info", {"type": "weighted", "historical_mae": {"weekday": 1.0, "weekend": 1.2}}, raising=False)
    monkeypatch.setattr(app_module, "DATABASE_ENABLED", False, raising=False)
    monkeypatch.setattr(app_module.holiday_service, "get_holiday_info", lambda d: {"is_holiday": True, "holiday_name": "Svátek"})

    client = TestClient(app_module.app)
    response = client.post("/predict", json={"date": "2026-01-10"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["holiday_info"]["is_holiday"] is True
    assert payload["weather_info"]["is_nice_weather"] is True
