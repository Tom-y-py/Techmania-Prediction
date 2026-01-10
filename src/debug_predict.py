"""Debug script for prediction"""
from predict import load_models, predict_single_date
import traceback

models = load_models()
try:
    result = predict_single_date('2026-01-11', models)
    print("SUCCESS:", result)
except Exception as e:
    print("ERROR:")
    traceback.print_exc()
