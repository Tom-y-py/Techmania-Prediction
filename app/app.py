"""
Flask API pro predikci návštěvnosti Techmanie.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
from datetime import datetime

# Přidat src do path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from feature_engineering import create_all_features

app = Flask(__name__)
CORS(app)  # Povolit CORS pro komunikaci s React frontendem

# Globální proměnné pro model
model = None
feature_columns = None

def load_model():
    """Načte natrénovaný model."""
    global model, feature_columns
    try:
        model_path = Path(__file__).parent.parent / 'models' / 'best_model.pkl'
        model = joblib.load(model_path)
        # TODO: Načíst seznam features
        print("Model úspěšně načten")
    except Exception as e:
        print(f"Chyba při načítání modelu: {e}")

@app.route('/')
def index():
    """Root endpoint - informace o API."""
    return jsonify({
        'name': 'Techmania Prediction API',
        'version': '1.0.0',
        'message': 'API pro predikci návštěvnosti Techmanie',
        'endpoints': {
            '/': 'Tento endpoint',
            '/api': 'Dokumentace API',
            '/predict': 'POST - Predikce pro konkrétní datum',
            '/predict/range': 'POST - Predikce pro období',
            '/health': 'GET - Health check'
        }
    })

@app.route('/api')
def api_docs():
    """Dokumentace API."""
    return jsonify({
        'name': 'Techmania Prediction API',
        'version': '1.0.0',
        'endpoints': {
            '/': 'Webové rozhraní',
            '/api': 'Tato dokumentace',
            '/predict': 'POST - Predikce pro konkrétní datum',
            '/predict/range': 'POST - Predikce pro období',
            '/health': 'GET - Health check'
        }
    })

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predikce návštěvnosti pro konkrétní datum.
    
    Očekává JSON:
    {
        "date": "2025-01-15",
        "is_holiday": false,
        "opening_hours": "9-17"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'date' not in data:
            return jsonify({'error': 'Chybí datum'}), 400
        
        # Parsování data
        date = pd.to_datetime(data['date'])
        
        # Vytvoření DataFrame pro predikci
        df_pred = pd.DataFrame({
            'date': [date],
            'is_holiday': [data.get('is_holiday', False)],
            'opening_hours': [data.get('opening_hours', '9-17')]
        })
        
        # TODO: Vytvoření features a predikce
        # df_pred = create_all_features(df_pred)
        # prediction = model.predict(df_pred[feature_columns])
        
        prediction = 250  # Placeholder
        
        return jsonify({
            'date': date.strftime('%Y-%m-%d'),
            'predicted_visitors': int(prediction),
            'confidence_interval': {
                'lower': int(prediction * 0.85),
                'upper': int(prediction * 1.15)
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/range', methods=['POST'])
def predict_range():
    """
    Predikce návštěvnosti pro časové období.
    
    Očekává JSON:
    {
        "start_date": "2025-01-01",
        "end_date": "2025-01-31"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'start_date' not in data or 'end_date' not in data:
            return jsonify({'error': 'Chybí start_date nebo end_date'}), 400
        
        start_date = pd.to_datetime(data['start_date'])
        end_date = pd.to_datetime(data['end_date'])
        
        # Vytvoření date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # TODO: Vytvoření features a predikce pro celé období
        
        predictions = []
        for date in date_range:
            predictions.append({
                'date': date.strftime('%Y-%m-%d'),
                'predicted_visitors': 250  # Placeholder
            })
        
        return jsonify({
            'predictions': predictions,
            'total_predicted': len(predictions) * 250
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
