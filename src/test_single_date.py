"""
Testovací script pro predikci na konkrétní datum 
"""

import sys
from pathlib import Path
from datetime import datetime

# Přidat src do path
sys.path.append(str(Path(__file__).parent))

from predict import load_models, predict_single_date

def main():
    print("\n" + "=" * 70)
    print("🧪 TEST PREDIKCE - 2.1.2026 (PÁTEK)")
    print("=" * 70)
    
    # Datum pro test - musí být starší než 5 dní pro archivní API
    test_date = "2026-01-11"
    
    print(f"\n📅 Testované datum: {test_date}")
    print(f"   Den: Pátek (všední den)")
    print(f"   Očekávání: CatBoost by měl být VYPNUTÝ")
    print(f"   Zdroj počasí: Archive API (historická data)")
    
    # Načíst modely
    print("\n📦 Načítání modelů...")
    models = load_models()
    
    if models is None:
        print("\n❌ CHYBA: Modely se nepodařilo načíst!")
        print("\n⚠️ Nejprve je potřeba přetrénovat modely:")
        print("   cd src && python ensemble_model_v3.py")
        return
    
    # Zkontrolovat, zda existuje historical_mae
    if 'historical_mae' not in models:
        print("\n⚠️ VAROVÁNÍ: historical_mae_v3.pkl neexistuje!")
        print("   Confidence intervaly budou počítány starou metodou (variance modelů)")
        print("\n   Pro správné CI je potřeba přetrénovat modely:")
        print("   cd src && python ensemble_model_v3.py")
        print("\n   Pokračuji s testem...\n")
    else:
        print(f"\n✅ Historical MAE načtena:")
        print(f"   Weekday MAE: {models['historical_mae']['weekday']:.2f}")
        print(f"   Weekend MAE: {models['historical_mae']['weekend']:.2f}")
    
    # Provést predikci
    print("\n" + "=" * 70)
    print("🔮 SPOUŠTÍM PREDIKCI")
    print("=" * 70)
    
    try:
        result = predict_single_date(test_date, models)
        
        # Zobrazit výsledky
        print("\n" + "=" * 70)
        print(f"🎯 VÝSLEDKY PRO {result['date'].strftime('%d.%m.%Y')} ({result['day_of_week']})")
        print("=" * 70)
        
        print(f"\n📊 ENSEMBLE PREDIKCE: {result['ensemble_prediction']} návštěvníků")
        print(f"   95% Confidence Interval: [{result['confidence_interval'][0]} - {result['confidence_interval'][1]}]")
        
        # Info o CI
        ci_width = result['confidence_interval'][1] - result['confidence_interval'][0]
        print(f"   CI šířka: {ci_width} (užší = přesnější)")
        
        # Status CatBoost
        catboost_used = result.get('catboost_used', True)
        catboost_status = "✅ ACTIVE" if catboost_used else "❌ DISABLED (weekday)"
        
        print(f"\n🤖 JEDNOTLIVÉ MODELY:")
        print(f"   LightGBM: {result['individual_predictions']['lightgbm']} návštěvníků")
        print(f"     Váha: {result['model_weights']['lightgbm']:.1%}")
        
        print(f"   XGBoost: {result['individual_predictions']['xgboost']} návštěvníků")
        print(f"     Váha: {result['model_weights']['xgboost']:.1%}")
        
        print(f"   CatBoost: {result['individual_predictions']['catboost']} návštěvníků")
        print(f"     Váha: {result['model_weights']['catboost']:.1%}")
        print(f"     Status: {catboost_status}")
        
        # Počasí
        print(f"\n🌤️ POČASÍ:")
        print(f"   Popis: {result['weather']['description']}")
        print(f"   Teplota: {result['weather']['temperature']:.1f}°C")
        print(f"   Srážky: {result['weather']['precipitation']:.1f}mm")
        if result['weather'].get('snowfall', 0) > 0:
            print(f"   Sníh: {result['weather']['snowfall']:.1f}mm")
        
        # Ověření
        print("\n" + "=" * 70)
        print("✅ OVĚŘENÍ IMPLEMENTACE")
        print("=" * 70)
        
        # Test 1: CatBoost 
        if not catboost_used:
            print("✅ Test 1: CatBoost je správně VYPNUTÝ pro všední den (pátek)")
            if result['model_weights']['catboost'] == 0.0:
                print("   ✅ Váha CatBoost je 0.0")
            else:
                print(f"   ⚠️ Váha CatBoost by měla být 0.0, ale je {result['model_weights']['catboost']:.1%}")
        else:
            print("❌ Test 1: CHYBA - CatBoost by měl být VYPNUTÝ pro pátek (není víkend/svátek)!")
        
        # Test 2: Váhy LightGBM + XGBoost by měly dát dohromady 1.0
        weight_sum = result['model_weights']['lightgbm'] + result['model_weights']['xgboost']
        if abs(weight_sum - 1.0) < 0.01:
            print(f"✅ Test 2: Součet vah LightGBM + XGBoost = {weight_sum:.3f} (OK)")
        else:
            print(f"⚠️ Test 2: Součet vah LightGBM + XGBoost = {weight_sum:.3f} (mělo by být ~1.0)")
        
        # Test 3: CI dolní mez by měla být >= 50
        if result['confidence_interval'][0] >= 50:
            print(f"✅ Test 3: Dolní mez CI = {result['confidence_interval'][0]} (>= 50)")
        else:
            print(f"⚠️ Test 3: Dolní mez CI = {result['confidence_interval'][0]} (měla by být >= 50)")
        
        # Test 4: Predikce by měla být mezi jednotlivými modely
        min_pred = min(result['individual_predictions']['lightgbm'], 
                       result['individual_predictions']['xgboost'])
        max_pred = max(result['individual_predictions']['lightgbm'], 
                       result['individual_predictions']['xgboost'])
        
        if min_pred <= result['ensemble_prediction'] <= max_pred:
            print(f"✅ Test 4: Ensemble ({result['ensemble_prediction']}) je mezi LightGBM ({min_pred}) a XGBoost ({max_pred})")
        else:
            print(f"⚠️ Test 4: Ensemble ({result['ensemble_prediction']}) není mezi LightGBM ({min_pred}) a XGBoost ({max_pred})")
        
        print("\n" + "=" * 70)
        print("✅ TEST DOKONČEN!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ CHYBA PŘI PREDIKCI: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
