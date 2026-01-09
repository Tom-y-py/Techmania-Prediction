"""
Test predikce na konkrétní data
"""

from predict import load_models, predict_single_date, print_prediction

def test_specific_dates():
    """
    Testuje predikci na konkrétních datech která NEJSOU v trénovacích datech
    """
    print("\n" + "=" * 70)
    print("🧪 TESTING PREDICTIONS ON UNSEEN DATA")
    print("=" * 70)
    
    # Načíst modely
    print("\n📦 Loading trained models...")
    models = load_models()
    
    if models is None:
        print("❌ Models not found. Please train first: python ensemble_model.py")
        return
    
    # Test data - data která nejsou v trénovacích datech (po 2025-12-31)
    # Generujeme dynamicky od aktuálního data
    from datetime import date, timedelta
    today = date.today()
    
    test_dates = [
        (today + timedelta(days=1)).strftime('%Y-%m-%d'),  # Následující den
        (today + timedelta(days=6)).strftime('%Y-%m-%d'),  # Za týden
        (today + timedelta(days=9)).strftime('%Y-%m-%d'),  # Nejbližší víkend (sobota)
        '2026-02-14',  # Valentýn
        '2026-07-15',  # Letní prázdniny
        '2026-12-24',  # Štědrý den
    ]
    
    results = []
    
    for date_str in test_dates:
        print("\n" + "=" * 70)
        try:
            result = predict_single_date(date_str, models)
            print_prediction(result)
            
            results.append({
                'date': date_str,
                'prediction': result['ensemble_prediction'],
                'day': result['day_of_week']
            })
        except Exception as e:
            print(f"❌ Error predicting {date_str}: {e}")
    
    # Shrnutí
    print("\n" + "=" * 70)
    print("📊 SUMMARY OF PREDICTIONS")
    print("=" * 70)
    print(f"\n{'Date':<15} {'Day':<12} {'Predicted Visitors':>20}")
    print("-" * 70)
    for r in results:
        print(f"{r['date']:<15} {r['day']:<12} {r['prediction']:>20}")
    
    print("\n" + "=" * 70)
    print("✅ TESTING COMPLETE!")
    print("=" * 70)


def test_single_custom_date():
    """
    Test na jedno vlastní datum
    """
    print("\n" + "=" * 70)
    print("🎯 CUSTOM DATE PREDICTION")
    print("=" * 70)
    
    # Načíst modely
    models = load_models()
    if models is None:
        return
    
    # Vlastní datum
    from datetime import date, timedelta
    next_day = date.today() + timedelta(days=1)
    default_date = next_day.strftime('%Y-%m-%d')
    
    custom_date = input(f"\n📅 Zadej datum (YYYY-MM-DD) nebo Enter pro následující den ({default_date}): ").strip()
    
    if not custom_date:
        custom_date = default_date
        print(f"   Použito: {custom_date}")
    
    try:
        result = predict_single_date(custom_date, models)
        print_prediction(result)
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--custom':
        # Mód pro vlastní datum
        test_single_custom_date()
    else:
        # Testuj na předdefinovaných datech
        test_specific_dates()
