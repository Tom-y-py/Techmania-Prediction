"""
Test predikce na konkrétní data
"""

from predict import load_models, predict_single_date, print_prediction

def test_specific_dates():
    """
    Testuje predikci na následujících 7 dnech (celý týden)
    """
    print("\n" + "=" * 70)
    print("🧪 TESTING PREDICTIONS - NEXT 7 DAYS")
    print("=" * 70)
    
    # Načíst modely
    print("\n📦 Loading trained models...")
    models = load_models()
    
    if models is None:
        print("❌ Models not found. Please train first: python ensemble_model.py")
        return
    
    # Generovat 7 po sobě jdoucích dnů od zítřka
    from datetime import date, timedelta
    today = date.today()
    
    test_dates = []
    for i in range(1, 8):  # 7 dní od zítřka
        next_date = today + timedelta(days=i)
        test_dates.append(next_date.strftime('%Y-%m-%d'))
    
    print(f"\n📅 Predikce od {test_dates[0]} do {test_dates[-1]}")
    print("=" * 70)
    
    results = []
    
    for date_str in test_dates:
        print("\n" + "-" * 70)
        try:
            result = predict_single_date(date_str, models)
            print_prediction(result)
            
            results.append({
                'date': date_str,
                'prediction': result['ensemble_prediction'],
                'day': result['day_of_week'],
                'weather_desc': result['weather']['description'],
                'temp': result['weather']['temperature'],
                'precip': result['weather']['precipitation'],
                'rain': result['weather']['rain'],
                'snow': result['weather']['snowfall']
            })
        except Exception as e:
            print(f"❌ Error predicting {date_str}: {e}")
    
    # Shrnutí
    print("\n" + "=" * 110)
    print("📊 SUMMARY - WEEKLY PREDICTIONS WITH WEATHER")
    print("=" * 110)
    print(f"\n{'Date':<12} {'Day':<10} {'Visitors':>8}  {'Weather':<35} {'Temp':>6}  {'Srážky':>7}")
    print("-" * 110)
    
    total = 0
    for r in results:
        # Zkrátit popis počasí pokud je moc dlouhý
        weather_short = r['weather_desc'][:33] + '..' if len(r['weather_desc']) > 35 else r['weather_desc']
        
        # Ikony pro srážky
        precip_str = ""
        if r['snow'] > 0:
            precip_str = f"❄️ {r['snow']:.1f}mm"
        elif r['rain'] > 0:
            precip_str = f"🌧️ {r['rain']:.1f}mm"
        elif r['precip'] > 0:
            precip_str = f"💧 {r['precip']:.1f}mm"
        else:
            precip_str = "☀️ 0mm"
        
        print(f"{r['date']:<12} {r['day']:<10} {r['prediction']:>8}  {weather_short:<35} {r['temp']:>5.1f}°C  {precip_str:>7}")
        total += r['prediction']
    
    print("-" * 110)
    print(f"{'TOTAL (7 days)':<22} {total:>8}")
    print(f"{'AVERAGE/day':<22} {total/len(results):>8.0f}")
    
    print("\n" + "=" * 110)
    print("✅ TESTING COMPLETE!")
    print("=" * 110)


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
