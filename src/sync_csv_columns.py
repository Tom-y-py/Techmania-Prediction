"""
Porovná sloupce mezi techmania_with_weather_and_holidays.csv a techmania_2026_template.csv
a synchronizuje je.
"""

import pandas as pd
import sys
from pathlib import Path

# Přidat app do path
sys.path.append(str(Path(__file__).parent.parent / 'app'))

print("=" * 80)
print("ANALÝZA A SYNCHRONIZACE SLOUPCŮ")
print("=" * 80)

# Načíst oba CSV
historical_path = Path(__file__).parent.parent / 'data' / 'processed' / 'techmania_with_weather_and_holidays.csv'
template_path = Path(__file__).parent.parent / 'data' / 'raw' / 'techmania_2026_template.csv'

print(f"\n📂 Načítám historická data...")
df_historical = pd.read_csv(historical_path)
print(f"   Řádků: {len(df_historical)}")
print(f"   Sloupců: {len(df_historical.columns)}")

print(f"\n📂 Načítám 2026 template...")
df_template = pd.read_csv(template_path)
print(f"   Řádků: {len(df_template)}")
print(f"   Sloupců: {len(df_template.columns)}")

# Porovnat sloupce
historical_cols = set(df_historical.columns)
template_cols = set(df_template.columns)

print("\n" + "=" * 80)
print("POROVNÁNÍ SLOUPCŮ")
print("=" * 80)

# Sloupce pouze v historical
only_in_historical = historical_cols - template_cols
if only_in_historical:
    print(f"\n❌ Sloupce POUZE v historical ({len(only_in_historical)}):")
    for col in sorted(only_in_historical):
        print(f"   - {col}")
else:
    print(f"\n✅ Všechny sloupce z historical jsou v template")

# Sloupce pouze v template
only_in_template = template_cols - historical_cols
if only_in_template:
    print(f"\n⚠️ Sloupce POUZE v template ({len(only_in_template)}):")
    for col in sorted(only_in_template):
        print(f"   - {col}")
else:
    print(f"\n✅ Template nemá žádné extra sloupce")

# Společné sloupce
common_cols = historical_cols & template_cols
print(f"\n✅ Společné sloupce: {len(common_cols)}")

print("\n" + "=" * 80)
print("SEZNAM VŠECH SLOUPCŮ V HISTORICAL")
print("=" * 80)
for i, col in enumerate(sorted(df_historical.columns), 1):
    print(f"{i:3}. {col}")

# Synchronizace
if only_in_historical:
    print("\n" + "=" * 80)
    print("SYNCHRONIZACE")
    print("=" * 80)
    print(f"\nPřidávám chybějící sloupce do template...")
    
    # Pro každý chybějící sloupec, přidat ho s NaN hodnotami
    for col in sorted(only_in_historical):
        # Zkontrolovat typ dat v historical
        dtype = df_historical[col].dtype
        
        if dtype == 'object':
            # Textové sloupce - použít None nebo prázdný řetězec
            if col in ['school_break_type']:
                df_template[col] = None
            else:
                df_template[col] = None
        elif dtype in ['int64', 'int32', 'uint8']:
            # Celá čísla - použít 0 nebo NaN podle kontextu
            if col.startswith('is_'):
                df_template[col] = 0  # Boolean flags
            else:
                df_template[col] = pd.NA
        else:
            # Float - použít NaN
            df_template[col] = pd.NA
        
        print(f"   ✓ Přidán: {col} (dtype: {dtype})")
    
    # Seřadit sloupce ve stejném pořadí jako historical
    df_template = df_template[df_historical.columns]
    
    # Uložit
    df_template.to_csv(template_path, index=False)
    
    print(f"\n✅ Template aktualizován!")
    print(f"   Nový počet sloupců: {len(df_template.columns)}")
    
    # Zobrazit ukázku
    print(f"\n📋 Ukázka prvních 3 řádků (vybrané sloupce):")
    sample_cols = ['date', 'total_visitors'] + list(sorted(only_in_historical))[:5]
    print(df_template[sample_cols].head(3).to_string(index=False))
else:
    print("\n✅ Template je již synchronizován s historical!")

print("\n✅ Hotovo!")
