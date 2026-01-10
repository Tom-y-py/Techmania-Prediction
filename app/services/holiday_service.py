"""
Služba pro automatickou detekci českých svátků.
Vrací informace o tom, zda je daný den svátek a jaký.
"""

from datetime import date, datetime, timedelta
from typing import Dict, Optional, Tuple
import pandas as pd


class HolidayService:
    """Služba pro detekci českých státních svátků a významných dnů."""
    
    # Státní svátky s pevným datem
    FIXED_HOLIDAYS = {
        (1, 1): "Nový rok / Den obnovy samostatného českého státu",
        (5, 1): "Svátek práce",
        (5, 8): "Den vítězství",
        (7, 5): "Den slovanských věrozvěstů Cyrila a Metoděje",
        (7, 6): "Den upálení mistra Jana Husa",
        (9, 28): "Den české státnosti",
        (10, 28): "Den vzniku samostatného československého státu",
        (11, 17): "Den boje za svobodu a demokracii",
        (12, 24): "Štědrý den",
        (12, 25): "1. svátek vánoční",
        (12, 26): "2. svátek vánoční",
    }
    
    # Významné dny (ne státní svátky, ale ovlivňují návštěvnost)
    SIGNIFICANT_DAYS = {
        (1, 6): "Tři králové",
        (2, 14): "Valentýn",
        (12, 31): "Silvester",
    }
    
    def __init__(self):
        """Inicializace služby."""
        pass
    
    def _calculate_easter(self, year: int) -> date:
        """
        Výpočet data Velikonoc pomocí Butcherova algoritmu.
        
        Args:
            year: Rok
            
        Returns:
            Datum velikonoční neděle
        """
        a = year % 19
        b = year // 100
        c = year % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1
        return date(year, month, day)
    
    def _get_movable_holidays(self, year: int) -> Dict[date, str]:
        """
        Vrací pohyblivé svátky pro daný rok (závislé na Velikonocích).
        
        Args:
            year: Rok
            
        Returns:
            Slovník {datum: název svátku}
        """
        easter = self._calculate_easter(year)
        
        return {
            # Velikonoční pondělí (den po Velikonoční neděli)
            easter + pd.Timedelta(days=1): "Velikonoční pondělí",
        }
    
    def is_holiday(self, check_date: date) -> Tuple[bool, Optional[str]]:
        """
        Zjistí, zda je daný den státní svátek.
        
        Args:
            check_date: Datum ke kontrole
            
        Returns:
            Tuple (je_svátek, název_svátku)
        """
        # Kontrola pevných svátků
        key = (check_date.month, check_date.day)
        if key in self.FIXED_HOLIDAYS:
            return True, self.FIXED_HOLIDAYS[key]
        
        # Kontrola pohyblivých svátků
        movable = self._get_movable_holidays(check_date.year)
        if check_date in movable:
            return True, movable[check_date]
        
        return False, None
    
    def is_significant_day(self, check_date: date) -> Tuple[bool, Optional[str]]:
        """
        Zjistí, zda je daný den významný (ne státní svátek, ale může ovlivnit návštěvnost).
        
        Args:
            check_date: Datum ke kontrole
            
        Returns:
            Tuple (je_významný, název)
        """
        key = (check_date.month, check_date.day)
        if key in self.SIGNIFICANT_DAYS:
            return True, self.SIGNIFICANT_DAYS[key]
        return False, None
    
    def _get_school_breaks_for_year(self, year: int) -> Dict[str, tuple]:
        """
        Vrací definice školních prázdnin pro daný rok.
        
        Args:
            year: Školní rok (např. 2026 = školní rok 2026/2027)
            
        Returns:
            Dict s definicemi prázdnin: {'break_type': (start_date, end_date)}
        """
        # Plzeň je v regionu: "8. 3. - 14. 3." pro jarní prázdniny
        SCHOOL_BREAKS = {
            2026: {
                'autumn': (date(2026, 10, 29), date(2026, 10, 30)),
                'winter': (date(2026, 12, 23), date(2027, 1, 3)),
                'halfyear': (date(2027, 1, 29), date(2027, 1, 29)),
                'spring': (date(2027, 3, 8), date(2027, 3, 14)),
                'easter': (date(2027, 3, 25), date(2027, 3, 26)),  # Čtvrtek + Pátek
                'summer': (date(2027, 7, 1), date(2027, 8, 31)),
            },
            2025: {
                'autumn': (date(2025, 10, 29), date(2025, 10, 30)),
                'winter': (date(2025, 12, 23), date(2026, 1, 2)),
                'halfyear': (date(2026, 1, 30), date(2026, 1, 30)),
                'spring': (date(2026, 2, 23), date(2026, 3, 1)),
                'easter': (date(2026, 4, 2), date(2026, 4, 3)),
                'summer': (date(2026, 6, 27), date(2026, 8, 31)),
            },
            2024: {
                'autumn': (date(2024, 10, 29), date(2024, 10, 30)),
                'winter': (date(2024, 12, 23), date(2025, 1, 3)),
                'halfyear': (date(2025, 1, 31), date(2025, 1, 31)),
                'spring': (date(2025, 2, 24), date(2025, 3, 2)),
                'easter': (date(2025, 4, 17), date(2025, 4, 18)),
                'summer': (date(2025, 6, 28), date(2025, 8, 31)),
            },
            2023: {
                'autumn': (date(2023, 10, 26), date(2023, 10, 27)),
                'winter': (date(2023, 12, 23), date(2024, 1, 2)),
                'halfyear': (date(2024, 2, 2), date(2024, 2, 2)),
                'spring': (date(2024, 2, 19), date(2024, 2, 25)),
                'easter': (date(2024, 3, 28), date(2024, 3, 29)),
                'summer': (date(2024, 6, 29), date(2024, 9, 1)),
            },
            2022: {
                'autumn': (date(2022, 10, 26), date(2022, 10, 27)),
                'winter': (date(2022, 12, 23), date(2023, 1, 2)),
                'halfyear': (date(2023, 2, 3), date(2023, 2, 3)),
                'spring': (date(2023, 2, 13), date(2023, 2, 19)),
                'easter': (date(2023, 4, 6), date(2023, 4, 7)),
                'summer': (date(2023, 7, 1), date(2023, 9, 3)),
            },
            2021: {
                'autumn': (date(2021, 10, 27), date(2021, 10, 29)),
                'winter': (date(2021, 12, 23), date(2022, 1, 2)),
                'halfyear': (date(2022, 2, 4), date(2022, 2, 4)),
                'spring': (date(2022, 2, 7), date(2022, 2, 13)),
                'easter': (date(2022, 4, 14), date(2022, 4, 15)),
                'summer': (date(2022, 7, 1), date(2022, 8, 31)),
            },
            2020: {
                'autumn': (date(2020, 10, 29), date(2020, 10, 30)),
                'winter': (date(2020, 12, 23), date(2021, 1, 3)),
                'halfyear': (date(2021, 1, 29), date(2021, 1, 29)),
                'spring': (date(2021, 3, 8), date(2021, 3, 14)),
                'easter': (date(2021, 4, 1), date(2021, 4, 2)),
                'summer': (date(2021, 7, 1), date(2021, 8, 31)),
            },
        }
        
        return SCHOOL_BREAKS.get(year, {})
    
    def _is_in_break(self, check_date: date, break_start: date, break_end: date) -> bool:
        """Zkontroluje, zda je datum v daném období prázdnin."""
        return break_start <= check_date <= break_end
    
    def _days_between(self, date1: date, date2: date) -> int:
        """Vrátí počet dní mezi dvěma daty (absolutní hodnota)."""
        return abs((date2 - date1).days)
    
    def _get_season(self, check_date: date) -> str:
        """Vrací přesné roční období."""
        month = check_date.month
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "autumn"
    
    def _get_week_position(self, check_date: date) -> str:
        """Vrací pozici týdne v měsíci."""
        day = check_date.day
        if day <= 7:
            return "first_week"
        elif day <= 14:
            return "second_week"
        elif day <= 21:
            return "third_week"
        else:
            return "last_week"
    
    def _get_school_week_number(self, check_date: date) -> int:
        """
        Vrací číslo týdne ve školním roce (1. září = týden 1).
        
        Args:
            check_date: Datum
            
        Returns:
            Číslo týdne (1-40)
        """
        # Najít začátek školního roku
        if check_date.month >= 9:
            school_year_start = date(check_date.year, 9, 1)
        else:
            school_year_start = date(check_date.year - 1, 9, 1)
        
        # Vypočítat počet týdnů od začátku školního roku
        days_since_start = (check_date - school_year_start).days
        week_number = (days_since_start // 7) + 1
        
        return max(1, min(week_number, 40))
    
    def _is_bridge_day(self, check_date: date) -> bool:
        """
        Zjistí, zda je datum "most" mezi svátkem a víkendem.
        
        Args:
            check_date: Datum
            
        Returns:
            True pokud je most
        """
        # Most = pátek nebo pondělí mezi svátkem a víkendem
        day_of_week = check_date.weekday()  # 0=Monday, 6=Sunday
        
        # Pokud je pátek a následující pondělí je svátek
        if day_of_week == 4:  # Friday
            next_monday = check_date + timedelta(days=3)
            is_holiday, _ = self.is_holiday(next_monday)
            if is_holiday:
                return True
        
        # Pokud je pondělí a předchozí pátek byl svátek
        if day_of_week == 0:  # Monday
            prev_friday = check_date - timedelta(days=3)
            is_holiday, _ = self.is_holiday(prev_friday)
            if is_holiday:
                return True
        
        return False
    
    def _get_long_weekend_length(self, check_date: date) -> int:
        """
        Vrací délku prodlouženého víkendu (pokud je součástí nějakého).
        
        Args:
            check_date: Datum
            
        Returns:
            Počet dnů (2 = normální víkend, 3+ = prodloužený)
        """
        day_of_week = check_date.weekday()
        
        # Pokud není víkend ani pátek/pondělí, není součástí víkendu
        if day_of_week not in [4, 5, 6, 0]:  # Fri, Sat, Sun, Mon
            return 0
        
        # Najít začátek a konec víkendu/prodlouženého víkendu
        # Hledat zpět do pátku
        current = check_date
        while current.weekday() > 4:  # Dokud nejsme na pátku nebo dřív
            current -= timedelta(days=1)
        
        # Pokud je čtvrtek a pátek je svátek, začneme od čtvrtka
        if current.weekday() == 4:
            prev_day = current - timedelta(days=1)
            is_holiday, _ = self.is_holiday(current)
            if is_holiday or self._is_bridge_day(current):
                current = prev_day
        
        start = current
        
        # Hledat dopředu do pondělí
        current = check_date
        while current.weekday() < 1:  # Dokud nejsme po pondělí
            current += timedelta(days=1)
        
        # Pokud je úterý a pondělí je svátek
        if current.weekday() == 1:
            prev_day = current - timedelta(days=1)
            is_holiday, _ = self.is_holiday(prev_day)
            if is_holiday or self._is_bridge_day(prev_day):
                current = current
            else:
                current = prev_day
        else:
            current = current - timedelta(days=1)
        
        end = current
        
        length = (end - start).days + 1
        return length if length > 2 else 2  # Min 2 (normální víkend)

    def get_holiday_info(self, check_date: date) -> Dict:
        """
        Vrací kompletní informace o svátku pro dané datum.
        VŠECHNY FEATURES PRO ML MODEL (19 features).
        
        Args:
            check_date: Datum ke kontrole
            
        Returns:
            Slovník s informacemi o svátku a školních prázdninách
        """
        # Státní svátky
        is_hol, holiday_name = self.is_holiday(check_date)
        is_sig, sig_name = self.is_significant_day(check_date)
        
        # Školní rok (září = začátek nového roku)
        school_year = check_date.year if check_date.month >= 9 else check_date.year - 1
        breaks = self._get_school_breaks_for_year(school_year)
        
        # Kontrola jednotlivých typů prázdnin
        is_spring_break = False
        is_autumn_break = False
        is_winter_break = False
        is_easter_break = False
        is_halfyear_break = False
        is_summer_holiday = False
        school_break_type = None
        
        if 'spring' in breaks and self._is_in_break(check_date, *breaks['spring']):
            is_spring_break = True
            school_break_type = "spring"
        
        if 'autumn' in breaks and self._is_in_break(check_date, *breaks['autumn']):
            is_autumn_break = True
            school_break_type = "autumn"
        
        if 'winter' in breaks and self._is_in_break(check_date, *breaks['winter']):
            is_winter_break = True
            school_break_type = "winter"
        
        if 'easter' in breaks and self._is_in_break(check_date, *breaks['easter']):
            is_easter_break = True
            school_break_type = "easter"
        
        if 'halfyear' in breaks and self._is_in_break(check_date, *breaks['halfyear']):
            is_halfyear_break = True
            school_break_type = "halfyear"
        
        if 'summer' in breaks and self._is_in_break(check_date, *breaks['summer']):
            is_summer_holiday = True
            school_break_type = "summer"
        
        is_any_school_break = any([
            is_spring_break, is_autumn_break, is_winter_break,
            is_easter_break, is_halfyear_break, is_summer_holiday
        ])
        
        # Vzdálenost k nejbližším prázdninám
        days_to_next_break = 999
        days_from_last_break = 999
        
        for break_type, (break_start, break_end) in breaks.items():
            # Dny do začátku prázdnin
            if check_date < break_start:
                days = (break_start - check_date).days
                days_to_next_break = min(days_to_next_break, days)
            
            # Dny od konce prázdnin
            if check_date > break_end:
                days = (check_date - break_end).days
                days_from_last_break = min(days_from_last_break, days)
        
        # Pokud jsme nenašli žádné budoucí/minulé prázdniny, nastavíme 0
        if days_to_next_break == 999:
            days_to_next_break = 0
        if days_from_last_break == 999:
            days_from_last_break = 0
        
        # Týden před/po prázdninách
        is_week_before_break = 0 < days_to_next_break <= 7
        is_week_after_break = 0 < days_from_last_break <= 7
        
        # Další features
        season_exact = self._get_season(check_date)
        week_position = self._get_week_position(check_date)
        is_month_end = check_date.day >= 25
        school_week_number = self._get_school_week_number(check_date)
        is_bridge_day = self._is_bridge_day(check_date)
        long_weekend_length = self._get_long_weekend_length(check_date)
        
        return {
            # Původní (kompatibilita)
            "is_holiday": is_hol,
            "holiday_name": holiday_name,
            "is_significant": is_sig,
            "significant_name": sig_name,
            
            # ML Model features (19 features)
            "is_spring_break": is_spring_break,
            "is_autumn_break": is_autumn_break,
            "is_winter_break": is_winter_break,
            "is_easter_break": is_easter_break,
            "is_halfyear_break": is_halfyear_break,
            "is_summer_holiday": is_summer_holiday,
            "is_any_school_break": is_any_school_break,
            "school_break_type": school_break_type,
            "days_to_next_break": days_to_next_break,
            "days_from_last_break": days_from_last_break,
            "is_week_before_break": is_week_before_break,
            "is_week_after_break": is_week_after_break,
            "season_exact": season_exact,
            "week_position": week_position,
            "is_month_end": is_month_end,
            "school_week_number": school_week_number,
            "is_bridge_day": is_bridge_day,
            "long_weekend_length": long_weekend_length,
        }
    
    def get_holidays_for_range(self, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Vrací všechny svátky v daném období.
        
        Args:
            start_date: Začátek období
            end_date: Konec období
            
        Returns:
            DataFrame s datumy a názvy svátků
        """
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        holidays = []
        for dt in date_range:
            info = self.get_holiday_info(dt.date())
            if info['is_holiday']:
                holidays.append({
                    'date': dt.date(),
                    'holiday_name': info['holiday_name'],
                    'type': 'state_holiday'
                })
            elif info['is_significant']:
                holidays.append({
                    'date': dt.date(),
                    'holiday_name': info['significant_name'],
                    'type': 'significant_day'
                })
        
        return pd.DataFrame(holidays) if holidays else pd.DataFrame(columns=['date', 'holiday_name', 'type'])
    
    def is_school_holiday_period(self, check_date: date) -> Tuple[bool, Optional[str]]:
        """
        Zjistí, zda je datum v období školních prázdnin.
        
        Args:
            check_date: Datum ke kontrole
            
        Returns:
            Tuple (jsou_prázdniny, název_období)
        """
        month = check_date.month
        day = check_date.day
        
        # Letní prázdniny (červenec + srpen)
        if month in [7, 8]:
            return True, "Letní prázdniny"
        
        # Vánoční prázdniny (23.12 - 2.1)
        if (month == 12 and day >= 23) or (month == 1 and day <= 2):
            return True, "Vánoční prázdniny"
        
        # Pololetní prázdniny (konec ledna - začátek února)
        # Přesné datum se mění, typicky poslední pátek v lednu
        if month == 1 and day >= 28:
            return True, "Pololetní prázdniny"
        if month == 2 and day <= 3:
            return True, "Pololetní prázdniny"
        
        # Jarní prázdniny (únor/březen - regionálně se liší)
        # Zjednodušeně celý únor může být zasažen
        if month == 2:
            return True, "Možné jarní prázdniny (regionální)"
        
        # Velikonoční prázdniny (čtvrtek před Velikonocemi až úterý po)
        easter = self._calculate_easter(check_date.year)
        easter_start = easter - timedelta(days=4)  # Zelený čtvrtek
        easter_end = easter + timedelta(days=2)    # Úterý po Velikonocích
        if easter_start <= check_date <= easter_end:
            return True, "Velikonoční prázdniny"
        
        return False, None


# Globální instance pro použití v API
holiday_service = HolidayService()


if __name__ == '__main__':
    # Testování služby
    print("=" * 60)
    print("Testing Holiday Service")
    print("=" * 60)
    
    # Test konkrétních dat
    test_dates = [
        date(2026, 1, 1),   # Nový rok
        date(2026, 5, 1),   # Svátek práce
        date(2026, 12, 25), # Vánoce
        date(2026, 4, 6),   # Velikonoční pondělí 2026
        date(2026, 7, 15),  # Letní prázdniny
        date(2026, 3, 10),  # Běžný den
    ]
    
    service = HolidayService()
    
    for test_date in test_dates:
        info = service.get_holiday_info(test_date)
        is_school, school_name = service.is_school_holiday_period(test_date)
        
        print(f"\n{test_date.strftime('%Y-%m-%d (%A)')}:")
        print(f"  Státní svátek: {info['is_holiday']}")
        if info['holiday_name']:
            print(f"  Název: {info['holiday_name']}")
        print(f"  Významný den: {info['is_significant']}")
        if info['significant_name']:
            print(f"  Název: {info['significant_name']}")
        print(f"  Školní prázdniny: {is_school}")
        if school_name:
            print(f"  Období: {school_name}")
    
    # Test období
    print("\n" + "=" * 60)
    print("Svátky v lednu 2026:")
    print("=" * 60)
    holidays_df = service.get_holidays_for_range(date(2026, 1, 1), date(2026, 1, 31))
    print(holidays_df.to_string())
