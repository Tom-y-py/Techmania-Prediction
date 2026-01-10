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
    
    def get_holiday_info(self, check_date: date) -> Dict:
        """
        Vrací kompletní informace o svátku pro dané datum.
        
        Args:
            check_date: Datum ke kontrole
            
        Returns:
            Slovník s informacemi o svátku
        """
        is_hol, holiday_name = self.is_holiday(check_date)
        is_sig, sig_name = self.is_significant_day(check_date)
        
        return {
            "is_holiday": is_hol,
            "holiday_name": holiday_name,
            "is_significant": is_sig,
            "significant_name": sig_name,
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
