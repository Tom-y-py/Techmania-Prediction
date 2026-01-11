"""
Event scraper service pro Plzen a okoli.
Scrape eventy z ruznych zdroju ktere by mohly ovlivnit navstevnost Techmanie.

Zdroje:
- GoOut Plzen
- Plzen.eu (udalosti)
- Vlastni manualni pridavani eventu
"""

import requests
from bs4 import BeautifulSoup
from datetime import date, datetime, timedelta
from typing import List, Dict, Optional
import re
from urllib.parse import urljoin
import time


class EventScraperService:
    """Scraper pro eventy v Plzni a okoli."""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.base_urls = {
            'goout': 'https://goout.net/cs/plzen/',
            'plzen': 'https://www.plzen.eu/'
        }
    
    def scrape_goout(self, start_date: date, end_date: date) -> List[Dict]:
        """
        Scrape eventy z GoOut Plzen.
        
        Args:
            start_date: Datum od
            end_date: Datum do
        
        Returns:
            List slovniku s informacemi o eventech
        """
        events = []
        
        try:
            # GoOut ma sekce: koncerty, party, divadlo, film, atd.
            categories = ['koncerty', 'party', 'divadlo', 'sport', 'festivaly']
            
            for category in categories:
                try:
                    url = f"{self.base_urls['goout']}{category}/"
                    print(f"Scraping GoOut category: {category}")
                    
                    response = requests.get(url, headers=self.headers, timeout=10)
                    
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        # Najit vsechny event karty
                        # GoOut struktura se muze zmenit, takze toto je genericky pristup
                        event_cards = soup.find_all('article', class_=re.compile('event|item'))
                        
                        if not event_cards:
                            # Zkusit alternativni selektory
                            event_cards = soup.find_all('div', class_=re.compile('event|card'))
                        
                        for card in event_cards[:50]:  # Limit na 50 eventu per kategorie
                            try:
                                event_data = self._parse_goout_event(card, category)
                                if event_data:
                                    # Kontrola datumu
                                    event_date = event_data.get('event_date')
                                    if event_date and start_date <= event_date <= end_date:
                                        events.append(event_data)
                            except Exception as e:
                                print(f"Chyba pri parsovani GoOut eventu: {e}")
                                continue
                    
                    # Respekt k serveru
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"Chyba pri scrapu GoOut kategorie {category}: {e}")
                    continue
        
        except Exception as e:
            print(f"Chyba pri scrapu GoOut: {e}")
        
        return events
    
    def _parse_goout_event(self, card_element, category: str) -> Optional[Dict]:
        """
        Parsuje GoOut event kartu.
        
        Args:
            card_element: BeautifulSoup element s event kartou
            category: Kategorie eventu
        
        Returns:
            Dict s event daty nebo None
        """
        try:
            # Najit nazev eventu
            title_elem = card_element.find(['h2', 'h3', 'a'], class_=re.compile('title|name|heading'))
            title = title_elem.get_text(strip=True) if title_elem else None
            
            if not title:
                return None
            
            # Najit datum
            date_elem = card_element.find(['time', 'span', 'div'], class_=re.compile('date|time|when'))
            date_str = date_elem.get_text(strip=True) if date_elem else None
            
            event_date = self._parse_date_string(date_str)
            
            # Najit misto
            venue_elem = card_element.find(['span', 'div', 'p'], class_=re.compile('venue|place|where|location'))
            venue = venue_elem.get_text(strip=True) if venue_elem else 'Plzen'
            
            # Najit popis
            desc_elem = card_element.find(['p', 'div'], class_=re.compile('desc|perex|text'))
            description = desc_elem.get_text(strip=True)[:500] if desc_elem else None
            
            # Najit URL
            link_elem = card_element.find('a', href=True)
            source_url = urljoin(self.base_urls['goout'], link_elem['href']) if link_elem else None
            
            # Odhad attendance podle kategorie
            expected_attendance = self._estimate_attendance(category, venue, title)
            
            # Impact level podle kategorie a attendance
            impact_level = self._calculate_impact_level(category, expected_attendance, venue)
            
            return {
                'event_date': event_date,
                'title': title,
                'description': description,
                'venue': venue,
                'category': category,
                'expected_attendance': expected_attendance,
                'source': 'goout',
                'source_url': source_url,
                'impact_level': impact_level
            }
            
        except Exception as e:
            print(f"Chyba pri parsovani event karty: {e}")
            return None
    
    def _parse_date_string(self, date_str: Optional[str]) -> Optional[date]:
        """
        Parsuje datum z retezce.
        
        Args:
            date_str: Retezec s datem
        
        Returns:
            date objekt nebo None
        """
        if not date_str:
            return None
        
        try:
            # Ceske mesice
            month_map = {
                'ledna': 1, 'února': 2, 'března': 3, 'dubna': 4,
                'května': 5, 'června': 6, 'července': 7, 'srpna': 8,
                'září': 9, 'října': 10, 'listopadu': 11, 'prosince': 12,
                'led': 1, 'úno': 2, 'bře': 3, 'dub': 4,
                'kvě': 5, 'čer': 6, 'čvc': 7, 'srp': 8,
                'zář': 9, 'říj': 10, 'lis': 11, 'pro': 12
            }
            
            # Odstranit extra whitespace
            date_str = ' '.join(date_str.split())
            
            # Format: "15. ledna 2026" nebo "15.1.2026"
            # Regex pro datum
            patterns = [
                r'(\d{1,2})\.\s*(\d{1,2})\.\s*(\d{4})',  # 15.1.2026
                r'(\d{1,2})\.\s*(\w+)\s*(\d{4})',         # 15. ledna 2026
                r'(\d{4})-(\d{2})-(\d{2})',               # 2026-01-15
            ]
            
            for pattern in patterns:
                match = re.search(pattern, date_str, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    
                    if pattern == patterns[0]:  # DD.MM.YYYY
                        day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
                    elif pattern == patterns[1]:  # DD. month YYYY
                        day = int(groups[0])
                        month_name = groups[1].lower()
                        month = month_map.get(month_name, 1)
                        year = int(groups[2])
                    else:  # YYYY-MM-DD
                        year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                    
                    return date(year, month, day)
            
            # Pokud nic nenaslo, zkusit relativni datum
            date_str_lower = date_str.lower()
            if 'dnes' in date_str_lower or 'today' in date_str_lower:
                return date.today()
            elif 'zítra' in date_str_lower or 'zitra' in date_str_lower:
                return date.today() + timedelta(days=1)
            
        except Exception as e:
            print(f"Chyba pri parsovani data '{date_str}': {e}")
        
        return None
    
    def _estimate_attendance(self, category: str, venue: str, title: str) -> str:
        """
        Odhadne velikost udalosti.
        
        Args:
            category: Kategorie eventu
            venue: Misto konani
            title: Nazev eventu
        
        Returns:
            'male', 'stredni', 'velke', nebo 'masivni'
        """
        title_lower = title.lower() if title else ''
        venue_lower = venue.lower() if venue else ''
        
        # Masivni eventy
        masivni_keywords = ['festival', 'fest', 'slavnosti', 'mezinarodni', 'international']
        if any(kw in title_lower for kw in masivni_keywords):
            return 'masivni'
        
        # Velke eventy
        velke_venues = ['arena', 'stadion', 'hall', 'centrum']
        if any(v in venue_lower for v in velke_venues):
            return 'velke'
        
        # Kategorie-specificke
        if category in ['sport', 'koncerty']:
            return 'stredni'
        elif category in ['party', 'festivaly']:
            return 'velke'
        
        return 'male'
    
    def _calculate_impact_level(self, category: str, attendance: str, venue: str) -> int:
        """
        Vypocita level vlivu na navstevnost Techmanie (1-5).
        
        Args:
            category: Kategorie eventu
            attendance: Velikost udalosti
            venue: Misto konani
        
        Returns:
            Impact level 1-5
        """
        base_impact = {
            'male': 1,
            'stredni': 2,
            'velke': 3,
            'masivni': 4
        }
        
        impact = base_impact.get(attendance, 1)
        
        # Zvysit impact pro urcite kategorie
        high_impact_categories = ['festival', 'sport', 'festivaly']
        if category in high_impact_categories:
            impact += 1
        
        # Snizit impact pro kulturni akce (neprimy konflikt)
        cultural_categories = ['divadlo', 'film']
        if category in cultural_categories:
            impact = max(1, impact - 1)
        
        return min(5, max(1, impact))
    
    def scrape_plzen_eu(self, start_date: date, end_date: date) -> List[Dict]:
        """
        Scrape eventy z Plzen.eu kalendar akci.
        
        Args:
            start_date: Datum od
            end_date: Datum do
        
        Returns:
            List slovniku s informacemi o eventech
        """
        events = []
        
        try:
            # Plzen.eu kalendar
            url = f"{self.base_urls['plzen']}kalendar-akci/"
            print(f"Scraping Plzen.eu: {url}")
            
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Najit event elementy
                event_items = soup.find_all(['article', 'div'], class_=re.compile('event|akce|item'))
                
                for item in event_items[:30]:  # Limit
                    try:
                        event_data = self._parse_plzen_eu_event(item)
                        if event_data:
                            event_date = event_data.get('event_date')
                            if event_date and start_date <= event_date <= end_date:
                                events.append(event_data)
                    except Exception as e:
                        print(f"Chyba pri parsovani Plzen.eu eventu: {e}")
                        continue
        
        except Exception as e:
            print(f"Chyba pri scrapu Plzen.eu: {e}")
        
        return events
    
    def _parse_plzen_eu_event(self, item_element) -> Optional[Dict]:
        """
        Parsuje Plzen.eu event.
        
        Args:
            item_element: BeautifulSoup element
        
        Returns:
            Dict s event daty nebo None
        """
        try:
            # Nazev
            title_elem = item_element.find(['h2', 'h3', 'a'], class_=re.compile('title|heading|name'))
            title = title_elem.get_text(strip=True) if title_elem else None
            
            if not title:
                return None
            
            # Datum
            date_elem = item_element.find(['time', 'span', 'div'], class_=re.compile('date|datum'))
            date_str = date_elem.get_text(strip=True) if date_elem else None
            event_date = self._parse_date_string(date_str)
            
            # Misto
            venue_elem = item_element.find(['span', 'div'], class_=re.compile('venue|misto|location'))
            venue = venue_elem.get_text(strip=True) if venue_elem else 'Plzen'
            
            # Popis
            desc_elem = item_element.find(['p', 'div'], class_=re.compile('desc|text|perex'))
            description = desc_elem.get_text(strip=True)[:500] if desc_elem else None
            
            # URL
            link_elem = item_element.find('a', href=True)
            source_url = urljoin(self.base_urls['plzen'], link_elem['href']) if link_elem else None
            
            return {
                'event_date': event_date,
                'title': title,
                'description': description,
                'venue': venue,
                'category': 'obecne',
                'expected_attendance': 'stredni',
                'source': 'plzen.eu',
                'source_url': source_url,
                'impact_level': 2
            }
            
        except Exception as e:
            print(f"Chyba pri parsovani Plzen.eu event: {e}")
            return None
    
    def scrape_all_sources(self, start_date: date, end_date: date) -> List[Dict]:
        """
        Scrape eventy ze vsech zdroju.
        
        Args:
            start_date: Datum od
            end_date: Datum do
        
        Returns:
            List vsech eventu
        """
        all_events = []
        
        print(f"Zahajuji scraping eventu od {start_date} do {end_date}")
        
        # GoOut
        print("\n[1/2] Scraping GoOut...")
        goout_events = self.scrape_goout(start_date, end_date)
        all_events.extend(goout_events)
        print(f"GoOut: {len(goout_events)} eventu nalezeno")
        
        # Plzen.eu
        print("\n[2/2] Scraping Plzen.eu...")
        plzen_events = self.scrape_plzen_eu(start_date, end_date)
        all_events.extend(plzen_events)
        print(f"Plzen.eu: {len(plzen_events)} eventu nalezeno")
        
        # Odstranit duplicity
        unique_events = self._deduplicate_events(all_events)
        
        print(f"\nCelkem: {len(unique_events)} unikatnich eventu")
        
        return unique_events
    
    def _deduplicate_events(self, events: List[Dict]) -> List[Dict]:
        """
        Odstrani duplicitni eventy.
        
        Args:
            events: List eventu
        
        Returns:
            List unikatnich eventu
        """
        unique = []
        seen = set()
        
        for event in events:
            # Klic: datum + normalizovany nazev
            title_normalized = event['title'].lower().strip()
            key = (event['event_date'], title_normalized)
            
            if key not in seen:
                seen.add(key)
                unique.append(event)
        
        return unique
    
    def create_manual_event(
        self,
        event_date: date,
        title: str,
        description: Optional[str] = None,
        venue: str = 'Plzen',
        category: str = 'custom',
        expected_attendance: str = 'stredni',
        impact_level: int = 2
    ) -> Dict:
        """
        Vytvori manualni event.
        
        Args:
            event_date: Datum eventu
            title: Nazev
            description: Popis
            venue: Misto
            category: Kategorie
            expected_attendance: Odhad navstevnosti
            impact_level: Impact level 1-5
        
        Returns:
            Dict s event daty
        """
        return {
            'event_date': event_date,
            'title': title,
            'description': description,
            'venue': venue,
            'category': category,
            'expected_attendance': expected_attendance,
            'source': 'manual',
            'source_url': None,
            'impact_level': impact_level
        }


# Globalni instance
event_scraper_service = EventScraperService()


if __name__ == '__main__':
    # Test scrapu
    from datetime import date, timedelta
    
    scraper = EventScraperService()
    
    # Scrape eventy pro dalsi 2 tydny
    start = date.today()
    end = start + timedelta(days=14)
    
    events = scraper.scrape_all_sources(start, end)
    
    print("\n" + "="*60)
    print("NALEZENE EVENTY:")
    print("="*60)
    
    for event in events[:10]:  # Zobrazit prvnich 10
        print(f"\nDatum: {event['event_date']}")
        print(f"Nazev: {event['title']}")
        print(f"Misto: {event['venue']}")
        print(f"Kategorie: {event['category']}")
        print(f"Zdroj: {event['source']}")
        print(f"Impact: {event['impact_level']}/5")
