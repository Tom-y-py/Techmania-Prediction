"""
Pomocny skript pro spusteni event scraperu z prikazove radky.
Scrape eventy z GoOut a Plzen.eu a ulozi je do databaze.

Pouziti:
    python run_scraper.py --start 2026-01-01 --end 2026-12-31
    python run_scraper.py --year 2026
    python run_scraper.py --month 2026-03
"""

import argparse
import sys
from pathlib import Path
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from database import SessionLocal, init_db, Event, update_template_event_flag
from services.event_scraper_service import event_scraper_service


def parse_date_args(args):
    """
    Parsuje argumenty pro datum.
    
    Args:
        args: Argumenty z argparse
    
    Returns:
        Tuple (start_date, end_date)
    """
    if args.year:
        # Cely rok
        start_date = date(args.year, 1, 1)
        end_date = date(args.year, 12, 31)
    elif args.month:
        # Cely mesic
        dt = datetime.strptime(args.month, '%Y-%m')
        start_date = dt.date()
        # Posledni den mesice
        end_date = (dt + relativedelta(months=1) - timedelta(days=1)).date()
    elif args.start and args.end:
        # Konkretni rozsah
        start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
        end_date = datetime.strptime(args.end, '%Y-%m-%d').date()
    else:
        # Default: dalsi 3 mesice
        start_date = date.today()
        end_date = start_date + timedelta(days=90)
    
    return start_date, end_date


def run_scraper(start_date: date, end_date: date, dry_run: bool = False):
    """
    Spusti scraper a ulozi eventy do databaze.
    
    Args:
        start_date: Datum od
        end_date: Datum do
        dry_run: Pokud True, jen vypise eventy bez ulozeni
    """
    print("=" * 70)
    print("EVENT SCRAPER PRO TECHMANIA")
    print("=" * 70)
    print(f"\nDatum rozsah: {start_date} - {end_date}")
    print(f"Pocet dni: {(end_date - start_date).days + 1}")
    print(f"Dry run: {'ANO (eventy se neuloží)' if dry_run else 'NE (eventy se uloží do DB)'}")
    print("\n" + "-" * 70 + "\n")
    
    # Spustit scraper
    try:
        events = event_scraper_service.scrape_all_sources(start_date, end_date)
        
        print("\n" + "=" * 70)
        print(f"NALEZENO CELKEM: {len(events)} eventu")
        print("=" * 70)
        
        if len(events) == 0:
            print("\nŽádné eventy nenalezeny.")
            return
        
        # Zobrazit summary
        print("\nSummary podle kategorii:")
        categories = {}
        for event in events:
            cat = event['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count} eventu")
        
        print("\nSummary podle zdroju:")
        sources = {}
        for event in events:
            src = event['source']
            sources[src] = sources.get(src, 0) + 1
        
        for src, count in sorted(sources.items()):
            print(f"  {src}: {count} eventu")
        
        # Zobrazit prvnich 10 eventu
        print("\n" + "-" * 70)
        print("PREVIEW (prvnich 10 eventu):")
        print("-" * 70)
        
        for i, event in enumerate(events[:10], 1):
            print(f"\n{i}. {event['title']}")
            print(f"   Datum: {event['event_date']}")
            print(f"   Misto: {event['venue']}")
            print(f"   Kategorie: {event['category']}")
            print(f"   Attendance: {event['expected_attendance']}")
            print(f"   Impact: {event['impact_level']}/5")
            print(f"   Zdroj: {event['source']}")
        
        if len(events) > 10:
            print(f"\n... a dalších {len(events) - 10} eventu")
        
        # Ulozit do databaze
        if not dry_run:
            print("\n" + "=" * 70)
            print("UKLADANI DO DATABAZE...")
            print("=" * 70)
            
            # Inicializovat databazi
            init_db()
            db = SessionLocal()
            
            try:
                events_saved = 0
                events_skipped = 0
                events_updated = 0
                
                for event_data in events:
                    try:
                        # Kontrola jestli event uz existuje
                        existing = db.query(Event).filter(
                            Event.event_date == event_data['event_date'],
                            Event.title == event_data['title']
                        ).first()
                        
                        if existing:
                            # Update existujiciho eventu
                            if not existing.is_active:
                                existing.is_active = True
                                events_updated += 1
                            else:
                                events_skipped += 1
                        else:
                            # Vytvorit novy event
                            new_event = Event(
                                event_date=event_data['event_date'],
                                title=event_data['title'],
                                description=event_data.get('description'),
                                venue=event_data.get('venue', 'Plzen'),
                                category=event_data.get('category', 'obecne'),
                                expected_attendance=event_data.get('expected_attendance', 'stredni'),
                                source=event_data['source'],
                                source_url=event_data.get('source_url'),
                                impact_level=event_data.get('impact_level', 2),
                                is_active=True
                            )
                            db.add(new_event)
                            events_saved += 1
                        
                        # Aktualizovat template_data is_event flag
                        update_template_event_flag(db, event_data['event_date'])
                    
                    except Exception as e:
                        print(f"\nChyba pri ukladani eventu '{event_data['title']}': {e}")
                        continue
                
                db.commit()
                
                print(f"\nVYSLEDEK:")
                print(f"  Nove ulozeno: {events_saved}")
                print(f"  Aktualizovano: {events_updated}")
                print(f"  Preskoceno (duplicity): {events_skipped}")
                print(f"  Celkem zpracovano: {len(events)}")
                
            except Exception as e:
                db.rollback()
                print(f"\nChyba pri ukladani do databaze: {e}")
                raise
            finally:
                db.close()
        else:
            print("\n[DRY RUN] Eventy nebyly ulozeny do databaze.")
    
    except Exception as e:
        print(f"\nChyba pri spusteni scraperu: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("HOTOVO")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Event scraper pro Plzen a okoli',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Priklady pouziti:
  # Scrape cely rok 2026
  python run_scraper.py --year 2026
  
  # Scrape konkretni mesic
  python run_scraper.py --month 2026-03
  
  # Scrape konkretni rozsah
  python run_scraper.py --start 2026-01-01 --end 2026-03-31
  
  # Dry run (bez ulozeni do DB)
  python run_scraper.py --year 2026 --dry-run
  
  # Dalsi 3 mesice (default)
  python run_scraper.py
        """
    )
    
    # Datum argumenty (vylucujici se)
    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument(
        '--year',
        type=int,
        help='Scrape cely rok (napr. 2026)'
    )
    date_group.add_argument(
        '--month',
        type=str,
        help='Scrape cely mesic (format: YYYY-MM, napr. 2026-03)'
    )
    date_group.add_argument(
        '--start',
        type=str,
        help='Pocatecni datum (format: YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        help='Konecne datum (format: YYYY-MM-DD, pouzije se s --start)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Pouze vypise nalezene eventy bez ulozeni do databaze'
    )
    
    args = parser.parse_args()
    
    # Validace argumentu
    if args.end and not args.start:
        parser.error("--end vyžaduje --start")
    
    # Parsovat datum
    try:
        start_date, end_date = parse_date_args(args)
    except ValueError as e:
        parser.error(f"Chybny format data: {e}")
    
    # Spustit scraper
    run_scraper(start_date, end_date, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
