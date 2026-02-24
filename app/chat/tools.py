"""
MCP (Model Context Protocol) Tools pro AI Chat.
Definuje nástroje, které může AI volat pro přístup k datům.
Rozšířená verze s pokročilou analytikou a predikcemi.
"""

import sys
from pathlib import Path
import json
import statistics
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict, Any

# Přidat parent složku do path pro import database
sys.path.insert(0, str(Path(__file__).parent.parent))
from database import SessionLocal, TemplateData, HistoricalData, Prediction, Event


# Definice dostupných nástrojů pro AI
MCP_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_holidays",
            "description": "Získá seznam státních svátků v daném roce. Vrací datum, název svátku a den v týdnu.",
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {
                        "type": "integer",
                        "description": "Rok pro který chceme svátky (např. 2026)"
                    }
                },
                "required": ["year"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_vacations",
            "description": "Získá seznam školních prázdnin v daném roce. Vrací typ prázdnin, datum začátku a konce.",
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {
                        "type": "integer",
                        "description": "Rok pro který chceme prázdniny (např. 2026)"
                    }
                },
                "required": ["year"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_monthly_events",
            "description": "Získá všechny události (svátky, prázdniny) pro konkrétní měsíc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {
                        "type": "integer",
                        "description": "Rok"
                    },
                    "month": {
                        "type": "integer",
                        "description": "Měsíc (1-12)"
                    }
                },
                "required": ["year", "month"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_historical_stats",
            "description": "Získá statistiky historické návštěvnosti - průměr, minimum, maximum, celkový počet záznamů.",
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {
                        "type": "integer",
                        "description": "Volitelně filtrovat podle roku"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_visitors_by_date",
            "description": "Získá návštěvnost pro konkrétní datum nebo rozsah dat z historických dat.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Počáteční datum ve formátu YYYY-MM-DD"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Koncové datum ve formátu YYYY-MM-DD (volitelné, pokud chceme jen jeden den)"
                    }
                },
                "required": ["start_date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_day_of_week_stats",
            "description": "Získá průměrnou návštěvnost podle dne v týdnu (pondělí až neděle).",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_monthly_stats",
            "description": "Získá průměrnou návštěvnost podle měsíce v roce.",
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {
                        "type": "integer",
                        "description": "Volitelně filtrovat podle roku"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "predict_visitors",
            "description": "Vytvoří predikci návštěvnosti pro konkrétní datum v budoucnosti.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Datum pro predikci ve formátu YYYY-MM-DD"
                    }
                },
                "required": ["date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_upcoming_days",
            "description": "Získá informace o následujících dnech včetně svátků a prázdnin.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Počet dní dopředu (výchozí 7)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_years",
            "description": "Porovná návštěvnost mezi dvěma roky - celkový součet, průměr, změna v procentech.",
            "parameters": {
                "type": "object",
                "properties": {
                    "year1": {
                        "type": "integer",
                        "description": "První rok k porovnání"
                    },
                    "year2": {
                        "type": "integer",
                        "description": "Druhý rok k porovnání"
                    }
                },
                "required": ["year1", "year2"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_days",
            "description": "Získá dny s nejvyšší nebo nejnižší návštěvností.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Počet dní k zobrazení (výchozí 10)"
                    },
                    "order": {
                        "type": "string",
                        "description": "Řazení: 'highest' pro nejvyšší, 'lowest' pro nejnižší"
                    },
                    "year": {
                        "type": "integer",
                        "description": "Volitelně filtrovat podle roku"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "search_dates",
            "description": "Vyhledá data podle různých kritérií - svátek, prázdniny, den v týdnu.",
            "parameters": {
                "type": "object",
                "properties": {
                    "is_holiday": {
                        "type": "boolean",
                        "description": "Filtrovat pouze svátky"
                    },
                    "is_vacation": {
                        "type": "boolean",
                        "description": "Filtrovat pouze prázdniny"
                    },
                    "day_of_week": {
                        "type": "string",
                        "description": "Den v týdnu (Pondělí, Úterý, ...)"
                    },
                    "month": {
                        "type": "integer",
                        "description": "Měsíc (1-12)"
                    },
                    "year": {
                        "type": "integer",
                        "description": "Rok"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max počet výsledků"
                    }
                },
                "required": []
            }
        }
    },
    # === NOVÉ POKROČILÉ NÁSTROJE ===
    {
        "type": "function",
        "function": {
            "name": "predict_range",
            "description": "Vytvoří predikce návštěvnosti pro rozsah dat (např. celý měsíc nebo týden). Vrací součet, průměr, min/max a denní breakdown.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Počáteční datum ve formátu YYYY-MM-DD"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Koncové datum ve formátu YYYY-MM-DD"
                    },
                    "include_daily": {
                        "type": "boolean",
                        "description": "Zahrnout denní rozpis (výchozí: false pro rozsahy > 14 dní)"
                    }
                },
                "required": ["start_date", "end_date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_trends",
            "description": "Analyzuje trendy v návštěvnosti - sezónnost, růst/pokles, vliv počasí a prázdnin.",
            "parameters": {
                "type": "object",
                "properties": {
                    "period": {
                        "type": "string",
                        "description": "Období analýzy: 'yearly' (roční), 'monthly' (měsíční), 'weekly' (týdenní), 'seasonal' (sezónní)"
                    },
                    "year": {
                        "type": "integer",
                        "description": "Rok pro analýzu (volitelné)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_impact",
            "description": "Analyzuje vliv počasí na návštěvnost. Vrací korelace mezi teplotou, srážkami a návštěvností.",
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {
                        "type": "integer",
                        "description": "Rok pro analýzu (volitelné)"
                    },
                    "season": {
                        "type": "string",
                        "description": "Roční období: 'spring', 'summer', 'autumn', 'winter'"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_prediction_accuracy",
            "description": "Porovná predikce s reálnými daty a vypočítá přesnost modelu (MAPE, MAE, R2).",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Počáteční datum pro analýzu"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Koncové datum pro analýzu"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_events_impact",
            "description": "Analyzuje vliv událostí v Plzni na návštěvnost Techmanie.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Počáteční datum"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Koncové datum"
                    },
                    "category": {
                        "type": "string",
                        "description": "Kategorie událostí (koncert, sport, festival, ...)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_best_worst_periods",
            "description": "Najde nejlepší a nejhorší období pro návštěvnost (týdny/měsíce).",
            "parameters": {
                "type": "object",
                "properties": {
                    "granularity": {
                        "type": "string",
                        "description": "Granularita: 'week' (týden), 'month' (měsíc)"
                    },
                    "year": {
                        "type": "integer",
                        "description": "Rok pro analýzu"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Počet období k zobrazení (výchozí 5)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_visitor_segments",
            "description": "Analyzuje segmenty návštěvníků - školní vs veřejní, rozdělení podle dnů.",
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {
                        "type": "integer",
                        "description": "Rok pro analýzu"
                    },
                    "month": {
                        "type": "integer",
                        "description": "Měsíc (1-12)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "forecast_summary",
            "description": "Vytvoří souhrnný forecast pro nadcházející období s klíčovými insights.",
            "parameters": {
                "type": "object",
                "properties": {
                    "period": {
                        "type": "string",
                        "description": "Období: 'week' (týden), 'month' (měsíc), 'quarter' (čtvrtletí)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_similar_periods",
            "description": "Porovná podobná období z různých let (např. všechny únory, jarní prázdniny).",
            "parameters": {
                "type": "object",
                "properties": {
                    "period_type": {
                        "type": "string",
                        "description": "Typ období: 'month' (měsíc), 'vacation' (prázdniny), 'season' (roční období)"
                    },
                    "period_value": {
                        "type": "string",
                        "description": "Hodnota: číslo měsíce (1-12), typ prázdnin (winter, spring, summer), nebo roční období"
                    }
                },
                "required": ["period_type", "period_value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_anomalies",
            "description": "Najde anomálie v návštěvnosti - neobvykle vysoké nebo nízké hodnoty.",
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {
                        "type": "integer",
                        "description": "Rok pro analýzu"
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Prahová hodnota pro odchylku (výchozí 2 směrodatné odchylky)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stored_predictions",
            "description": "Získá uložené predikce z databáze pro dané období.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Počáteční datum ve formátu YYYY-MM-DD"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Koncové datum ve formátu YYYY-MM-DD"
                    },
                    "latest_only": {
                        "type": "boolean",
                        "description": "Pouze nejnovější verze predikcí (výchozí: true)"
                    }
                },
                "required": []
            }
        }
    }
]


def execute_tool(tool_name: str, arguments: Dict[str, Any], db_session=None) -> Dict[str, Any]:
    """
    Vykoná nástroj a vrátí výsledek.
    
    Args:
        tool_name: Název nástroje k vykonání
        arguments: Argumenty pro nástroj
        db_session: SQLAlchemy session (volitelné, pokud není předáno, vytvoří se nové)
    
    Returns:
        Dict s výsledkem
    """
    
    # Použít předanou session nebo vytvořit novou
    if db_session:
        db = db_session
        close_db = False
    else:
        db = SessionLocal()
        close_db = True
    
    try:
        if tool_name == "get_holidays":
            result = _get_holidays(db, arguments.get("year", 2026))
        
        elif tool_name == "get_vacations":
            result = _get_vacations(db, arguments.get("year", 2026))
        
        elif tool_name == "get_monthly_events":
            result = _get_monthly_events(db, arguments.get("year", 2026), arguments.get("month", 1))
        
        elif tool_name == "get_historical_stats":
            result = _get_historical_stats(db, arguments.get("year"))
        
        elif tool_name == "get_visitors_by_date":
            result = _get_visitors_by_date(db, arguments.get("start_date"), arguments.get("end_date"))
        
        elif tool_name == "get_day_of_week_stats":
            result = _get_day_of_week_stats(db)
        
        elif tool_name == "get_monthly_stats":
            result = _get_monthly_stats(db, arguments.get("year"))
        
        elif tool_name == "predict_visitors":
            result = _predict_visitors(arguments.get("date"))
        
        elif tool_name == "get_upcoming_days":
            result = _get_upcoming_days(db, arguments.get("days", 7))
        
        elif tool_name == "compare_years":
            result = _compare_years(db, arguments.get("year1"), arguments.get("year2"))
        
        elif tool_name == "get_top_days":
            result = _get_top_days(db, arguments.get("limit", 10), arguments.get("order", "highest"), arguments.get("year"))
        
        elif tool_name == "search_dates":
            result = _search_dates(db, arguments)
        
        # === NOVÉ NÁSTROJE ===
        elif tool_name == "predict_range":
            result = _predict_range(arguments.get("start_date"), arguments.get("end_date"), arguments.get("include_daily", None))
        
        elif tool_name == "analyze_trends":
            result = _analyze_trends(db, arguments.get("period", "monthly"), arguments.get("year"))
        
        elif tool_name == "get_weather_impact":
            result = _get_weather_impact(db, arguments.get("year"), arguments.get("season"))
        
        elif tool_name == "get_prediction_accuracy":
            result = _get_prediction_accuracy(db, arguments.get("start_date"), arguments.get("end_date"))
        
        elif tool_name == "get_events_impact":
            result = _get_events_impact(db, arguments.get("start_date"), arguments.get("end_date"), arguments.get("category"))
        
        elif tool_name == "get_best_worst_periods":
            result = _get_best_worst_periods(db, arguments.get("granularity", "month"), arguments.get("year"), arguments.get("limit", 5))
        
        elif tool_name == "get_visitor_segments":
            result = _get_visitor_segments(db, arguments.get("year"), arguments.get("month"))
        
        elif tool_name == "forecast_summary":
            result = _forecast_summary(db, arguments.get("period", "week"))
        
        elif tool_name == "compare_similar_periods":
            result = _compare_similar_periods(db, arguments.get("period_type"), arguments.get("period_value"))
        
        elif tool_name == "get_anomalies":
            result = _get_anomalies(db, arguments.get("year"), arguments.get("threshold", 2.0))
        
        elif tool_name == "get_stored_predictions":
            result = _get_stored_predictions(db, arguments.get("start_date"), arguments.get("end_date"), arguments.get("latest_only", True))
        
        else:
            result = {"error": f"Neznámý nástroj: {tool_name}"}
        
        return {"success": True, "data": result}
    
    except Exception as e:
        return {"success": False, "error": f"Chyba při vykonávání nástroje {tool_name}: {str(e)}"}
    finally:
        if close_db:
            db.close()


# Implementace jednotlivých nástrojů

def _get_holidays(db, year: int) -> str:
    """Získá svátky pro daný rok."""
    holidays = db.query(TemplateData).filter(
        TemplateData.is_holiday == 1,
        TemplateData.date.like(f"{year}-%")
    ).order_by(TemplateData.date).all()
    
    if not holidays:
        # Fallback na holiday_service
        from services import holiday_service
        result = []
        for month in range(1, 13):
            for day in range(1, 32):
                try:
                    d = date(year, month, day)
                    is_h, name = holiday_service.is_holiday(d)
                    if is_h and name:
                        result.append(f"- **{d.strftime('%d.%m.%Y')}** ({d.strftime('%A')}): {name}")
                except Exception:
                    pass
        return f"Svátky v roce {year}:\n" + "\n".join(result) if result else f"Žádné svátky nenalezeny pro rok {year}"
    
    result = [f"**Svátky v roce {year}:**\n"]
    for h in holidays:
        result.append(f"- **{h.date}** ({h.day_of_week}): {h.nazvy_svatek}")
    
    return "\n".join(result)


def _get_vacations(db, year: int) -> str:
    """Získá prázdniny pro daný rok."""
    vacations = db.query(TemplateData).filter(
        TemplateData.school_break_type != None,
        TemplateData.school_break_type != '',
        TemplateData.date.like(f"{year}-%")
    ).order_by(TemplateData.date).all()
    
    if not vacations:
        return f"Žádné prázdniny nenalezeny pro rok {year}"
    
    # Seskupit podle typu
    vacation_periods = {}
    vacation_names = {
        'winter': 'Vánoční prázdniny',
        'halfyear': 'Pololetní prázdniny',
        'spring': 'Jarní prázdniny',
        'easter': 'Velikonoční prázdniny',
        'summer': 'Letní prázdniny',
        'autumn': 'Podzimní prázdniny'
    }
    
    for v in vacations:
        vtype = v.school_break_type
        if vtype not in vacation_periods:
            vacation_periods[vtype] = {'start': v.date, 'end': v.date, 'count': 1}
        else:
            vacation_periods[vtype]['end'] = v.date
            vacation_periods[vtype]['count'] += 1
    
    result = [f"**Prázdniny v roce {year}:**\n"]
    for vtype, info in sorted(vacation_periods.items(), key=lambda x: x[1]['start']):
        name = vacation_names.get(vtype, vtype)
        result.append(f"- **{name}**: {info['start']} až {info['end']} ({info['count']} dní)")
    
    return "\n".join(result)


def _get_monthly_events(db, year: int, month: int) -> str:
    """Získá události pro měsíc."""
    events = db.query(TemplateData).filter(
        TemplateData.date.like(f"{year}-{month:02d}-%")
    ).order_by(TemplateData.date).all()
    
    month_names = ['', 'leden', 'únor', 'březen', 'duben', 'květen', 'červen',
                   'červenec', 'srpen', 'září', 'říjen', 'listopad', 'prosinec']
    
    result = [f"**{month_names[month].capitalize()} {year}:**\n"]
    
    holidays = [e for e in events if e.is_holiday]
    vacations = [e for e in events if e.school_break_type]
    
    if holidays:
        result.append("*Svátky:*")
        for h in holidays:
            result.append(f"- {h.date} ({h.day_of_week}): {h.nazvy_svatek}")
    
    if vacations:
        vac_types = {}
        for v in vacations:
            if v.school_break_type not in vac_types:
                vac_types[v.school_break_type] = {'start': v.date, 'end': v.date}
            else:
                vac_types[v.school_break_type]['end'] = v.date
        
        vacation_names = {
            'winter': 'Vánoční prázdniny', 'halfyear': 'Pololetní prázdniny',
            'spring': 'Jarní prázdniny', 'easter': 'Velikonoční prázdniny',
            'summer': 'Letní prázdniny', 'autumn': 'Podzimní prázdniny'
        }
        
        result.append("\n*Prázdniny:*")
        for vtype, period in vac_types.items():
            name = vacation_names.get(vtype, vtype)
            result.append(f"- {name}: {period['start']} až {period['end']}")
    
    if not holidays and not vacations:
        result.append("Žádné události v tomto měsíci.")
    
    result.append(f"\nCelkem dní v měsíci: {len(events)}")
    
    return "\n".join(result)


def _get_historical_stats(db, year: Optional[int] = None) -> str:
    """Získá statistiky historické návštěvnosti."""
    query = db.query(HistoricalData)
    
    if year:
        query = query.filter(HistoricalData.date.like(f"{year}-%"))
    
    records = query.all()
    
    if not records:
        return "Žádná historická data nenalezena."
    
    visitors = [r.total_visitors for r in records if r.total_visitors and r.total_visitors > 0]
    
    if not visitors:
        return "Žádná data o návštěvnosti nenalezena."
    
    avg_visitors = sum(visitors) / len(visitors)
    min_visitors = min(visitors)
    max_visitors = max(visitors)
    total_visitors = sum(visitors)
    
    year_str = f" za rok {year}" if year else ""
    
    return f"""**Statistiky návštěvnosti{year_str}:**

- Celkem záznamů: **{len(visitors)}**
- Celková návštěvnost: **{total_visitors:,}**
- Průměrná denní návštěvnost: **{avg_visitors:.0f}**
- Minimální návštěvnost: **{min_visitors}**
- Maximální návštěvnost: **{max_visitors}**"""


def _get_visitors_by_date(db, start_date: str, end_date: Optional[str] = None) -> str:
    """Získá návštěvnost pro datum nebo rozsah."""
    query = db.query(HistoricalData)
    
    if end_date:
        query = query.filter(
            HistoricalData.date >= start_date,
            HistoricalData.date <= end_date
        )
    else:
        query = query.filter(HistoricalData.date == start_date)
    
    records = query.order_by(HistoricalData.date).all()
    
    if not records:
        return f"Pro období {start_date}" + (f" až {end_date}" if end_date else "") + " nejsou k dispozici žádná data v databázi."
    
    result = [f"**Návštěvnost:**\n"]
    for r in records:
        # Hodnota 0 znamená zavřeno
        if r.total_visitors is not None:
            if r.total_visitors == 0:
                result.append(f"- {r.date} ({r.day_of_week}): **Zavřeno** (0 návštěvníků)")
            else:
                result.append(f"- {r.date} ({r.day_of_week}): **{int(r.total_visitors)}** návštěvníků")
        else:
            result.append(f"- {r.date}: Data nejsou k dispozici")
    
    if len(records) > 1:
        visitors = [r.total_visitors for r in records if r.total_visitors and r.total_visitors > 0]
        if visitors:
            result.append(f"\nPrůměr (bez zavřených dní): **{sum(visitors)/len(visitors):.0f}**")
            result.append(f"Celkem: **{sum(visitors):.0f}**")
    
    return "\n".join(result)


def _get_day_of_week_stats(db) -> str:
    """Statistiky podle dne v týdnu."""
    records = db.query(HistoricalData).all()
    
    day_stats = {i: [] for i in range(7)}
    day_names = ['Pondělí', 'Úterý', 'Středa', 'Čtvrtek', 'Pátek', 'Sobota', 'Neděle']
    
    for r in records:
        if r.total_visitors and r.total_visitors > 0 and r.date:
            try:
                d = datetime.strptime(str(r.date), "%Y-%m-%d")
                day_stats[d.weekday()].append(r.total_visitors)
            except Exception:
                pass
    
    result = ["**Průměrná návštěvnost podle dne v týdnu:**\n"]
    for i, name in enumerate(day_names):
        if day_stats[i]:
            avg = sum(day_stats[i]) / len(day_stats[i])
            result.append(f"- {name}: **{avg:.0f}** (z {len(day_stats[i])} dní)")
    
    return "\n".join(result)


def _get_monthly_stats(db, year: Optional[int] = None) -> str:
    """Statistiky podle měsíce."""
    query = db.query(HistoricalData)
    if year:
        query = query.filter(HistoricalData.date.like(f"{year}-%"))
    
    records = query.all()
    
    month_stats = {i: [] for i in range(1, 13)}
    month_names = ['', 'Leden', 'Únor', 'Březen', 'Duben', 'Květen', 'Červen',
                   'Červenec', 'Srpen', 'Září', 'Říjen', 'Listopad', 'Prosinec']
    
    for r in records:
        if r.total_visitors and r.total_visitors > 0 and r.date:
            try:
                d = datetime.strptime(str(r.date), "%Y-%m-%d")
                month_stats[d.month].append(r.total_visitors)
            except Exception:
                pass
    
    year_str = f" za rok {year}" if year else ""
    result = [f"**Průměrná návštěvnost podle měsíce{year_str}:**\n"]
    
    for i in range(1, 13):
        if month_stats[i]:
            avg = sum(month_stats[i]) / len(month_stats[i])
            total = sum(month_stats[i])
            result.append(f"- {month_names[i]}: **{avg:.0f}** (celkem {total:,})")
    
    return "\n".join(result)


def _predict_visitors(date_str: str) -> str:
    """Vytvoří predikci pro datum."""
    import requests
    
    try:
        # Volat interní API
        response = requests.post(
            "http://localhost:8000/predict",
            json={"date": date_str},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            result = [f"**Predikce pro {date_str}:**\n"]
            result.append(f"- Očekávaná návštěvnost: **{data['predicted_visitors']}**")
            result.append(f"- Confidence interval: {data.get('confidence_interval', {}).get('lower', 'N/A')} - {data.get('confidence_interval', {}).get('upper', 'N/A')}")
            result.append(f"- Den v týdnu: {data.get('day_of_week', 'N/A')}")
            
            if data.get('holiday_info', {}).get('is_holiday'):
                result.append(f"- Svátek: {data['holiday_info'].get('holiday_name', 'Ano')}")
            
            return "\n".join(result)
        else:
            return f"Chyba při predikci: {response.text}"
    
    except Exception as e:
        return f"Nepodařilo se vytvořit predikci: {str(e)}"


def _get_upcoming_days(db, days: int = 7) -> str:
    """Získá informace o následujících dnech."""
    today = date.today()
    end_date = today + timedelta(days=days)
    
    records = db.query(TemplateData).filter(
        TemplateData.date >= str(today),
        TemplateData.date <= str(end_date)
    ).order_by(TemplateData.date).all()
    
    result = [f"**Následujících {days} dní:**\n"]
    
    vacation_names = {
        'winter': 'Vánoční prázdniny', 'halfyear': 'Pololetní prázdniny',
        'spring': 'Jarní prázdniny', 'easter': 'Velikonoční prázdniny',
        'summer': 'Letní prázdniny', 'autumn': 'Podzimní prázdniny'
    }
    
    for r in records:
        info = f"- **{r.date}** ({r.day_of_week})"
        extras = []
        
        if r.is_holiday:
            extras.append(f"🎉 {r.nazvy_svatek}")
        if r.school_break_type:
            extras.append(f"🏖️ {vacation_names.get(r.school_break_type, r.school_break_type)}")
        
        if extras:
            info += " - " + ", ".join(extras)
        
        result.append(info)
    
    return "\n".join(result)


def _compare_years(db, year1: int, year2: int) -> str:
    """Porovná dva roky."""
    data1 = db.query(HistoricalData).filter(HistoricalData.date.like(f"{year1}-%")).all()
    data2 = db.query(HistoricalData).filter(HistoricalData.date.like(f"{year2}-%")).all()
    
    visitors1 = [r.total_visitors for r in data1 if r.total_visitors and r.total_visitors > 0]
    visitors2 = [r.total_visitors for r in data2 if r.total_visitors and r.total_visitors > 0]
    
    if not visitors1 or not visitors2:
        return f"Nedostatek dat pro porovnání let {year1} a {year2}"
    
    total1, total2 = sum(visitors1), sum(visitors2)
    avg1, avg2 = total1 / len(visitors1), total2 / len(visitors2)
    
    change_total = ((total2 - total1) / total1) * 100 if total1 > 0 else 0
    change_avg = ((avg2 - avg1) / avg1) * 100 if avg1 > 0 else 0
    
    return f"""**Porovnání let {year1} a {year2}:**

| Metrika | {year1} | {year2} | Změna |
|---------|---------|---------|-------|
| Celkem návštěvníků | {total1:,} | {total2:,} | {change_total:+.1f}% |
| Průměr/den | {avg1:.0f} | {avg2:.0f} | {change_avg:+.1f}% |
| Počet dní | {len(visitors1)} | {len(visitors2)} | - |"""


def _get_top_days(db, limit: int = 10, order: str = "highest", year: Optional[int] = None) -> str:
    """Získá dny s nejvyšší/nejnižší návštěvností."""
    query = db.query(HistoricalData).filter(HistoricalData.total_visitors > 0)
    
    if year:
        query = query.filter(HistoricalData.date.like(f"{year}-%"))
    
    if order == "lowest":
        query = query.order_by(HistoricalData.total_visitors.asc())
        title = "nejnižší"
    else:
        query = query.order_by(HistoricalData.total_visitors.desc())
        title = "nejvyšší"
    
    records = query.limit(limit).all()
    
    year_str = f" v roce {year}" if year else ""
    result = [f"**Top {limit} dní s {title} návštěvností{year_str}:**\n"]
    
    for i, r in enumerate(records, 1):
        result.append(f"{i}. **{r.date}**: {int(r.total_visitors)} návštěvníků")
    
    return "\n".join(result)


def _search_dates(db, args: Dict) -> str:
    """Vyhledá data podle kritérií."""
    query = db.query(TemplateData)
    
    if args.get("is_holiday"):
        query = query.filter(TemplateData.is_holiday == 1)
    
    if args.get("is_vacation"):
        query = query.filter(
            TemplateData.school_break_type != None,
            TemplateData.school_break_type != ''
        )
    
    if args.get("day_of_week"):
        query = query.filter(TemplateData.day_of_week == args["day_of_week"])
    
    if args.get("month"):
        query = query.filter(TemplateData.month == args["month"])
    
    if args.get("year"):
        query = query.filter(TemplateData.date.like(f"{args['year']}-%"))
    
    limit = args.get("limit", 20)
    records = query.order_by(TemplateData.date).limit(limit).all()
    
    if not records:
        return "Žádné výsledky pro zadaná kritéria."
    
    result = [f"**Nalezeno {len(records)} záznamů:**\n"]
    
    vacation_names = {
        'winter': 'Vánoční prázdniny', 'halfyear': 'Pololetní prázdniny',
        'spring': 'Jarní prázdniny', 'easter': 'Velikonoční prázdniny',
        'summer': 'Letní prázdniny', 'autumn': 'Podzimní prázdniny'
    }
    
    for r in records:
        info = f"- {r.date} ({r.day_of_week})"
        if r.is_holiday:
            info += f" - 🎉 {r.nazvy_svatek}"
        if r.school_break_type:
            info += f" - 🏖️ {vacation_names.get(r.school_break_type, r.school_break_type)}"
        result.append(info)
    
    return "\n".join(result)


# =====================================================
# NOVÉ POKROČILÉ ANALYTICKÉ FUNKCE
# =====================================================

def _predict_range(start_date: str, end_date: str, include_daily: Optional[bool] = None) -> str:
    """Vytvoří predikce pro rozsah dat."""
    import requests
    
    try:
        response = requests.post(
            "http://localhost:8000/predict/range",
            json={"start_date": start_date, "end_date": end_date},
            timeout=60
        )
        
        if response.status_code != 200:
            return f"Chyba při predikci: {response.text}"
        
        data = response.json()
        predictions = data.get("predictions", [])
        
        if not predictions:
            return f"Žádné predikce pro období {start_date} až {end_date}"
        
        # Automaticky rozhodnout o denním rozpisu
        days_count = len(predictions)
        if include_daily is None:
            include_daily = days_count <= 14
        
        result = [f"**📊 Predikce pro období {start_date} až {end_date}:**\n"]
        
        # Souhrnné statistiky
        visitors = [p["predicted_visitors"] for p in predictions]
        result.append(f"📈 **Souhrn:**")
        result.append(f"- Celkem očekáváno: **{sum(visitors):,}** návštěvníků")
        result.append(f"- Průměr/den: **{sum(visitors)/len(visitors):.0f}**")
        result.append(f"- Minimum: **{min(visitors)}** | Maximum: **{max(visitors)}**")
        result.append(f"- Počet dní: **{days_count}**")
        
        # Analýza podle dne v týdnu
        weekday_totals = {}
        for p in predictions:
            dow = p.get("day_of_week", "Unknown")
            weekday_totals[dow] = weekday_totals.get(dow, 0) + p["predicted_visitors"]
        
        result.append(f"\n📅 **Podle dne v týdnu:**")
        for dow, total in sorted(weekday_totals.items(), key=lambda x: x[1], reverse=True):
            result.append(f"- {dow}: **{total:,}**")
        
        # Svátky a prázdniny
        holidays = [p for p in predictions if p.get("holiday_info", {}).get("is_holiday")]
        if holidays:
            result.append(f"\n🎉 **Svátky v období:** {len(holidays)} dní")
        
        # Denní rozpis (pokud je povolený)
        if include_daily:
            result.append(f"\n📋 **Denní rozpis:**")
            for p in predictions:
                info = f"- {p['date']} ({p.get('day_of_week', '')[:2]}): **{p['predicted_visitors']}**"
                if p.get("holiday_info", {}).get("is_holiday"):
                    info += " 🎉"
                result.append(info)
        else:
            result.append(f"\n_(Denní rozpis vynechán, období > 14 dní)_")
        
        return "\n".join(result)
        
    except Exception as e:
        return f"Nepodařilo se vytvořit predikce: {str(e)}"


def _analyze_trends(db, period: str = "monthly", year: Optional[int] = None) -> str:
    """Analyzuje trendy v návštěvnosti."""
    query = db.query(HistoricalData).filter(HistoricalData.total_visitors > 0)
    
    if year:
        query = query.filter(HistoricalData.date.like(f"{year}-%"))
    
    records = query.order_by(HistoricalData.date).all()
    
    if not records:
        return "Nedostatek dat pro analýzu trendů."
    
    result = [f"**📈 Analýza trendů návštěvnosti:**\n"]
    
    if period == "monthly":
        # Měsíční trendy
        monthly = {}
        for r in records:
            if r.date and r.total_visitors:
                try:
                    d = datetime.strptime(str(r.date), "%Y-%m-%d")
                    key = f"{d.year}-{d.month:02d}"
                    if key not in monthly:
                        monthly[key] = []
                    monthly[key].append(r.total_visitors)
                except Exception:
                    pass
        
        result.append("**Měsíční průměry:**")
        prev_avg = None
        for key in sorted(monthly.keys())[-12:]:  # Posledních 12 měsíců
            avg = sum(monthly[key]) / len(monthly[key])
            trend = ""
            if prev_avg:
                change = ((avg - prev_avg) / prev_avg) * 100
                trend = f" ({change:+.1f}%)" if abs(change) > 0.5 else ""
            result.append(f"- {key}: **{avg:.0f}**{trend}")
            prev_avg = avg
    
    elif period == "seasonal":
        # Sezónní analýza
        seasons = {"Jaro": [], "Léto": [], "Podzim": [], "Zima": []}
        for r in records:
            if r.date and r.total_visitors:
                try:
                    d = datetime.strptime(str(r.date), "%Y-%m-%d")
                    if d.month in [3, 4, 5]:
                        seasons["Jaro"].append(r.total_visitors)
                    elif d.month in [6, 7, 8]:
                        seasons["Léto"].append(r.total_visitors)
                    elif d.month in [9, 10, 11]:
                        seasons["Podzim"].append(r.total_visitors)
                    else:
                        seasons["Zima"].append(r.total_visitors)
                except Exception:
                    pass
        
        result.append("**Sezónní průměry:**")
        for season, values in seasons.items():
            if values:
                avg = sum(values) / len(values)
                total = sum(values)
                result.append(f"- {season}: **{avg:.0f}**/den (celkem {total:,})")
    
    elif period == "yearly":
        # Roční trendy
        yearly = {}
        for r in records:
            if r.date and r.total_visitors:
                try:
                    d = datetime.strptime(str(r.date), "%Y-%m-%d")
                    if d.year not in yearly:
                        yearly[d.year] = []
                    yearly[d.year].append(r.total_visitors)
                except Exception:
                    pass
        
        result.append("**Roční přehled:**")
        prev_total = None
        for y in sorted(yearly.keys()):
            total = sum(yearly[y])
            avg = total / len(yearly[y])
            trend = ""
            if prev_total:
                change = ((total - prev_total) / prev_total) * 100
                trend = f" ({change:+.1f}%)"
            result.append(f"- {y}: **{total:,}** celkem, **{avg:.0f}**/den{trend}")
            prev_total = total
    
    elif period == "weekly":
        # Týdenní vzory
        weekdays = {i: [] for i in range(7)}
        day_names = ['Po', 'Út', 'St', 'Čt', 'Pá', 'So', 'Ne']
        
        for r in records:
            if r.date and r.total_visitors:
                try:
                    d = datetime.strptime(str(r.date), "%Y-%m-%d")
                    weekdays[d.weekday()].append(r.total_visitors)
                except Exception:
                    pass
        
        result.append("**Průměr podle dne v týdnu:**")
        for i, name in enumerate(day_names):
            if weekdays[i]:
                avg = sum(weekdays[i]) / len(weekdays[i])
                result.append(f"- {name}: **{avg:.0f}**")
    
    return "\n".join(result)


def _get_weather_impact(db, year: Optional[int] = None, season: Optional[str] = None) -> str:
    """Analyzuje vliv počasí na návštěvnost."""
    query = db.query(HistoricalData).filter(
        HistoricalData.total_visitors > 0,
        HistoricalData.temperature_mean != None
    )
    
    if year:
        query = query.filter(HistoricalData.date.like(f"{year}-%"))
    
    records = query.all()
    
    if season:
        season_months = {
            "spring": [3, 4, 5],
            "summer": [6, 7, 8],
            "autumn": [9, 10, 11],
            "winter": [12, 1, 2]
        }
        months = season_months.get(season, [])
        if months:
            filtered = []
            for r in records:
                try:
                    d = datetime.strptime(str(r.date), "%Y-%m-%d")
                    if d.month in months:
                        filtered.append(r)
                except Exception:
                    pass
            records = filtered
    
    if len(records) < 10:
        return "Nedostatek dat pro analýzu vlivu počasí."
    
    result = ["**🌤️ Vliv počasí na návštěvnost:**\n"]
    
    # Teplota
    temp_visitors = [(r.temperature_mean, r.total_visitors) for r in records if r.temperature_mean is not None]
    if temp_visitors:
        cold = [v for t, v in temp_visitors if t < 5]
        mild = [v for t, v in temp_visitors if 5 <= t < 15]
        warm = [v for t, v in temp_visitors if 15 <= t < 25]
        hot = [v for t, v in temp_visitors if t >= 25]
        
        result.append("**Podle teploty:**")
        if cold:
            result.append(f"- < 5°C: **{sum(cold)/len(cold):.0f}**/den ({len(cold)} dní)")
        if mild:
            result.append(f"- 5-15°C: **{sum(mild)/len(mild):.0f}**/den ({len(mild)} dní)")
        if warm:
            result.append(f"- 15-25°C: **{sum(warm)/len(warm):.0f}**/den ({len(warm)} dní)")
        if hot:
            result.append(f"- > 25°C: **{sum(hot)/len(hot):.0f}**/den ({len(hot)} dní)")
    
    # Srážky
    rainy = [r.total_visitors for r in records if r.is_rainy]
    dry = [r.total_visitors for r in records if not r.is_rainy]
    
    if rainy and dry:
        result.append("\n**Podle srážek:**")
        result.append(f"- Deštivé dny: **{sum(rainy)/len(rainy):.0f}**/den ({len(rainy)} dní)")
        result.append(f"- Suché dny: **{sum(dry)/len(dry):.0f}**/den ({len(dry)} dní)")
        diff = ((sum(dry)/len(dry)) - (sum(rainy)/len(rainy))) / (sum(rainy)/len(rainy)) * 100
        result.append(f"- Rozdíl: **{diff:+.1f}%** při hezkém počasí")
    
    # Hezké počasí
    nice = [r.total_visitors for r in records if r.is_nice_weather]
    bad = [r.total_visitors for r in records if not r.is_nice_weather]
    
    if nice and bad:
        result.append("\n**Hezké vs špatné počasí:**")
        result.append(f"- Hezké: **{sum(nice)/len(nice):.0f}**/den")
        result.append(f"- Špatné: **{sum(bad)/len(bad):.0f}**/den")
    
    return "\n".join(result)


def _get_prediction_accuracy(db, start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
    """Porovná predikce s reálnými daty."""
    # Získat predikce a kompletní template data
    pred_query = db.query(Prediction)
    template_query = db.query(TemplateData).filter(TemplateData.is_complete == True)
    
    if start_date:
        pred_query = pred_query.filter(Prediction.prediction_date >= start_date)
        template_query = template_query.filter(TemplateData.date >= start_date)
    if end_date:
        pred_query = pred_query.filter(Prediction.prediction_date <= end_date)
        template_query = template_query.filter(TemplateData.date <= end_date)
    
    predictions = pred_query.all()
    actuals = {str(t.date): t.total_visitors for t in template_query.all() if t.total_visitors}
    
    if not predictions or not actuals:
        return "Nedostatek dat pro porovnání predikcí s realitou. Potřebujeme predikce i reálná data."
    
    # Spárovat predikce s reálnými hodnotami
    pairs = []
    for p in predictions:
        date_str = str(p.prediction_date)
        if date_str in actuals:
            pairs.append({
                "date": date_str,
                "predicted": p.predicted_visitors,
                "actual": actuals[date_str]
            })
    
    if len(pairs) < 3:
        return f"Nalezeno pouze {len(pairs)} spárovaných záznamů. Potřebujeme minimálně 3."
    
    # Výpočet metrik
    errors = [abs(p["predicted"] - p["actual"]) for p in pairs]
    percentage_errors = [abs(p["predicted"] - p["actual"]) / p["actual"] * 100 for p in pairs if p["actual"] > 0]
    
    mae = sum(errors) / len(errors)
    mape = sum(percentage_errors) / len(percentage_errors) if percentage_errors else 0
    
    result = ["**📊 Přesnost predikcí:**\n"]
    result.append(f"Analyzováno **{len(pairs)}** dní s reálnými daty.\n")
    result.append(f"- **MAE** (průměrná absolutní chyba): **{mae:.0f}** návštěvníků")
    result.append(f"- **MAPE** (průměrná procentuální chyba): **{mape:.1f}%**")
    
    # Nejlepší a nejhorší predikce
    pairs_sorted = sorted(pairs, key=lambda x: abs(x["predicted"] - x["actual"]))
    
    result.append("\n**Nejpřesnější predikce:**")
    for p in pairs_sorted[:3]:
        err = abs(p["predicted"] - p["actual"])
        result.append(f"- {p['date']}: predikce {p['predicted']}, realita {p['actual']} (±{err})")
    
    result.append("\n**Největší odchylky:**")
    for p in pairs_sorted[-3:]:
        err = p["predicted"] - p["actual"]
        result.append(f"- {p['date']}: predikce {p['predicted']}, realita {p['actual']} ({err:+d})")
    
    return "\n".join(result)


def _get_events_impact(db, start_date: Optional[str] = None, end_date: Optional[str] = None, category: Optional[str] = None) -> str:
    """Analyzuje vliv událostí na návštěvnost."""
    query = db.query(Event).filter(Event.is_active == True)
    
    if start_date:
        query = query.filter(Event.event_date >= start_date)
    if end_date:
        query = query.filter(Event.event_date <= end_date)
    if category:
        query = query.filter(Event.category == category)
    
    events = query.order_by(Event.event_date).all()
    
    if not events:
        return "Žádné události nenalezeny pro zadané období."
    
    result = ["**🎭 Události a jejich vliv:**\n"]
    
    # Seskupit podle kategorie
    by_category = {}
    for e in events:
        cat = e.category or "Ostatní"
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(e)
    
    for cat, cat_events in sorted(by_category.items()):
        result.append(f"\n**{cat}** ({len(cat_events)} událostí):")
        for e in cat_events[:5]:  # Max 5 per kategorie
            impact = "⭐" * e.impact_level if e.impact_level else ""
            result.append(f"- {e.event_date}: {e.title} {impact}")
        if len(cat_events) > 5:
            result.append(f"  _...a dalších {len(cat_events) - 5}_")
    
    # Souhrn
    high_impact = len([e for e in events if e.impact_level and e.impact_level >= 4])
    result.append(f"\n**Souhrn:** {len(events)} událostí, {high_impact} s vysokým vlivem")
    
    return "\n".join(result)


def _get_best_worst_periods(db, granularity: str = "month", year: Optional[int] = None, limit: int = 5) -> str:
    """Najde nejlepší a nejhorší období."""
    query = db.query(HistoricalData).filter(HistoricalData.total_visitors > 0)
    
    if year:
        query = query.filter(HistoricalData.date.like(f"{year}-%"))
    
    records = query.all()
    
    if not records:
        return "Nedostatek dat pro analýzu."
    
    periods = {}
    
    for r in records:
        if not r.date or not r.total_visitors:
            continue
        try:
            d = datetime.strptime(str(r.date), "%Y-%m-%d")
            if granularity == "week":
                key = f"{d.year}-W{d.isocalendar()[1]:02d}"
            else:  # month
                key = f"{d.year}-{d.month:02d}"
            
            if key not in periods:
                periods[key] = {"total": 0, "count": 0, "days": []}
            periods[key]["total"] += r.total_visitors
            periods[key]["count"] += 1
            periods[key]["days"].append(r.total_visitors)
        except Exception:
            pass
    
    # Seřadit podle průměru
    sorted_periods = sorted(
        [(k, v["total"]/v["count"], v["total"], v["count"]) for k, v in periods.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    gran_name = "týdnů" if granularity == "week" else "měsíců"
    year_str = f" v {year}" if year else ""
    
    result = [f"**🏆 Nejlepší a nejhorší období{year_str}:**\n"]
    
    result.append(f"**Top {limit} {gran_name}:**")
    for key, avg, total, count in sorted_periods[:limit]:
        result.append(f"- {key}: **{avg:.0f}**/den (celkem {total:,}, {count} dní)")
    
    result.append(f"\n**Nejslabších {limit} {gran_name}:**")
    for key, avg, total, count in sorted_periods[-limit:]:
        result.append(f"- {key}: **{avg:.0f}**/den (celkem {total:,}, {count} dní)")
    
    return "\n".join(result)


def _get_visitor_segments(db, year: Optional[int] = None, month: Optional[int] = None) -> str:
    """Analyzuje segmenty návštěvníků."""
    query = db.query(HistoricalData).filter(HistoricalData.total_visitors > 0)
    
    if year:
        if month:
            query = query.filter(HistoricalData.date.like(f"{year}-{month:02d}-%"))
        else:
            query = query.filter(HistoricalData.date.like(f"{year}-%"))
    
    records = query.all()
    
    if not records:
        return "Nedostatek dat pro analýzu segmentů."
    
    result = ["**👥 Analýza segmentů návštěvníků:**\n"]
    
    # Celkové součty
    school_total = sum(r.school_visitors or 0 for r in records)
    public_total = sum(r.public_visitors or 0 for r in records)
    total = sum(r.total_visitors or 0 for r in records)
    
    if school_total > 0 or public_total > 0:
        result.append("**Rozdělení návštěvníků:**")
        result.append(f"- Školní skupiny: **{school_total:,}** ({school_total/total*100:.1f}%)")
        result.append(f"- Veřejnost: **{public_total:,}** ({public_total/total*100:.1f}%)")
        result.append(f"- Celkem: **{total:,}**")
    
    # Podle dne v týdnu
    weekday_segments = {i: {"school": 0, "public": 0, "total": 0} for i in range(7)}
    day_names = ['Po', 'Út', 'St', 'Čt', 'Pá', 'So', 'Ne']
    
    for r in records:
        if r.date:
            try:
                d = datetime.strptime(str(r.date), "%Y-%m-%d")
                weekday_segments[d.weekday()]["school"] += r.school_visitors or 0
                weekday_segments[d.weekday()]["public"] += r.public_visitors or 0
                weekday_segments[d.weekday()]["total"] += r.total_visitors or 0
            except Exception:
                pass
    
    result.append("\n**Podle dne v týdnu:**")
    for i, name in enumerate(day_names):
        seg = weekday_segments[i]
        if seg["total"] > 0:
            school_pct = seg["school"] / seg["total"] * 100 if seg["total"] > 0 else 0
            result.append(f"- {name}: školy **{school_pct:.0f}%**, veřejnost **{100-school_pct:.0f}%**")
    
    return "\n".join(result)


def _forecast_summary(db, period: str = "week") -> str:
    """Vytvoří souhrnný forecast."""
    import requests
    
    today = date.today()
    
    if period == "week":
        end_date = today + timedelta(days=7)
        period_name = "příští týden"
    elif period == "month":
        end_date = today + timedelta(days=30)
        period_name = "příští měsíc"
    elif period == "quarter":
        end_date = today + timedelta(days=90)
        period_name = "příští čtvrtletí"
    else:
        end_date = today + timedelta(days=7)
        period_name = "příští týden"
    
    try:
        response = requests.post(
            "http://localhost:8000/predict/range",
            json={"start_date": str(today), "end_date": str(end_date)},
            timeout=60
        )
        
        if response.status_code != 200:
            return f"Chyba při získávání forecastu: {response.text}"
        
        data = response.json()
        predictions = data.get("predictions", [])
        
        if not predictions:
            return "Nepodařilo se získat predikce."
        
        result = [f"**🔮 Forecast na {period_name}:**\n"]
        result.append(f"Období: {today} až {end_date}\n")
        
        visitors = [p["predicted_visitors"] for p in predictions]
        
        result.append("**📊 Klíčové metriky:**")
        result.append(f"- Očekávaná celková návštěvnost: **{sum(visitors):,}**")
        result.append(f"- Průměr/den: **{sum(visitors)/len(visitors):.0f}**")
        result.append(f"- Nejsilnější den: **{max(visitors)}**")
        result.append(f"- Nejslabší den: **{min(visitors)}**")
        
        # Top 3 dny
        sorted_preds = sorted(predictions, key=lambda x: x["predicted_visitors"], reverse=True)
        result.append("\n**🏆 Nejsilnější dny:**")
        for p in sorted_preds[:3]:
            result.append(f"- {p['date']} ({p.get('day_of_week', '')[:2]}): **{p['predicted_visitors']}**")
        
        # Svátky a prázdniny
        holidays = [p for p in predictions if p.get("holiday_info", {}).get("is_holiday")]
        if holidays:
            result.append(f"\n🎉 **Svátky:** {len(holidays)} dní")
        
        # Varování
        low_days = [p for p in predictions if p["predicted_visitors"] < sum(visitors)/len(visitors) * 0.7]
        if low_days:
            result.append(f"\n⚠️ **Pozor:** {len(low_days)} dní s nižší než průměrnou návštěvností")
        
        return "\n".join(result)
        
    except Exception as e:
        return f"Nepodařilo se vytvořit forecast: {str(e)}"


def _compare_similar_periods(db, period_type: str, period_value: str) -> str:
    """Porovná podobná období z různých let."""
    query = db.query(HistoricalData).filter(HistoricalData.total_visitors > 0)
    records = query.all()
    
    if not records:
        return "Nedostatek dat pro porovnání."
    
    # Filtrovat podle typu období
    yearly_data = {}
    
    for r in records:
        if not r.date or not r.total_visitors:
            continue
        try:
            d = datetime.strptime(str(r.date), "%Y-%m-%d")
            year = d.year
            
            include = False
            
            if period_type == "month":
                if d.month == int(period_value):
                    include = True
            elif period_type == "vacation":
                if r.school_break_type and r.school_break_type.lower() == period_value.lower():
                    include = True
            elif period_type == "season":
                season_months = {
                    "spring": [3, 4, 5],
                    "summer": [6, 7, 8],
                    "autumn": [9, 10, 11],
                    "winter": [12, 1, 2]
                }
                if d.month in season_months.get(period_value.lower(), []):
                    include = True
            
            if include:
                if year not in yearly_data:
                    yearly_data[year] = []
                yearly_data[year].append(r.total_visitors)
        except Exception:
            pass
    
    if not yearly_data:
        return f"Žádná data pro {period_type}={period_value}"
    
    period_names = {
        "month": {
            "1": "leden", "2": "únor", "3": "březen", "4": "duben",
            "5": "květen", "6": "červen", "7": "červenec", "8": "srpen",
            "9": "září", "10": "říjen", "11": "listopad", "12": "prosinec"
        },
        "vacation": {
            "winter": "Vánoční prázdniny", "spring": "Jarní prázdniny",
            "summer": "Letní prázdniny", "autumn": "Podzimní prázdniny",
            "easter": "Velikonoční prázdniny", "halfyear": "Pololetní prázdniny"
        },
        "season": {
            "spring": "Jaro", "summer": "Léto", "autumn": "Podzim", "winter": "Zima"
        }
    }
    
    name = period_names.get(period_type, {}).get(period_value, period_value)
    
    result = [f"**📊 Porovnání: {name}**\n"]
    
    result.append("| Rok | Celkem | Průměr/den | Dní | Změna |")
    result.append("|-----|--------|------------|-----|-------|")
    
    prev_total = None
    for year in sorted(yearly_data.keys()):
        values = yearly_data[year]
        total = sum(values)
        avg = total / len(values)
        change = ""
        if prev_total:
            pct = ((total - prev_total) / prev_total) * 100
            change = f"{pct:+.1f}%"
        result.append(f"| {year} | {total:,} | {avg:.0f} | {len(values)} | {change} |")
        prev_total = total
    
    return "\n".join(result)


def _get_anomalies(db, year: Optional[int] = None, threshold: float = 2.0) -> str:
    """Najde anomálie v návštěvnosti."""
    query = db.query(HistoricalData).filter(HistoricalData.total_visitors > 0)
    
    if year:
        query = query.filter(HistoricalData.date.like(f"{year}-%"))
    
    records = query.order_by(HistoricalData.date).all()
    
    if len(records) < 10:
        return "Nedostatek dat pro detekci anomálií."
    
    visitors = [r.total_visitors for r in records if r.total_visitors]
    mean = sum(visitors) / len(visitors)
    std = statistics.stdev(visitors)
    
    anomalies_high = []
    anomalies_low = []
    
    for r in records:
        if r.total_visitors:
            z_score = (r.total_visitors - mean) / std
            if z_score > threshold:
                anomalies_high.append((r.date, r.total_visitors, z_score))
            elif z_score < -threshold:
                anomalies_low.append((r.date, r.total_visitors, z_score))
    
    year_str = f" v roce {year}" if year else ""
    result = [f"**🔍 Anomálie v návštěvnosti{year_str}:**\n"]
    result.append(f"Průměr: {mean:.0f}, Směrodatná odchylka: {std:.0f}")
    result.append(f"Práh: ±{threshold} σ\n")
    
    if anomalies_high:
        result.append(f"**📈 Neobvykle vysoká návštěvnost ({len(anomalies_high)} dní):**")
        for date, visitors, z in sorted(anomalies_high, key=lambda x: x[2], reverse=True)[:5]:
            result.append(f"- {date}: **{visitors}** (+{z:.1f}σ)")
    
    if anomalies_low:
        result.append(f"\n**📉 Neobvykle nízká návštěvnost ({len(anomalies_low)} dní):**")
        for date, visitors, z in sorted(anomalies_low, key=lambda x: x[2])[:5]:
            result.append(f"- {date}: **{visitors}** ({z:.1f}σ)")
    
    if not anomalies_high and not anomalies_low:
        result.append("Žádné významné anomálie nenalezeny.")
    
    return "\n".join(result)


def _get_stored_predictions(db, start_date: Optional[str] = None, end_date: Optional[str] = None, latest_only: bool = True) -> str:
    """Získá uložené predikce z databáze."""
    query = db.query(Prediction)
    
    if start_date:
        query = query.filter(Prediction.prediction_date >= start_date)
    if end_date:
        query = query.filter(Prediction.prediction_date <= end_date)
    
    if latest_only:
        # Získat pouze nejnovější verze pro každé datum
        from sqlalchemy import func
        subquery = db.query(
            Prediction.prediction_date,
            func.max(Prediction.version).label("max_version")
        ).group_by(Prediction.prediction_date).subquery()
        
        query = query.join(
            subquery,
            (Prediction.prediction_date == subquery.c.prediction_date) &
            (Prediction.version == subquery.c.max_version)
        )
    
    predictions = query.order_by(Prediction.prediction_date).limit(100).all()
    
    if not predictions:
        return "Žádné uložené predikce nenalezeny."
    
    result = ["**📦 Uložené predikce:**\n"]
    
    # Souhrn
    visitors = [p.predicted_visitors for p in predictions]
    result.append(f"Nalezeno **{len(predictions)}** predikcí")
    result.append(f"- Rozsah: {predictions[0].prediction_date} až {predictions[-1].prediction_date}")
    result.append(f"- Průměr: **{sum(visitors)/len(visitors):.0f}**/den")
    result.append(f"- Min/Max: **{min(visitors)}** / **{max(visitors)}**\n")
    
    # Posledních 10
    result.append("**Posledních 10 predikcí:**")
    for p in predictions[-10:]:
        conf = ""
        if p.confidence_lower and p.confidence_upper:
            conf = f" [{p.confidence_lower:.0f}-{p.confidence_upper:.0f}]"
        result.append(f"- {p.prediction_date}: **{p.predicted_visitors}**{conf}")
    
    return "\n".join(result)
