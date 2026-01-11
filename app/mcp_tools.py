"""
MCP (Model Context Protocol) Tools pro AI Chat.
Definuje nástroje, které může AI volat pro přístup k datům.
"""

import json
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict, Any
from database import SessionLocal, TemplateData, HistoricalData, Prediction


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
                except:
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
    
    visitors = [r.visitors for r in records if r.visitors and r.visitors > 0]
    
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
        return f"Žádná data pro období {start_date}" + (f" až {end_date}" if end_date else "")
    
    result = [f"**Návštěvnost:**\n"]
    for r in records:
        visitors = r.visitors if r.visitors else "N/A"
        result.append(f"- {r.date}: **{visitors}** návštěvníků")
    
    if len(records) > 1:
        visitors = [r.visitors for r in records if r.visitors]
        if visitors:
            result.append(f"\nPrůměr: **{sum(visitors)/len(visitors):.0f}**")
            result.append(f"Celkem: **{sum(visitors)}**")
    
    return "\n".join(result)


def _get_day_of_week_stats(db) -> str:
    """Statistiky podle dne v týdnu."""
    records = db.query(HistoricalData).all()
    
    day_stats = {i: [] for i in range(7)}
    day_names = ['Pondělí', 'Úterý', 'Středa', 'Čtvrtek', 'Pátek', 'Sobota', 'Neděle']
    
    for r in records:
        if r.visitors and r.visitors > 0 and r.date:
            try:
                d = datetime.strptime(str(r.date), "%Y-%m-%d")
                day_stats[d.weekday()].append(r.visitors)
            except:
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
        if r.visitors and r.visitors > 0 and r.date:
            try:
                d = datetime.strptime(str(r.date), "%Y-%m-%d")
                month_stats[d.month].append(r.visitors)
            except:
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
    
    visitors1 = [r.visitors for r in data1 if r.visitors and r.visitors > 0]
    visitors2 = [r.visitors for r in data2 if r.visitors and r.visitors > 0]
    
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
    query = db.query(HistoricalData).filter(HistoricalData.visitors > 0)
    
    if year:
        query = query.filter(HistoricalData.date.like(f"{year}-%"))
    
    if order == "lowest":
        query = query.order_by(HistoricalData.visitors.asc())
        title = "nejnižší"
    else:
        query = query.order_by(HistoricalData.visitors.desc())
        title = "nejvyšší"
    
    records = query.limit(limit).all()
    
    year_str = f" v roce {year}" if year else ""
    result = [f"**Top {limit} dní s {title} návštěvností{year_str}:**\n"]
    
    for i, r in enumerate(records, 1):
        result.append(f"{i}. **{r.date}**: {r.visitors} návštěvníků")
    
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
