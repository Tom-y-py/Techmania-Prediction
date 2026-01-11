"""
AI Chat modul pro Techmania Prediction API.
Poskytuje inteligentní asistenci s přístupem k datům o návštěvnosti.
Používá MCP (Model Context Protocol) pro přístup k databázi a API.
"""

import os
import json
from datetime import date, datetime, timedelta
from typing import Generator, Optional, List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# Import MCP tools
from mcp_tools import MCP_TOOLS, execute_tool

# Inicializace Groq klienta - lazy loading
_client = None

def get_groq_client():
    """Získá Groq klienta (lazy loading)."""
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY není nastaven. Přidejte ho do .env souboru.")
        from groq import Groq
        _client = Groq(api_key=api_key)
    return _client

# Systémový prompt pro AI s MCP
SYSTEM_PROMPT = """Jsi AI asistent pro Techmania Science Center v Plzni. Pomáháš analyzovat data o návštěvnosti a poskytovat insights.

Máš k dispozici nástroje (tools) pro přístup k datům:
- get_holidays: Získej seznam svátků
- get_vacations: Získej období prázdnin
- get_monthly_events: Všechny události v daném měsíci
- get_historical_stats: Historické statistiky návštěvnosti
- get_visitors_by_date: Návštěvnost pro konkrétní datum/období
- get_day_of_week_stats: Statistiky podle dnů v týdnu
- get_monthly_stats: Měsíční statistiky návštěvnosti
- predict_visitors: Predikce návštěvnosti pro konkrétní datum
- get_upcoming_days: Nadcházející dny s událostmi
- compare_years: Porovnání roků
- get_top_days: Nejnavštěvovanější/nejméně navštěvované dny
- search_dates: Vyhledávání podle kritérií

VŽDY používej nástroje k získání aktuálních dat. Nikdy nevymýšlej čísla.
Odpovídej česky, stručně a věcně.

Formátuj odpovědi s markdown:
- **Tučné** pro důležité hodnoty
- Seznamy pro přehlednost
- Tabulky pro srovnání"""


def get_tools_for_groq():
    """Vrátí MCP tools ve formátu pro Groq API."""
    # MCP_TOOLS už jsou ve správném formátu pro Groq
    return MCP_TOOLS


def chat_stream(
    message: str,
    db_session,
    history: Optional[List[Dict[str, str]]] = None
) -> Generator[str, None, None]:
    """
    Streamuje odpověď od AI s podporou MCP tools.
    
    Args:
        message: Zpráva od uživatele
        db_session: SQLAlchemy session pro přístup k databázi
        history: Historie konverzace
        
    Yields:
        Části odpovědi pro streaming
    """
    
    # Sestavit zprávy
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    
    # Přidat historii
    if history:
        for h in history[-10:]:  # Posledních 10 zpráv
            messages.append({"role": h["role"], "content": h["content"]})
    
    # Přidat aktuální zprávu
    messages.append({"role": "user", "content": message})
    
    try:
        # Získat klienta
        client = get_groq_client()
        tools = get_tools_for_groq()
        
        # První volání - AI rozhodne jestli použít nástroje
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.7,
            max_tokens=2048
        )
        
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        
        # Pokud AI chce použít nástroje
        if tool_calls:
            # Přidat odpověď AI do historie
            messages.append({
                "role": "assistant",
                "content": response_message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in tool_calls
                ]
            })
            
            # Vykonat všechny tool calls
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                # Spustit nástroj
                tool_result = execute_tool(function_name, function_args, db_session)
                
                # Přidat výsledek do zpráv
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result, ensure_ascii=False, default=str)
                })
            
            # Druhé volání - AI zpracuje výsledky a odpoví
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
                stream=True
            )
            
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        else:
            # AI nepotřebuje nástroje, vrátit odpověď přímo
            if response_message.content:
                yield response_message.content
                
    except ValueError as e:
        yield f"\n\n⚠️ {str(e)}"
    except Exception as e:
        yield f"\n\n❌ Chyba při generování odpovědi: {str(e)}"


def chat_sync(
    message: str,
    db_session,
    history: Optional[List[Dict[str, str]]] = None
) -> str:
    """Synchronní verze chatu (bez streamování)."""
    
    response_parts = []
    for part in chat_stream(message, db_session, history):
        response_parts.append(part)
    
    return "".join(response_parts)
