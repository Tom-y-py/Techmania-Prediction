"""
AI Chat modul pro Techmania Prediction API.
Poskytuje inteligentní asistenci s přístupem k datům o návštěvnosti.
Používá MCP (Model Context Protocol) pro přístup k databázi a API.
"""

import json
import re
from datetime import date as date_cls
from typing import Generator, Optional, List, Dict, Any

from .client import get_ai_client, ChatMessage
from .tools import MCP_TOOLS, execute_tool


# === KONSTANTY ===

# Klíčová slova indikující potřebu dat
DATA_KEYWORDS = frozenset([
    'návštěvnost', 'kolik', 'predikce', 'forecast', 'statistik',
    'průměr', 'porovnej', 'rok', 'měsíc', 'den', 'počasí', 'trend',
    'kdy', 'jaká', 'jaký', 'maximum', 'minimum', 'celkem'
])

# Maximální počet zpráv z historie
MAX_HISTORY_MESSAGES = 10


# === SYSTEM PROMPT ===

SYSTEM_PROMPT = """Jsi AI asistent pro Techmania Science Center v Plzni.

PRAVIDLA:
- Nikdy nevymýšlej ani neodhaduj data nebo čísla.
- Čísla uváděj pouze z nástrojů.
- Když data nemáš, odpověz přesně:
  "❌ Nemám k dispozici data pro tento dotaz. Zkuste to prosím znovu nebo přeformulujte otázku."
- Nikdy nepředstírej volání nástroje.
- Pokud dotaz vyžaduje data, MUSÍŠ použít nástroj.

ODPOVĚDI:
- Česky, stručně.
- Čísla z nástrojů tučně.
"""


# === POMOCNÉ FUNKCE ===

def needs_data_tools(message: str) -> bool:
    """Zjistí, zda zpráva vyžaduje použití datových nástrojů."""
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in DATA_KEYWORDS)


def extract_date_from_message(message: str) -> Optional[str]:
    """
    Extrahuje datum z textu zprávy.
    
    Returns:
        Datum ve formátu YYYY-MM-DD nebo None
    """
    # Formát: 5.12.2025, 5. 12. 2025, atd.
    date_match = re.search(r'(\d{1,2})\.?\s*(\d{1,2})\.?\s*(\d{4})', message)
    if date_match:
        day, month, year = date_match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"
    return None


def extract_year_from_message(message: str) -> Optional[int]:
    """Extrahuje rok z textu zprávy."""
    year_match = re.search(r'\b(20\d{2})\b', message)
    if year_match:
        return int(year_match.group(1))
    return None


def is_future_date(date_str: str) -> bool:
    """Zjistí, zda je datum v budoucnosti."""
    try:
        parts = date_str.split('-')
        query_date = date_cls(int(parts[0]), int(parts[1]), int(parts[2]))
        return query_date > date_cls.today()
    except (ValueError, IndexError):
        return False


def build_messages(
    message: str,
    history: Optional[List[Dict[str, str]]] = None
) -> List[Dict[str, Any]]:
    """
    Sestaví seznam zpráv pro AI včetně systémového promptu a historie.
    
    Args:
        message: Aktuální zpráva od uživatele
        history: Historie konverzace
        
    Returns:
        Seznam zpráv pro API
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Přidat historii (posledních N zpráv)
    if history:
        for h in history[-MAX_HISTORY_MESSAGES:]:
            messages.append({"role": h["role"], "content": h["content"]})
    
    # Přidat aktuální zprávu
    messages.append({"role": "user", "content": message})
    
    return messages


def format_tool_call_for_message(tool_call) -> Dict[str, Any]:
    """Formátuje tool call pro přidání do zpráv."""
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments
        }
    }


# === ERROR RESPONSES ===

class ChatResponses:
    """Předpřipravené odpovědi pro různé situace."""
    
    NO_DATA = (
        "❌ Nemám k dispozici data pro tento dotaz. "
        "Zkuste to prosím znovu nebo přeformulujte otázku."
    )
    
    EXAMPLES = (
        "\n\nPříklady dotazů:\n"
        "- \"Jaká byla návštěvnost 5.12.2025?\"\n"
        "- \"Predikce na 15.1.2026\"\n"
        "- \"Statistiky roku 2024\"\n"
        "- \"Porovnej roky 2020 a 2024\""
    )
    
    TOOL_ERROR = (
        "❌ Nepodařilo se získat data. Zkuste přeformulovat dotaz, např.:\n"
        "- \"Jaká byla návštěvnost 5. prosince 2025?\"\n"
        "- \"Jaká je predikce na 13. ledna 2026?\"\n"
        "- \"Porovnej roky 2016 a 2020\""
    )
    
    DEFAULT_GREETING = (
        "Ahoj! Jsem AI asistent pro Techmania Science Center. "
        "Zeptej se mě na návštěvnost, predikce nebo statistiky."
    )


# === HLAVNÍ CHAT HANDLER ===

class ChatHandler:
    """Handler pro zpracování chat zpráv s AI."""
    
    def __init__(self, db_session):
        """
        Inicializuje chat handler.
        
        Args:
            db_session: SQLAlchemy session pro přístup k databázi
        """
        self.db_session = db_session
        self.client = get_ai_client()
        self.tools = MCP_TOOLS
    
    def _execute_tool_calls(
        self,
        tool_calls: List,
        messages: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Vykoná všechny tool calls a přidá výsledky do zpráv.
        
        Returns:
            Seznam názvů použitých nástrojů
        """
        tools_used = []
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            
            try:
                function_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                function_args = {}
            
            # Spustit nástroj
            tool_result = execute_tool(function_name, function_args, self.db_session)
            tools_used.append(function_name)
            
            # Přidat výsledek do zpráv
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(tool_result, ensure_ascii=False, default=str)
            })
        
        return tools_used
    
    def _try_extract_and_call_tool(self, message: str) -> Optional[str]:
        """
        Zkusí extrahovat informace ze zprávy a zavolat příslušný nástroj.
        
        Returns:
            Výsledek nástroje jako string nebo None
        """
        # Zkusit extrahovat datum
        date_str = extract_date_from_message(message)
        if date_str:
            if is_future_date(date_str):
                result = execute_tool('predict_visitors', {'date': date_str}, self.db_session)
            else:
                result = execute_tool('get_visitors_by_date', {'start_date': date_str}, self.db_session)
            
            if result.get('success'):
                return f"📊 {result['data']}"
        
        # Zkusit detekovat rok pro statistiky
        year = extract_year_from_message(message)
        message_lower = message.lower()
        if year and ('průměr' in message_lower or 'statistik' in message_lower or 'rok' in message_lower):
            result = execute_tool('get_historical_stats', {'year': year}, self.db_session)
            if result.get('success'):
                return f"📊 {result['data']}"
        
        return None
    
    def _handle_tool_calls(
        self,
        response_message,
        messages: List[Dict[str, Any]]
    ) -> Generator[str, None, None]:
        """
        Zpracuje tool calls z odpovědi AI.
        
        Yields:
            Části odpovědi pro streaming
        """
        tool_calls = response_message.tool_calls
        
        # Přidat odpověď AI do historie
        messages.append({
            "role": "assistant",
            "content": response_message.content or "",
            "tool_calls": [format_tool_call_for_message(tc) for tc in tool_calls]
        })
        
        # Vykonat všechny tool calls
        self._execute_tool_calls(tool_calls, messages)
        
        # Druhé volání - AI zpracuje výsledky a odpoví (streaming)
        completion = self.client.chat_completion(
            messages=messages,
            stream=True,
        )
        
        for chunk in completion:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def _handle_no_tool_calls(
        self,
        response_message,
        original_message: str,
        requires_data: bool
    ) -> Generator[str, None, None]:
        """
        Zpracuje odpověď bez tool calls.
        
        Yields:
            Části odpovědi
        """
        if requires_data:
            # AI měla použít nástroj ale nepoužila - zkusíme sami
            extracted_result = self._try_extract_and_call_tool(original_message)
            if extracted_result:
                yield extracted_result
                return
            
            # Nic se nepodařilo
            yield ChatResponses.NO_DATA
            yield ChatResponses.EXAMPLES
        else:
            # Obecný dotaz bez potřeby dat
            if response_message.content:
                yield response_message.content
            else:
                yield ChatResponses.DEFAULT_GREETING
    
    def _handle_api_error(
        self,
        error: Exception,
        messages: List[Dict[str, Any]],
        requires_data: bool
    ) -> Generator[str, None, None]:
        """
        Zpracuje chybu z API.
        
        Yields:
            Chybová zpráva nebo fallback odpověď
        """
        error_str = str(error)
        
        if "tool_use_failed" in error_str and not requires_data:
            # Fallback bez nástrojů pro obecné dotazy
            response = self.client.chat_completion(
                messages=messages,
                temperature=0.7,
            )
            if response.choices[0].message.content:
                yield response.choices[0].message.content
        elif "tool_use_failed" in error_str:
            yield ChatResponses.TOOL_ERROR
        else:
            raise error
    
    def stream(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> Generator[str, None, None]:
        """
        Streamuje odpověď od AI s podporou MCP tools.
        
        Args:
            message: Zpráva od uživatele
            history: Historie konverzace
            
        Yields:
            Části odpovědi pro streaming
        """
        messages = build_messages(message, history)
        requires_data = needs_data_tools(message)
        
        try:
            # První volání s nástroji (s retry)
            response = self.client.chat_completion_with_retry(
                messages=messages,
                tools=self.tools,
            )
            
            response_message = response.choices[0].message
            
            if response_message.tool_calls:
                # AI chce použít nástroje
                yield from self._handle_tool_calls(response_message, messages)
            else:
                # AI nepoužila nástroje
                yield from self._handle_no_tool_calls(
                    response_message, 
                    message, 
                    requires_data
                )
                
        except ValueError as e:
            yield f"\n\n⚠️ {str(e)}"
        except Exception as e:
            # Zkusit zpracovat známé chyby
            try:
                yield from self._handle_api_error(e, messages, requires_data)
            except Exception:
                yield f"\n\n❌ Chyba: {str(e)}"


# === VEŘEJNÉ API ===

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
    handler = ChatHandler(db_session)
    yield from handler.stream(message, history)


def chat_sync(
    message: str,
    db_session,
    history: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Synchronní verze chatu (bez streamování).
    
    Args:
        message: Zpráva od uživatele
        db_session: SQLAlchemy session pro přístup k databázi
        history: Historie konverzace
        
    Returns:
        Kompletní odpověď jako string
    """
    return "".join(chat_stream(message, db_session, history))
