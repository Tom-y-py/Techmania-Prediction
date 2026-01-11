"""
Chat modul pro Techmania Prediction API.
Poskytuje AI chat s přístupem k datům o návštěvnosti.
"""

from .handler import chat_stream, chat_sync, ChatHandler
from .client import get_ai_client, AIClient, AIClientFactory
from .tools import MCP_TOOLS, execute_tool

__all__ = [
    # Hlavní API
    "chat_stream",
    "chat_sync",
    "ChatHandler",
    # AI klient
    "get_ai_client",
    "AIClient", 
    "AIClientFactory",
    # Nástroje
    "MCP_TOOLS",
    "execute_tool",
]
