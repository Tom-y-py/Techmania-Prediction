"""
AI Client modul pro Techmania Prediction API.
Spravuje připojení k AI službám (Groq/OpenAI) a poskytuje jednotné rozhraní.
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Generator
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Přidat parent složku do path pro import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config


@dataclass
class ChatMessage:
    """Reprezentace zprávy v chatu."""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertuje zprávu na slovník pro API."""
        result = {"role": self.role, "content": self.content}
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result


class AIClient(ABC):
    """Abstraktní třída pro AI klienty."""
    
    @abstractmethod
    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
    ) -> Any:
        """Vytvoří chat completion."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Zkontroluje dostupnost klienta."""
        pass


class GroqClient(AIClient):
    """Groq AI klient."""
    
    def __init__(self):
        self._client = None
        self._config = None
    
    def _get_client(self):
        """Lazy loading Groq klienta."""
        if self._client is None:
            self._config = config.ai  # Může vyhodit ValueError pokud chybí API key
            from groq import Groq
            self._client = Groq(api_key=self._config.api_key)
        return self._client
    
    @property
    def model(self) -> str:
        """Vrátí název modelu."""
        if self._config is None:
            self._config = config.ai
        return self._config.model
    
    @property
    def default_temperature(self) -> float:
        """Vrátí výchozí teplotu."""
        if self._config is None:
            self._config = config.ai
        return self._config.temperature
    
    @property
    def stream_temperature(self) -> float:
        """Vrátí teplotu pro streaming."""
        if self._config is None:
            self._config = config.ai
        return self._config.stream_temperature
    
    @property
    def max_tokens(self) -> int:
        """Vrátí maximální počet tokenů."""
        if self._config is None:
            self._config = config.ai
        return self._config.max_tokens
    
    @property
    def max_retries(self) -> int:
        """Vrátí maximální počet opakování."""
        if self._config is None:
            self._config = config.ai
        return self._config.max_retries
    
    def is_available(self) -> bool:
        """Zkontroluje dostupnost klienta."""
        try:
            self._get_client()
            return True
        except (ValueError, ImportError):
            return False
    
    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
    ) -> Any:
        """
        Vytvoří chat completion pomocí Groq API.
        
        Args:
            messages: Seznam zpráv
            tools: Seznam nástrojů pro function calling
            stream: Zda streamovat odpověď
            temperature: Teplota (volitelné, použije se výchozí)
            
        Returns:
            Response objekt z Groq API
        """
        client = self._get_client()
        
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }
        
        # Nastavit teplotu
        if temperature is not None:
            kwargs["temperature"] = temperature
        elif stream:
            kwargs["temperature"] = self.stream_temperature
        else:
            kwargs["temperature"] = self.default_temperature
        
        # Přidat nástroje pokud jsou k dispozici
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        
        return client.chat.completions.create(**kwargs)
    
    def chat_completion_with_retry(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        temperature: Optional[float] = None,
    ) -> Any:
        """
        Chat completion s opakováním při selhání.
        
        Returns:
            Tuple[response, bool] - response a příznak zda byly použity nástroje
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                response = self.chat_completion(
                    messages=messages,
                    tools=tools,
                    stream=False,
                    temperature=temperature,
                )
                return response
            except Exception as e:
                last_error = e
                error_str = str(e)
                
                # Pokud jde o tool_use_failed nebo 400 error, zkusit znovu
                if attempt < self.max_retries - 1:
                    if "tool_use_failed" in error_str or "400" in error_str:
                        continue
                
                raise
        
        raise last_error


class AIClientFactory:
    """Továrna pro vytváření AI klientů."""
    
    _instance: Optional[GroqClient] = None
    
    @classmethod
    def get_client(cls, provider: str = "groq") -> AIClient:
        """
        Vrátí AI klienta pro daného poskytovatele.
        
        Args:
            provider: Poskytovatel AI služby ("groq", "openai", ...)
            
        Returns:
            Instance AI klienta
        """
        if provider == "groq":
            if cls._instance is None:
                cls._instance = GroqClient()
            return cls._instance
        else:
            raise ValueError(f"Nepodporovaný AI provider: {provider}")
    
    @classmethod
    def reset(cls) -> None:
        """Resetuje singleton instance (pro testování)."""
        cls._instance = None


# Pomocná funkce pro snadný přístup
def get_ai_client() -> AIClient:
    """Vrátí výchozího AI klienta."""
    return AIClientFactory.get_client(config.ai.provider)
