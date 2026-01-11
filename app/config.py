"""
Centrální konfigurace pro Techmania Prediction API.
Všechna nastavení na jednom místě.
"""

import os
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Načíst proměnné prostředí
load_dotenv()


@dataclass(frozen=True)
class AIConfig:
    """Konfigurace pro AI model."""
    provider: str = "groq"
    model: str = "llama-3.1-8b-instant"
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    temperature: float = 0.3
    max_tokens: int = 2048
    stream_temperature: float = 0.5
    max_retries: int = 3
    
    def __post_init__(self):
        if not self.api_key:
            raise ValueError("GROQ_API_KEY není nastaven. Přidejte ho do .env souboru.")


@dataclass(frozen=True)
class ServerConfig:
    """Konfigurace serveru."""
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "5000")))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "true").lower() == "true")
    
    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@dataclass(frozen=True)
class APIConfig:
    """Konfigurace API."""
    title: str = field(default_factory=lambda: os.getenv("API_TITLE", "Techmania Prediction API"))
    version: str = field(default_factory=lambda: os.getenv("API_VERSION", "2.0.0"))
    cors_origins: List[str] = field(
        default_factory=lambda: os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    )


@dataclass
class PathConfig:
    """Konfigurace cest."""
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    
    @property
    def base_dir(self) -> Path:
        if self.environment == "production":
            return Path("/app")
        return Path(__file__).parent.parent
    
    @property
    def models_dir(self) -> Path:
        return self.base_dir / "models"
    
    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data" / "raw"
    
    @property
    def processed_data_dir(self) -> Path:
        return self.base_dir / "data" / "processed"


class Config:
    """Hlavní konfigurační třída - singleton."""
    
    _instance: Optional["Config"] = None
    
    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self) -> None:
        """Inicializuje konfiguraci."""
        self._server = ServerConfig()
        self._api = APIConfig()
        self._paths = PathConfig(environment=self._server.environment)
        self._ai: Optional[AIConfig] = None
    
    @property
    def server(self) -> ServerConfig:
        return self._server
    
    @property
    def api(self) -> APIConfig:
        return self._api
    
    @property
    def paths(self) -> PathConfig:
        return self._paths
    
    @property
    def ai(self) -> AIConfig:
        """Lazy loading AI konfigurace - vyhodí chybu až když je potřeba."""
        if self._ai is None:
            self._ai = AIConfig()
        return self._ai
    
    def print_info(self) -> None:
        """Vypíše informace o konfiguraci."""
        print(f"🔧 Prostředí: {self.server.environment}")
        print(f"📁 Adresář modelů: {self.paths.models_dir}")
        print(f"📁 Adresář dat: {self.paths.data_dir}")
        print(f"🌐 CORS origins: {self.api.cors_origins}")


# Globální instance
config = Config()


# Pro zpětnou kompatibilitu - exportovat jednotlivé hodnoty
ENVIRONMENT = config.server.environment
HOST = config.server.host
PORT = config.server.port
DEBUG = config.server.debug
CORS_ORIGINS = config.api.cors_origins
API_TITLE = config.api.title
API_VERSION = config.api.version
BASE_DIR = config.paths.base_dir
MODELS_DIR = config.paths.models_dir
DATA_DIR = config.paths.data_dir
