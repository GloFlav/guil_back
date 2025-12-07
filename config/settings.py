# backend/config/settings.py
"""
Configuration centralisée pour l'application Survey Generator Madagascar
Gère les variables d'environnement et les paramètres globaux
"""

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

from pydantic import Field, field_validator
from typing import List, Union, Dict, Any
import os

class Settings(BaseSettings):
    """Configuration pour l'API Survey Generator avec multi-clés"""
    
    # ==================== OpenAI Configuration ====================
    openai_api_key_1: str = Field(default="", alias="OPENAI_API_KEY_1")
    openai_api_key_2: str = Field(default="", alias="OPENAI_API_KEY_2")
    openai_model: str = Field(default="gpt-4-turbo", alias="OPENAI_MODEL")
    
    # ==================== Anthropic Configuration ====================
    anthropic_api_key_1: str = Field(default="", alias="ANTHROPIC_API_KEY_1")
    anthropic_api_key_2: str = Field(default="", alias="ANTHROPIC_API_KEY_2")
    anthropic_model: str = Field(default="claude-sonnet-4-5-20250929", alias="ANTHROPIC_MODEL")
    
    # ==================== Gemini Configuration ====================
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-pro", alias="GEMINI_MODEL")
    
    # ==================== Backup LLM Configuration ====================
    backup_provider: str = Field(default="openai", alias="BACKUP_PROVIDER")
    
    # ==================== Server Configuration ====================
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    debug: bool = Field(default=True, alias="DEBUG")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    
    # ==================== CORS Configuration ====================
    cors_origins: Union[List[str], str] = Field(
        default=["http://localhost:3000", "http://localhost:3001", "http://localhost:5173"],
        alias="CORS_ORIGINS"
    )
    
    # ==================== File Storage ====================
    excel_output_dir: str = Field(default="./exports", alias="EXCEL_OUTPUT_DIR")
    max_file_size_mb: int = Field(default=50, alias="MAX_FILE_SIZE_MB")
    
    # ==================== Logging ====================
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_file: str = Field(default="./logs/app.log", alias="LOG_FILE")
    
    # ==================== Data Files ====================
    mdg_data_file: str = Field(default="./data/mdg_adm3.csv", alias="MDG_DATA_FILE")
    
    # ==================== Google Maps ====================
    google_maps_api_key: str = Field(default="", alias="GOOGLE_MAPS_API_KEY")
    
    # ==================== Generation Parameters ====================
    min_questions: int = Field(default=24, alias="MIN_QUESTIONS")
    max_questions: int = Field(default=60, alias="MAX_QUESTIONS")
    default_num_categories: int = Field(default=6, alias="DEFAULT_NUM_CATEGORIES")
    max_generation_retries: int = Field(default=3, alias="MAX_GENERATION_RETRIES")
    default_num_locations: int = Field(default=5, alias="DEFAULT_NUM_LOCATIONS")
    max_num_locations: int = Field(default=20, alias="MAX_NUM_LOCATIONS")
    
    # ==================== API Timeouts ====================
    llm_timeout_seconds: int = Field(default=120, alias="LLM_TIMEOUT_SECONDS")
    websocket_timeout_seconds: int = Field(default=300, alias="WEBSOCKET_TIMEOUT_SECONDS")
    
    @field_validator('cors_origins', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS_ORIGINS depuis string ou list"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',') if origin.strip()]
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = False
        extra = "allow"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Créer les dossiers s'ils n'existent pas
        os.makedirs(self.excel_output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_file) or ".", exist_ok=True)
    
    # ==================== API Keys Management ====================
    
    def get_openai_keys(self) -> List[str]:
        """Retourne les clés OpenAI disponibles (ordre: key_1, key_2)"""
        keys = []
        if self.openai_api_key_1:
            keys.append(self.openai_api_key_1)
        if self.openai_api_key_2:
            keys.append(self.openai_api_key_2)
        return keys
    
    def get_anthropic_keys(self) -> List[str]:
        """Retourne les clés Anthropic disponibles (ordre: key_1, key_2)"""
        keys = []
        if self.anthropic_api_key_1:
            keys.append(self.anthropic_api_key_1)
        if self.anthropic_api_key_2:
            keys.append(self.anthropic_api_key_2)
        return keys
    
    def get_gemini_keys(self) -> List[str]:
        """Retourne les clés Gemini disponibles"""
        keys = []
        if self.gemini_api_key:
            keys.append(self.gemini_api_key)
        return keys
    
    def get_all_api_keys(self) -> Dict[str, List[str]]:
        """Retourne toutes les clés API organisées par provider"""
        return {
            "openai": self.get_openai_keys(),
            "anthropic": self.get_anthropic_keys(),
            "gemini": self.get_gemini_keys()
        }
    
    def validate_api_keys(self) -> Dict[str, Dict[str, Any]]:
        """Valide la disponibilité des clés API"""
        return {
            "openai": {
                "available": len(self.get_openai_keys()) > 0,
                "count": len(self.get_openai_keys()),
                "model": self.openai_model
            },
            "anthropic": {
                "available": len(self.get_anthropic_keys()) > 0,
                "count": len(self.get_anthropic_keys()),
                "model": self.anthropic_model
            },
            "gemini": {
                "available": len(self.get_gemini_keys()) > 0,
                "count": len(self.get_gemini_keys()),
                "model": self.gemini_model
            }
        }
    
    def has_google_maps(self) -> bool:
        """Vérifie si la clé Google Maps est configurée"""
        return bool(self.google_maps_api_key)

# Instance globale des paramètres
settings = Settings()