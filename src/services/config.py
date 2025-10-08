"""
Configuration management for the RAG Agent application.
"""
import os
from typing import Optional
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Keys
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    openweather_api_key: Optional[str] = Field(None, env="OPENWEATHER_API_KEY")
    langsmith_api_key: Optional[str] = Field(None, env="LANGSMITH_API_KEY")
    
    # Qdrant Configuration
    qdrant_url: str = Field("http://localhost:6333", env="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(None, env="QDRANT_API_KEY")
    
    # LangSmith Configuration
    langsmith_project: str = Field("RAG_Agent_Project", env="LANGSMITH_PROJECT")
    langsmith_endpoint: str = Field("https://api.smith.langchain.com", env="LANGSMITH_ENDPOINT")
    
    # Application Configuration
    debug: bool = Field(False, env="DEBUG")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    # Local LLM Configuration
    local_model_path: str = Field("model", env="LOCAL_MODEL_PATH")
    local_model_device: str = Field("auto", env="LOCAL_MODEL_DEVICE")
    local_model_max_new_tokens: int = Field(512, env="LOCAL_MODEL_MAX_NEW_TOKENS")

    # # Ollama Configuration
    # ollama_base_url: str = Field("http://localhost:11434", env="OLLAMA_BASE_URL")
    # ollama_model: str = Field("llama3", env="OLLAMA_MODEL")

    # Groq (OpenAI-compatible) Configuration
    groq_api_key: Optional[str] = Field(None, env="GROQ_API_KEY")
    groq_base_url: str = Field("https://api.groq.com/openai/v1", env="GROQ_BASE_URL")
    groq_model: str = Field("openai/gpt-oss-20b", env="GROQ_MODEL")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

def get_settings() -> Settings:
    """Get application settings."""
    return settings
