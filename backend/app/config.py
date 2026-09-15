from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:0.5b"
    ollama_request_timeout_s: float = 60.0

    # RAG
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_path: str = "./.chroma"
    kb_dir: str = "./data/kb"
    top_k: int = 4

    # Intent classifier
    intent_model_dir: str = "./app/classifier/model"
    intent_base_model: str = "distilbert-base-uncased"

    # RL bandit
    bandit_state_path: str = "./data/bandit_state.json"
    bandit_alpha: float = 1.0

    # Storage
    db_path: str = "./data/copilot.db"

    # API
    cors_origins: str = "http://localhost:8080,http://localhost:5173"
    log_level: str = "info"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
