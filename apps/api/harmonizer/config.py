from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # .env lives at the repo root; commands run from apps/api via ./hz
    model_config = SettingsConfigDict(env_file=(".env", "../../.env"), extra="ignore")

    database_url: str = "postgresql://harmonizer:harmonizer@localhost:5434/harmonizer"
    ollama_host: str = "http://localhost:11434"


settings = Settings()
