from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_DIR = Path(__file__).resolve().parent.parent


class Setting(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_DIR / ".env", extra="ignore")
    database_url: str = ""
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"


setting = Setting()
