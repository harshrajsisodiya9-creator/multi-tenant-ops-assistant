from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_DIR = Path(__file__).resolve().parent.parent


class Setting(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_DIR / ".env", extra="ignore")
    database_url: str = ""
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"
    groq_judge_model: str = "openai/gpt-oss-120b"
    gemini_api_key: str = ""
    gemini_model2: str = "gemini-3.1-flash-lite"
    gemini_model: str = "gemini-3.5-flash-lite"


setting = Setting()
