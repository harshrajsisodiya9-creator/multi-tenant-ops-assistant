from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env", override=False)


class Setting(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_DIR / ".env", extra="ignore")
    database_url: str = ""
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"
    groq_judge_model: str = "openai/gpt-oss-120b"
    gemini_api_key: str = ""
    gemini_model2: str = "gemini-3.1-flash-lite"
    gemini_model: str = "gemini-3.5-flash-lite"
    app_env: str = "dev"
    agent_mode: str = "mock"


def _check_factory_guard(s: Setting) -> None:
    if s.app_env == "prod" and s.agent_mode != "prod":
        raise RuntimeError(
            f"APP_ENV=prod requires AGENT_MODE=prod (got '{s.agent_mode}'). "
            "Mock data must never serve production traffic."
        )


setting = Setting()
_check_factory_guard(setting)
