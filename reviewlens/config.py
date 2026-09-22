"""Environment-backed application settings."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_embed_model: str = "gemini-embedding-2"

    demo_mode: bool = False
    db_path: Path = REPO_ROOT / "data" / "reviewlens.db"
    demo_db_path: Path = REPO_ROOT / "data" / "sample" / "demo.db"
    chroma_dir: Path = REPO_ROOT / "chroma"

    @property
    def effective_db_path(self) -> Path:
        """The demo snapshot in DEMO_MODE (no Gemini calls needed to browse it), else the live DB."""
        return self.demo_db_path if self.demo_mode else self.db_path


settings = Settings()
