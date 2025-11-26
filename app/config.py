from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Telegram Bot Configuration
    telegram_bot_token: str

    # Model Configuration
    model_path: Path = Path("models/best_model.pt")
    label_map_path: Path = Path("models/label_map.json")
    confidence_threshold: float = 0.7

    # API Configuration
    use_api_service: bool = False
    api_url: str = "http://localhost:8000"
    api_timeout: int = 30

    # Paths
    temp_dir: Path = Path("./temp")
    data_dir: Path = Path("./data")

    # Logging
    log_level: str = "INFO"
    log_file: Path = Path("logs/bot.log")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create necessary directories
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
