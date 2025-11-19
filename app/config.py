from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="TTS_SERVICE_", extra="ignore")

    model_id: str = "ResembleAI/chatterbox"
    model_variant: Literal["base", "multilingual"] = "base"
    device: Literal["cuda", "cpu"] = "cuda"
    output_sample_rate: int = 16000
    chunk_duration_ms: int = 200
    max_text_chars: int = 2000
    default_language_id: str = "en"
    default_cfg_scale: float = 0.5
    default_exaggeration: float = 0.5
    default_temperature: float = 1.0
    default_top_p: float = 0.9
    default_min_p: float = 0.05
    default_repetition_penalty: float = 1.2
    default_audio_prompt_path: Optional[str] = None
    auth_token: Optional[str] = None
    metrics_endpoint_enabled: bool = True
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
