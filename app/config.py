"""
App configuration using pydantic-settings (Pydantic v2+).

- Supports .env file and environment variables.
- Switch between dev and prod profiles without changing code.
"""

import os
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Central configuration.
    Priority: Environment variables > .env > defaults in this class.
    """

    # Profile selection
    model_profile: str = Field(default=os.getenv("MODEL_PROFILE", "dev"))  # "dev" | "prod"

    # Model and adapter
    base_model: Optional[str] = Field(default=os.getenv("BASE_MODEL"))
    adapter_path: Optional[str] = Field(default=os.getenv("ADAPTER_PATH"))
    model_cache: str = Field(default=os.getenv("MODEL_CACHE", "./hf_models/cache"))

    # Quantization control (heuristic by default)
    quantization: str = Field(default=os.getenv("QUANTIZATION", "auto"))  # "auto" | "none" | "4bit"

    # Generation params
    max_new_tokens: int = Field(default=int(os.getenv("MAX_NEW_TOKENS", "256")))
    temperature: float = Field(default=float(os.getenv("TEMPERATURE", "0.7")))
    top_p: float = Field(default=float(os.getenv("TOP_P", "0.9")))

    # Logging and auth
    log_level: str = Field(default=os.getenv("LOG_LEVEL", "info"))
    hf_token: Optional[str] = Field(default=os.getenv("HF_TOKEN"))

    class Config:
        # Automatically load a .env file from the project root if present
        env_file = ".env"
        extra = "ignore"

    def resolve_defaults(self) -> "Settings":
        # Choose base model by profile if not explicitly set
        if not self.base_model:
            if self.model_profile == "prod":
                self.base_model = "unsloth/llama-3-8b-bnb-4bit"
            else:
                self.base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        # Choose adapter path (prod expects a LoRA adapter by default)
        if self.adapter_path is None:
            self.adapter_path = "app/network_optimizer" if self.model_profile == "prod" else ""

        # Heuristic quantization
        if self.quantization == "auto":
            self.quantization = "4bit" if "4bit" in self.base_model.lower() else "none"

        return self


@lru_cache
def get_settings() -> Settings:
    """Return a singleton Settings instance with resolved defaults."""
    return Settings().resolve_defaults()