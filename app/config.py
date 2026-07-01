"""Centralized configuration, loaded from environment / .env.

Everything has a safe default so the service boots even with an empty
environment. Secrets (API keys) default to empty strings, which the LLM
layer interprets as "provider unavailable".
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = parent of the `app` package.
ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM providers
    llm_provider_priority: str = "gemini,groq,openai"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    groq_api_key: str = ""
    # llama-4-scout gives 30K TPM on Groq's free tier (vs 12K for llama-3.3-70b),
    # which comfortably fits a full 8-turn conversation without rate-limiting.
    groq_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

    # Retrieval
    retrieval_mode: str = "auto"  # auto | hybrid | lexical
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hybrid_semantic_weight: float = 0.55
    retrieval_top_k: int = 40
    # How many retrieved candidates are shown to the LLM per turn. Lower = fewer
    # tokens (useful on token-per-minute limited free tiers).
    llm_candidate_limit: int = 12

    # Agent behavior
    max_turns: int = 8
    request_budget_seconds: float = 25.0
    max_recommendations: int = 10

    # Data
    catalog_path: str = "data/shl_product_catalog.json"
    catalog_url: str = (
        "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
    )

    @property
    def catalog_file(self) -> Path:
        p = Path(self.catalog_path)
        return p if p.is_absolute() else ROOT_DIR / p

    @property
    def providers(self) -> list[str]:
        return [p.strip().lower() for p in self.llm_provider_priority.split(",") if p.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
