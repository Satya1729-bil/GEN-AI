"""App settings, read from environment variables (and a local .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    index_dir: Path
    embed_backend: str
    gemini_embed_model: str
    embed_dimension: int
    embed_model: str
    gemini_api_key: str | None
    gemini_model: str
    top_k: int
    min_score: float
    chunk_size: int
    chunk_overlap: int
    allow_upload: bool
    app_password: str | None
    max_upload_mb: int


def get_settings() -> Settings:
    """Read settings fresh from the environment each time."""
    return Settings(
        data_dir=Path(os.getenv("RX_DATA_DIR", "data")),
        index_dir=Path(os.getenv("RX_INDEX_DIR", "index")),
        embed_backend=os.getenv("RX_EMBED_BACKEND", "gemini"),
        gemini_embed_model=os.getenv("RX_GEMINI_EMBED_MODEL", "gemini-embedding-2"),
        embed_dimension=int(os.getenv("RX_EMBED_DIMENSION", "768")),
        embed_model=os.getenv("RX_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        gemini_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or None,
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        top_k=int(os.getenv("RX_TOP_K", "5")),
        min_score=float(os.getenv("RX_MIN_SCORE", "0.20")),
        chunk_size=int(os.getenv("RX_CHUNK_SIZE", "1200")),
        chunk_overlap=int(os.getenv("RX_CHUNK_OVERLAP", "150")),
        allow_upload=_bool("RX_ALLOW_UPLOAD", True),
        app_password=os.getenv("APP_PASSWORD") or None,
        max_upload_mb=int(os.getenv("RX_MAX_UPLOAD_MB", "25")),
    )
