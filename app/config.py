from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


def _resolve_path(value: str, default: str) -> Path:
    path = Path(value or default)
    return path if path.is_absolute() else ROOT_DIR / path


@dataclass(frozen=True)
class Settings:
    mongo_uri: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    mongo_db: str = os.getenv("MONGO_DB", "student_feedback_db")
    feedback_collection: str = os.getenv("FEEDBACK_COLLECTION", "feedbacks")
    model_dir: Path = _resolve_path(os.getenv("MODEL_DIR", "models"), "models")
    data_dir: Path = _resolve_path(os.getenv("DATA_DIR", "data"), "data")
    mongo_timeout_ms: int = int(os.getenv("MONGO_TIMEOUT_MS", "2000"))


settings = Settings()
