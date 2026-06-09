from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from app.cleaning import clean_text, extract_keywords
from app.config import settings


class ModelNotReadyError(RuntimeError):
    pass


class ModelService:
    def __init__(self, model_dir: Path | None = None) -> None:
        self.model_dir = model_dir or settings.model_dir
        self.sentiment_model: Any | None = None
        self.topic_model: Any | None = None
        self.warning: str | None = None

    @property
    def ready(self) -> bool:
        return self.sentiment_model is not None and self.topic_model is not None

    def load(self) -> None:
        self.sentiment_model = self._load_pickle("sentiment_model.pkl")
        self.topic_model = self._load_pickle("topic_model.pkl")
        if self.ready:
            self.warning = None
        else:
            self.warning = (
                "Chưa huấn luyện đủ mô hình, vui lòng chạy "
                "python scripts/train_models.py"
            )

    def _load_pickle(self, filename: str) -> Any | None:
        path = self.model_dir / filename
        if not path.exists():
            return None
        try:
            with path.open("rb") as file:
                return pickle.load(file)
        except Exception as exc:
            self.warning = f"Không thể tải {filename}: {exc}"
            return None

    @staticmethod
    def _predict(model: Any, processed_text: str) -> dict[str, Any]:
        label = str(model.predict([processed_text])[0])
        confidence = 1.0
        probabilities: dict[str, float] = {}
        if hasattr(model, "predict_proba"):
            values = model.predict_proba([processed_text])[0]
            probabilities = {
                str(name): round(float(value), 4)
                for name, value in zip(model.classes_, values)
            }
            confidence = float(max(values))
        return {
            "label": label,
            "confidence": round(confidence, 4),
            "probabilities": probabilities,
        }

    def predict(self, text: str) -> dict[str, Any]:
        if not self.ready:
            raise ModelNotReadyError(
                self.warning
                or "Chưa huấn luyện mô hình, vui lòng chạy scripts/train_models.py"
            )
        processed_text = clean_text(text, do_segment=True)
        if not processed_text:
            raise ValueError("Nội dung không còn từ hợp lệ sau khi tiền xử lý")
        return {
            "input": text,
            "processed_text": processed_text,
            "keywords": extract_keywords(text),
            "sentiment": self._predict(self.sentiment_model, processed_text),
            "topic": self._predict(self.topic_model, processed_text),
        }


model_service = ModelService()
