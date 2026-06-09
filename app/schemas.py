from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


SentimentLabel = Literal["positive", "neutral", "negative"]
TopicLabel = Literal["lecturer", "training_program", "facility", "others"]
SplitLabel = Literal["train", "validation", "test", "manual"]


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Nội dung phản hồi không được để trống")
        return value


class FeedbackCreate(BaseModel):
    sentence: str = Field(min_length=1, max_length=5000)
    sentiment_label: SentimentLabel | None = None
    topic_label: TopicLabel | None = None
    source: str = Field(default="manual", max_length=100)
    split: SplitLabel = "manual"

    @field_validator("sentence", "source")
    @classmethod
    def strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Giá trị không được để trống")
        return value


class FeedbackUpdate(BaseModel):
    sentence: str | None = Field(default=None, min_length=1, max_length=5000)
    sentiment_label: SentimentLabel | None = None
    topic_label: TopicLabel | None = None
    source: str | None = Field(default=None, max_length=100)
    split: SplitLabel | None = None

    @field_validator("sentence", "source")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("Giá trị không được để trống")
        return value
