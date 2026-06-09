from __future__ import annotations

import re
import unicodedata
from collections import Counter

import pandas as pd

try:
    from underthesea import word_tokenize
except ImportError:  # The basic cleaner remains usable without the optional NLP package.
    word_tokenize = None


STOP_WORDS = {
    "và", "của", "có", "được", "một", "các", "cho", "trong", "những",
    "với", "là", "này", "khi", "về", "đến", "từ", "đã", "sẽ", "đang",
    "để", "ra", "ở", "trên", "rất", "như", "lại", "cũng", "nên", "vì",
    "mà", "bị", "đó", "tôi", "em", "bạn", "thì", "qua", "sau", "trước",
    "tại", "theo", "vào", "hay", "hoặc", "nếu", "đây", "ấy", "nhưng",
    "vẫn", "đều", "chỉ", "cùng", "mỗi", "việc",
}

TEENCODE_MAP = {
    "ko": "không",
    "k": "không",
    "kh": "không",
    "hok": "học",
    "hk": "học",
    "dc": "được",
    "dk": "được",
    "đc": "được",
    "đk": "được",
    "vs": "với",
    "j": "gì",
    "z": "vậy",
    "r": "rồi",
    "ntn": "như thế nào",
}

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
NON_TEXT_RE = re.compile(r"[^0-9A-Za-zÀ-ỹĐđ\s]", re.UNICODE)
SPACE_RE = re.compile(r"\s+")


def clean_text(text: object, do_segment: bool = True) -> str:
    if not isinstance(text, str):
        return ""

    value = unicodedata.normalize("NFC", text).strip().lower()
    value = URL_RE.sub(" ", value)
    value = MENTION_RE.sub(" ", value)
    value = NON_TEXT_RE.sub(" ", value)
    words = [TEENCODE_MAP.get(word, word) for word in SPACE_RE.sub(" ", value).split()]
    value = " ".join(word for word in words if word not in STOP_WORDS)

    if do_segment and value and word_tokenize is not None:
        try:
            value = word_tokenize(value, format="text")
        except Exception:
            pass
    return SPACE_RE.sub(" ", value).strip()


def extract_keywords(text: str, limit: int = 6) -> list[str]:
    tokens = [
        token.replace("_", " ")
        for token in clean_text(text, do_segment=False).split()
        if len(token) > 1 and token not in STOP_WORDS
    ]
    return [token for token, _ in Counter(tokens).most_common(limit)]


def batch_clean(
    df: pd.DataFrame,
    text_col: str = "sentence",
    do_segment: bool = True,
) -> pd.DataFrame:
    output = df.copy()
    output["processed_text"] = output[text_col].apply(
        lambda value: clean_text(value, do_segment=do_segment)
    )
    return output
