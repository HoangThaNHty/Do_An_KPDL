from __future__ import annotations

import re
import unicodedata
from collections import Counter

import pandas as pd

try:
    from underthesea import word_tokenize
except ImportError:  # The basic cleaner remains usable without the optional NLP package.
    word_tokenize = None


# Cường hóa STOP_WORDS bằng cách giữ lại các từ biểu cảm tình thái quan trọng: "được", "bị", "rất", "nhưng"
STOP_WORDS = {
    "và", "của", "có", "một", "các", "cho", "trong", "những",
    "với", "là", "này", "khi", "về", "đến", "từ", "đã", "sẽ", "đang",
    "để", "ra", "ở", "trên", "như", "lại", "cũng", "nên", "vì",
    "mà", "đó", "tôi", "em", "bạn", "thì", "qua", "sau", "trước",
    "tại", "theo", "vào", "hay", "hoặc", "nếu", "đây", "ấy",
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
    "chả": "không",
    "chẳng": "không",
}

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
NON_TEXT_RE = re.compile(r"[^0-9A-Za-zÀ-ỹĐđ\s]", re.UNICODE)
SPACE_RE = re.compile(r"\s+")


def clean_text(text: object, do_segment: bool = True) -> str:
    if not isinstance(text, str):
        return ""

    # Chuẩn hóa Unicode và loại nhiễu cơ bản
    value = unicodedata.normalize("NFC", text).strip().lower()
    value = URL_RE.sub(" ", value)
    value = MENTION_RE.sub(" ", value)
    value = NON_TEXT_RE.sub(" ", value)

    # 1. Map teencode trước để giữ tính nhất quán của văn bản tiếng Việt
    words = [TEENCODE_MAP.get(word, word) for word in SPACE_RE.sub(" ", value).split()]
    value = " ".join(words)

    # 2. Tách từ tiếng Việt trước (giữ ngữ cảnh đầy đủ cho underthesea)
    if do_segment and value and word_tokenize is not None:
        try:
            value = word_tokenize(value, format="text")
        except Exception:
            pass

    # 3. Lọc bỏ từ dừng sau khi đã tách từ (so sánh cả cụm từ có gạch dưới và từ đơn)
    tokens = value.split()
    filtered_tokens = []
    for token in tokens:
        clean_tok = token.replace("_", " ")
        if clean_tok not in STOP_WORDS and token not in STOP_WORDS:
            filtered_tokens.append(token)

    return " ".join(filtered_tokens)


def extract_keywords(text: str, limit: int = 6) -> list[str]:
    # Trích xuất từ khóa trên văn bản không phân tách từ (tách khoảng trắng)
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
