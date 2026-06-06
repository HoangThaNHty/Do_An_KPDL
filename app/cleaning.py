from __future__ import annotations

import re
import string
from pathlib import Path

import pandas as pd
from underthesea import word_tokenize

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

STOP_WORDS: set[str] = {
    "và", "của", "có", "được", "một", "các", "cho", "trong",
    "những", "với", "là", "này", "khi", "về", "đến",
    "từ", "đã", "sẽ", "đang", "để", "ra", "ở", "trên",
    "rất", "như", "lại", "cũng", "nên", "vì", "mà", "bị",
    "đó", "tôi", "em", "bạn", "thì", "qua", "sau", "trước",
    "tại", "theo", "vào", "hay", "hoặc", "nếu", "thì", "đây",
    "đó", "ấy", "nhưng", "vẫn", "đều", "chỉ", "cùng", "mỗi",
    "nơi", "việc", "người",
}


def remove_emoji(text: str) -> str:
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002600-\U000026FF"
        "\U0000FE00-\U0000FE0F"
        "\U00002000-\U0000206F"
        "\U00002100-\U0000214F"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)


def remove_urls(text: str) -> str:
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    return url_pattern.sub("", text)


def remove_mentions(text: str) -> str:
    return re.sub(r"@\w+", "", text)


def remove_special_chars(text: str) -> str:
    allowed = string.ascii_letters + string.digits + string.whitespace + "ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯăâêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ"
    return "".join(c if c in allowed else " " for c in text)


def normalize_unicode(text: str) -> str:
    replacements = {
        "òa": "oà", "óa": "oá", "ỏa": "oả", "õa": "oã", "ọa": "oạ",
        "òe": "oè", "óe": "oé", "ỏe": "oẻ", "õe": "oẽ", "ọe": "oẹ",
        "ùy": "uỳ", "úy": "uý", "ủy": "uỷ", "ũy": "uỹ", "ụy": "uỵ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def fix_teencode(text: str) -> str:
    teencode_map = {
        "ko": "không", "k": "không", "kh": "không",
        "hok": "học", "hk": "học",
        "dc": "được", "dk": "được", "đc": "được", "đk": "được",
        "vs": "với", "va": "và",
        "j": "gì", "z": "vậy", "r": "rồi",
        "m": "mình", "t": "tôi", "b": "bạn",
        "ng": "người", "hp": "hạnh phúc", "bt": "bình thường",
        "nk": "nên", "ntn": "như thế nào",
        "ah": "à", "eh": "è", "oh": "ồ",
        "iu": "yêu", "iu": "yêu",
        "ck": "chắc", "cj": "chị",
        "iu": "yêu", "iu": "yêu",
        "ms": "mất", "ms": "mất",
        "ntn": "như thế nào",
        "đc": "được",
    }
    words = text.split()
    result = []
    for word in words:
        clean = word.strip(string.punctuation).lower()
        punct = word[len(word.rstrip(string.punctuation)):] if word.rstrip(string.punctuation) else ""
        prefix = word[:len(word) - len(word.lstrip(string.punctuation))] if word.lstrip(string.punctuation) else ""
        core = word.strip(string.punctuation)
        if core.lower() in teencode_map:
            replacement = teencode_map[core.lower()]
            if core.isupper():
                replacement = replacement.upper()
            elif core[0].isupper() if core else False:
                replacement = replacement[0].upper() + replacement[1:]
            result.append(prefix + replacement + punct)
        else:
            result.append(word)
    return " ".join(result)


def remove_stopwords(text: str) -> str:
    words = text.split()
    return " ".join(w for w in words if w.lower() not in STOP_WORDS)


def segment_words(text: str) -> str:
    try:
        tokens = word_tokenize(text)
        return " ".join(tokens)
    except Exception:
        return text


def clean_text(text: str, do_segment: bool = True) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = remove_urls(text)
    text = remove_mentions(text)
    text = remove_emoji(text)
    text = remove_special_chars(text)
    text = normalize_unicode(text)

    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()

    text = fix_teencode(text)
    text = remove_stopwords(text)

    if do_segment:
        text = segment_words(text)

    return text.strip()


def batch_clean(
    df: pd.DataFrame,
    text_col: str = "text",
    do_segment: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    out["clean_text"] = out[text_col].apply(lambda t: clean_text(t, do_segment=do_segment))
    return out


if __name__ == "__main__":
    noisy = pd.read_parquet(DATA_DIR / "uit_vsfc_noisy.parquet")
    noisy_sample = noisy[noisy["is_noisy"]].head(3)

    print("=== Before cleaning ===")
    for _, row in noisy_sample.iterrows():
        print(f"  [{row['sentiment']}] {row['text'][:150]}")

    cleaned = batch_clean(noisy_sample, do_segment=False)
    print("\n=== After cleaning (no segmentation) ===")
    for _, row in cleaned.iterrows():
        print(f"  [{row['sentiment']}] {row['clean_text'][:150]}")

    segmented = batch_clean(noisy_sample, do_segment=True)
    print("\n=== After cleaning (with word segmentation) ===")
    for _, row in segmented.iterrows():
        print(f"  [{row['sentiment']}] {row['clean_text'][:150]}")
