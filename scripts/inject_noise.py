from __future__ import annotations

import random
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

random.seed(42)

EMOJIS = [
    "😊", "❤️", "👍", "😂", "🔥", "😍", "🎉", "💯",
    "😢", "😡", "🤔", "😴", "🙏", "💪", "😅", "🥺",
    "😤", "😭", "✨", "⭐", "💕", "😁", "😎", "🤗",
]

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
    "va": "và",
    "j": "gì",
    "z": "vậy",
    "r": "rồi",
    "m": "mình",
    "t": "tôi",
    "b": "bạn",
    "ng": "người",
    "hp": "hạnh phúc",
    "bt": "bình thường",
    "nk": "nên",
    "ntn": "như thế nào",
    "ah": "à",
    "eh": "è",
    "oh": "ồ",
}

SPECIAL_CHARS = ["!!!", "???", "...", "@@@", "###", "***", "~~~", "^^^"]


def inject_emoji(text: str) -> str:
    n = random.randint(1, 3)
    emojis = " ".join(random.choices(EMOJIS, k=n))
    if random.random() < 0.5:
        return f"{emojis} {text}"
    return f"{text} {emojis}"


def inject_uppercase(text: str) -> str:
    words = text.split()
    result = []
    for word in words:
        if random.random() < 0.4:
            mode = random.choice(["all", "first", "random"])
            if mode == "all":
                result.append(word.upper())
            elif mode == "first":
                result.append(word[0].upper() + word[1:] if len(word) > 1 else word.upper())
            else:
                result.append("".join(c.upper() if random.random() < 0.5 else c for c in word))
        else:
            result.append(word)
    return " ".join(result)


def inject_teencode(text: str) -> str:
    words = text.lower().split()
    result = []
    for word in words:
        clean_word = re.sub(r"[^\w]", "", word)
        if clean_word in TEENCODE_MAP and random.random() < 0.7:
            punct = word[len(clean_word):] if len(word) > len(clean_word) else ""
            teencode_word = random.choice(
                [k for k in TEENCODE_MAP if TEENCODE_MAP[k] == TEENCODE_MAP[clean_word]]
            )
            result.append(teencode_word + punct)
        else:
            result.append(word)
    return " ".join(result)


def inject_special_chars(text: str) -> str:
    if random.random() < 0.3:
        idx = random.randint(0, len(text))
        sc = random.choice(SPECIAL_CHARS)
        text = text[:idx] + sc + text[idx:]
    return text


def inject_whitespace(text: str) -> str:
    words = text.split()
    result = []
    for word in words:
        result.append(word)
        if random.random() < 0.2:
            result.append(" " * random.randint(2, 5))
    return " ".join(result).strip()


def inject_noise(text: str) -> str:
    text = inject_emoji(text)
    text = inject_uppercase(text)
    text = inject_teencode(text)
    text = inject_special_chars(text)
    text = inject_whitespace(text)
    return text


def main() -> None:
    uit = pd.read_parquet(DATA_DIR / "uit_vsfc_normalized.parquet")

    n_noisy = max(1, int(len(uit) * 0.3))
    noisy_indices = random.sample(range(len(uit)), n_noisy)

    uit["is_noisy"] = False
    uit.loc[noisy_indices, "is_noisy"] = True
    uit.loc[noisy_indices, "text"] = uit.loc[noisy_indices, "text"].apply(inject_noise)

    uit.to_parquet(DATA_DIR / "uit_vsfc_noisy.parquet", index=False)
    print(f"Total: {len(uit)} rows")
    print(f"Noisy: {noisy_indices} rows ({n_noisy} / {len(uit)})")

    noisy_sample = uit[uit["is_noisy"]].head(5)
    print("\n=== Noisy samples ===")
    for _, row in noisy_sample.iterrows():
        print(f"  [{row['sentiment']}] {row['text'][:120]}")

    clean_sample = uit[~uit["is_noisy"]].head(2)
    print("\n=== Clean samples (for comparison) ===")
    for _, row in clean_sample.iterrows():
        print(f"  [{row['sentiment']}] {row['text'][:120]}")


if __name__ == "__main__":
    main()
