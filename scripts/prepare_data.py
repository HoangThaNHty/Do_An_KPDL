from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.cleaning import clean_text  # noqa: E402
from app.config import settings  # noqa: E402


RAW_DIR = settings.data_dir / "raw"
PROCESSED_DIR = settings.data_dir / "processed"
OUTPUT_FILE = PROCESSED_DIR / "feedbacks.csv"
SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}
TOPIC_MAP = {0: "lecturer", 1: "training_program", 2: "facility", 3: "others"}


def normalize_label(value, mapping: dict[int, str]) -> str | None:
    if pd.isna(value):
        return None
    value_str = str(value).strip().lower()
    if value_str in mapping.values():
        return value_str
    try:
        return mapping.get(int(float(value_str)))
    except ValueError:
        return None


def load_raw_data() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("*.csv"))
    if files:
        frames = [pd.read_csv(path) for path in files]
        return pd.concat(frames, ignore_index=True)

    fallback = settings.data_dir / "uit_vsfc_raw.parquet"
    if fallback.exists():
        print(f"No raw CSV found; using local fallback {fallback}")
        return pd.read_parquet(fallback)
    raise FileNotFoundError(
        "No input data found. Run python scripts/download_dataset.py first."
    )


def main() -> None:
    frame = load_raw_data()
    if "text" in frame.columns and "sentence" not in frame.columns:
        frame = frame.rename(columns={"text": "sentence"})
    required = {"sentence", "sentiment", "topic"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame["sentence"] = frame["sentence"].fillna("").astype(str).str.strip()
    frame["sentiment"] = frame["sentiment"].apply(
        lambda value: normalize_label(value, SENTIMENT_MAP)
    )
    frame["topic"] = frame["topic"].apply(
        lambda value: normalize_label(value, TOPIC_MAP)
    )
    if "split" not in frame.columns:
        frame["split"] = "train"
    frame["split"] = frame["split"].replace({"val": "validation"}).fillna("train")
    if "source" not in frame.columns:
        frame["source"] = "UIT-VSFC"
    frame["source"] = frame["source"].fillna("UIT-VSFC")

    before = len(frame)
    frame = frame[
        frame["sentence"].ne("")
        & frame["sentiment"].notna()
        & frame["topic"].notna()
    ].copy()
    print(f"Cleaning {len(frame):,} Vietnamese feedback rows...")
    frame["processed_text"] = frame["sentence"].apply(
        lambda value: clean_text(value, do_segment=True)
    )
    frame = frame[frame["processed_text"].ne("")].drop_duplicates(
        subset=["sentence", "split"]
    )
    frame = frame[
        ["sentence", "sentiment", "topic", "split", "source", "processed_text"]
    ]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Removed {before - len(frame):,} invalid/duplicate rows.")
    print(f"Saved {len(frame):,} rows -> {OUTPUT_FILE}")
    print("Sentiment:", frame["sentiment"].value_counts().to_dict())
    print("Topic:", frame["topic"].value_counts().to_dict())


if __name__ == "__main__":
    main()
