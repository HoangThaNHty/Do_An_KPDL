from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

DATASET_ID = "uitnlp/vietnamese_students_feedback"
SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}
TOPIC_MAP = {0: "lecturer", 1: "training_program", 2: "facility", 3: "others"}


def _label_value(dataset, column: str, value) -> str:
    feature = dataset.features.get(column)
    if hasattr(feature, "int2str"):
        try:
            value = feature.int2str(int(value))
        except (TypeError, ValueError):
            pass
    if column == "sentiment":
        return SENTIMENT_MAP.get(int(value), str(value)) if str(value).isdigit() else str(value)
    if column == "topic":
        return TOPIC_MAP.get(int(value), str(value)) if str(value).isdigit() else str(value)
    return str(value)


def download_with_datasets() -> int:
    from datasets import load_dataset

    dataset_dict = load_dataset(DATASET_ID)
    saved = 0
    for split_name, dataset in dataset_dict.items():
        frame = dataset.to_pandas()
        if "text" in frame.columns and "sentence" not in frame.columns:
            frame = frame.rename(columns={"text": "sentence"})
        frame["sentiment"] = [
            _label_value(dataset, "sentiment", value) for value in frame["sentiment"]
        ]
        frame["topic"] = [
            _label_value(dataset, "topic", value) for value in frame["topic"]
        ]
        frame["split"] = "validation" if split_name == "val" else split_name
        frame["source"] = "UIT-VSFC"
        output = RAW_DIR / f"{frame['split'].iloc[0]}.csv"
        frame[["sentence", "sentiment", "topic", "split", "source"]].to_csv(
            output, index=False, encoding="utf-8-sig"
        )
        print(f"Saved {len(frame):,} rows -> {output}")
        saved += len(frame)
    return saved


def use_local_fallback() -> int:
    local_file = DATA_DIR / "uit_vsfc_raw.parquet"
    if not local_file.exists():
        return 0
    frame = pd.read_parquet(local_file).rename(columns={"text": "sentence"})
    saved = 0
    for split_name, split_frame in frame.groupby("split"):
        split_name = "validation" if split_name == "val" else str(split_name)
        split_frame = split_frame.copy()
        split_frame["split"] = split_name
        split_frame["source"] = "UIT-VSFC"
        output = RAW_DIR / f"{split_name}.csv"
        split_frame[["sentence", "sentiment", "topic", "split", "source"]].to_csv(
            output, index=False, encoding="utf-8-sig"
        )
        print(f"Used local fallback: {len(split_frame):,} rows -> {output}")
        saved += len(split_frame)
    return saved


def main() -> None:
    try:
        total = download_with_datasets()
        print(f"Download completed: {total:,} rows.")
    except Exception as exc:
        print(f"Hugging Face download failed: {exc}")
        total = use_local_fallback()
        if total:
            print(f"Local dataset fallback completed: {total:,} rows.")
            return
        print(
            "\nManual fallback:\n"
            f"1. Open https://huggingface.co/datasets/{DATASET_ID}\n"
            "2. Download train/validation/test files.\n"
            f"3. Put them in {RAW_DIR} as train.csv, validation.csv, test.csv.\n"
            "4. Ensure columns are sentence, sentiment and topic."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
