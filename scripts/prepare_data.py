from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from cleaning import batch_clean

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RANDOM_STATE = 42


def load_data() -> dict[str, pd.DataFrame]:
    uit = pd.read_parquet(DATA_DIR / "uit_vsfc_noisy.parquet")
    neu = pd.read_parquet(DATA_DIR / "neu_esc_filtered.parquet")
    synth = pd.read_parquet(DATA_DIR / "synthetic_raw.parquet")
    return {"uit": uit, "neu": neu, "synth": synth}


def clean_and_report(name: str, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    print(f"Cleaning {name} ({len(df)} rows)...")
    out = batch_clean(df, text_col=text_col, do_segment=True)
    before = df[text_col].str.len().mean() if text_col in df.columns else 0
    after = out["clean_text"].str.len().mean()
    tokens = out["clean_text"].str.split().str.len().mean()
    print(f"  chars: {before:.0f} -> {after:.0f} | avg tokens: {tokens:.1f}")
    return out


def build_splits(
    uit: pd.DataFrame,
    neu: pd.DataFrame,
    synth: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    uit_renamed = uit.rename(columns={"clean_text": "text_clean"})
    neu_renamed = neu.rename(columns={"clean_text": "text_clean"})
    synth_renamed = synth.rename(columns={"clean_text": "text_clean"})
    synth_renamed["topic"] = None
    synth_renamed["split"] = "train"

    base_cols = ["text_clean", "sentiment", "topic", "source", "split"]

    uit_train = uit_renamed[uit_renamed["split"] == "train"][base_cols]
    uit_val = uit_renamed[uit_renamed["split"].isin(["validation", "val"])][base_cols]
    uit_test = uit_renamed[uit_renamed["split"] == "test"][base_cols]

    if len(uit_val) == 0 and len(uit_test) == 0:
        print("  No split info found, creating splits...")
        train, temp = train_test_split(
            uit_renamed, test_size=0.2, random_state=RANDOM_STATE,
            stratify=uit_renamed["sentiment"],
        )
        val, test = train_test_split(
            temp, test_size=0.5, random_state=RANDOM_STATE,
            stratify=temp["sentiment"],
        )
        uit_train = train[base_cols]
        uit_val = val[base_cols]
        uit_test = test[base_cols]

    neu_train = neu_renamed[base_cols]
    synth_train = synth_renamed[base_cols]

    train = pd.concat([uit_train, neu_train, synth_train], ignore_index=True)
    val = uit_val.reset_index(drop=True)
    test = uit_test.reset_index(drop=True)

    train, val = _rebalance_split(train, val)

    print(f"\nTrain: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        dist = split["sentiment"].value_counts().to_dict()
        print(f"  {name}: {dist}")

    return train, val, test


def _rebalance_split(
    train: pd.DataFrame, val: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat([train, val], ignore_index=True)
    labels = combined["sentiment"]
    train_idx, val_idx = train_test_split(
        range(len(combined)), test_size=0.10, random_state=RANDOM_STATE,
        stratify=labels,
    )
    return combined.iloc[train_idx].reset_index(drop=True), combined.iloc[val_idx].reset_index(drop=True)


def save_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    train.to_parquet(DATA_DIR / "train_clean.parquet", index=False)
    val.to_parquet(DATA_DIR / "val_clean.parquet", index=False)
    test.to_parquet(DATA_DIR / "test_clean.parquet", index=False)
    print("=> train_clean.parquet, val_clean.parquet, test_clean.parquet")


def main() -> None:
    print("=== Phase 2: Data Preparation ===\n")

    data = load_data()
    noisy_count = data["uit"]["is_noisy"].sum()
    print(f"Loaded: UIT={len(data['uit'])} ({noisy_count} noisy) | NEU={len(data['neu'])} | Synth={len(data['synth'])}")

    data["uit"] = clean_and_report("UIT (noise-injected)", data["uit"])
    data["neu"] = clean_and_report("NEU-ESC", data["neu"])
    data["synth"] = clean_and_report("Synthetic", data["synth"])

    train, val, test = build_splits(data["uit"], data["neu"], data["synth"])
    save_splits(train, val, test)

    summary = {
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "total_rows": len(train) + len(val) + len(test),
        "train_sentiment": train["sentiment"].value_counts().to_dict(),
        "val_sentiment": val["sentiment"].value_counts().to_dict(),
        "test_sentiment": test["sentiment"].value_counts().to_dict(),
        "sources": pd.concat([train, val, test], ignore_index=True)["source"].value_counts().to_dict(),
    }
    (DATA_DIR / "phase2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
