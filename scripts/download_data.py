from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.errors import GatedRepoError


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / ".env")
HF_TOKEN: str | None = os.getenv("HF_TOKEN")


UIT_URLS = {
    "train": {
        "sentences": "https://drive.google.com/uc?id=1nzak5OkrheRV1ltOGCXkT671bmjODLhP&export=download",
        "sentiments": "https://drive.google.com/uc?id=1ye-gOZIBqXdKOoi_YxvpT6FeRNmViPPv&export=download",
        "topics": "https://drive.google.com/uc?id=14MuDtwMnNOcr4z_8KdpxprjbwaQ7lJ_C&export=download",
    },
    "validation": {
        "sentences": "https://drive.google.com/uc?id=1sMJSR3oRfPc3fe1gK-V3W5F24tov_517&export=download",
        "sentiments": "https://drive.google.com/uc?id=1GiY1AOp41dLXIIkgES4422AuDwmbUseL&export=download",
        "topics": "https://drive.google.com/uc?id=1DwLgDEaFWQe8mOd7EpF-xqMEbDLfdT-W&export=download",
    },
    "test": {
        "sentences": "https://drive.google.com/uc?id=1aNMOeZZbNwSRkjyCWAGtNCMa3YrshR-n&export=download",
        "sentiments": "https://drive.google.com/uc?id=1vkQS5gI0is4ACU58-AbWusnemw7KZNfO&export=download",
        "topics": "https://drive.google.com/uc?id=1_ArMpDguVsbUGl-xSMkTF_p5KpZrmpSB&export=download",
    },
}

UIT_TOPIC_MAP = {0: "lecturer", 1: "training_program", 2: "facility", 3: "others"}
UIT_SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}

NEU_TOPIC_MAP = {
    0: "general", 1: "facility", 2: "academic", 3: "lecturer",
    4: "program", 5: "social", 6: "administrative", 7: "student_service",
    8: "financial", 9: "other",
}
NEU_SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive", 3: "toxic"}

SYNTHETIC_REPO_ID = "thnhan3/vietnamese-students-feedback-synthetic"


def save_frame(frame: pd.DataFrame, file_name: str) -> None:
    frame.to_parquet(DATA_DIR / file_name, index=False)


def summarize_frame(frame: pd.DataFrame) -> dict:
    summary = {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "missing_per_column": {c: int(v) for c, v in frame.isna().sum().items()},
    }
    for col in ("sentiment", "topic"):
        if col in frame.columns:
            key = f"{col}_distribution"
            summary[key] = {str(k): int(v) for k, v in frame[col].value_counts(dropna=False).items()}
    return summary


def fetch_text_lines(url: str) -> list[str]:
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return [line.strip() for line in resp.text.splitlines() if line.strip()]


def download_uit_vsfc() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split_name, urls in UIT_URLS.items():
        sentences = fetch_text_lines(urls["sentences"])
        sentiments = [int(v) for v in fetch_text_lines(urls["sentiments"])]
        topics = [int(v) for v in fetch_text_lines(urls["topics"])]
        frames.append(pd.DataFrame({
            "text": sentences,
            "sentiment": [UIT_SENTIMENT_MAP[v] for v in sentiments],
            "topic": [UIT_TOPIC_MAP[v] for v in topics],
            "split": split_name,
            "source": "uit_vsfc",
        }))
    return pd.concat(frames, ignore_index=True)


def download_neu_esc() -> tuple[pd.DataFrame | None, str | None]:
    token: str | None = HF_TOKEN
    files = ["train_set.csv", "val_set.csv", "test_set.csv"]
    frames: list[pd.DataFrame] = []
    try:
        for fname in files:
            path = hf_hub_download("hung20gg/NEU-ESC", repo_type="dataset", filename=fname, token=token)
            df = pd.read_csv(path)
            df["split"] = fname.replace("_set.csv", "")
            df["source"] = "neu_esc"
            if "classification" in df.columns:
                df["topic"] = df["classification"].map(NEU_TOPIC_MAP)
                df.drop(columns=["classification"], inplace=True)
            if "sentiment" in df.columns:
                df["sentiment"] = df["sentiment"].map(NEU_SENTIMENT_MAP)
            rename_cols = {}
            if "sentence" in df.columns and "text" not in df.columns:
                rename_cols["sentence"] = "text"
            if rename_cols:
                df.rename(columns=rename_cols, inplace=True)
            frames.append(df)
    except GatedRepoError:
        return None, "NEU-ESC is gated. Provide HF_TOKEN in .env to access."
    except Exception as exc:
        return None, f"NEU-ESC download failed: {exc}"
    return pd.concat(frames, ignore_index=True), None


def filter_neu_esc(frame: pd.DataFrame) -> pd.DataFrame:
    if "topic" not in frame.columns:
        return pd.DataFrame()
    mask = frame["topic"] == "academic"
    if "sentiment" in frame.columns:
        mask &= frame["sentiment"] != "toxic"
    out = frame.loc[mask].copy()
    if "topic" in out.columns:
        out["topic"] = "academic"
    return out


def download_synthetic() -> tuple[pd.DataFrame | None, str | None]:
    token: str | None = HF_TOKEN
    try:
        files = list_repo_files(SYNTHETIC_REPO_ID, repo_type="dataset", token=token)
    except Exception as exc:
        return None, f"Synthetic repo not accessible: {exc}"
    csv_files = [f for f in files if f.lower().endswith(".csv")]
    parquet_files = [f for f in files if f.lower().endswith(".parquet")]
    frames: list[pd.DataFrame] = []
    try:
        for fname in csv_files:
            path = hf_hub_download(SYNTHETIC_REPO_ID, repo_type="dataset", filename=fname, token=token)
            df = pd.read_csv(path)
            df["source"] = "synthetic"
            frames.append(df)
        for fname in parquet_files:
            path = hf_hub_download(SYNTHETIC_REPO_ID, repo_type="dataset", filename=fname, token=token)
            df = pd.read_parquet(path)
            df["source"] = "synthetic"
            frames.append(df)
    except Exception as exc:
        return None, f"Synthetic download failed: {exc}"
    if not frames:
        return None, "No CSV or parquet files found in synthetic repo."
    return pd.concat(frames, ignore_index=True), None


def normalize_synthetic(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    rename_cols = {}
    if "sentence" in out.columns and "text" not in out.columns:
        rename_cols["sentence"] = "text"
    if rename_cols:
        out.rename(columns=rename_cols, inplace=True)
    if "sentiment" in out.columns:
        out["sentiment"] = out["sentiment"].map(UIT_SENTIMENT_MAP)
    return out


def main() -> None:
    summary: dict[str, dict] = {}
    issues: list[str] = []

    uit = download_uit_vsfc()
    save_frame(uit, "uit_vsfc_raw.parquet")
    save_frame(uit, "uit_vsfc_normalized.parquet")
    summary["uit_vsfc"] = summarize_frame(uit)

    neu, neu_err = download_neu_esc()
    if neu is not None:
        save_frame(neu, "neu_esc_raw.parquet")
        save_frame(neu, "neu_esc_normalized.parquet")
        neu_filtered = filter_neu_esc(neu)
        save_frame(neu_filtered, "neu_esc_filtered.parquet")
        summary["neu_esc"] = summarize_frame(neu)
        summary["neu_esc_filtered"] = summarize_frame(neu_filtered)
    else:
        issues.append(neu_err or "NEU-ESC download failed.")

    syn, syn_err = download_synthetic()
    if syn is not None:
        syn = normalize_synthetic(syn)
        save_frame(syn, "synthetic_raw.parquet")
        save_frame(syn, "synthetic_normalized.parquet")
        summary["synthetic"] = summarize_frame(syn)
    else:
        issues.append(syn_err or "Synthetic download failed.")

    output = {"summary": summary, "issues": issues}
    (DATA_DIR / "phase1_summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
