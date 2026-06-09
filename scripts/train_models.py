from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import settings


INPUT_FILE = settings.data_dir / "processed" / "feedbacks.csv"
REPORT_DIR = settings.data_dir / "reports"
RANDOM_STATE = 42


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=15000,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def metrics(y_true: pd.Series, y_pred) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(
            float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            4,
        ),
        "recall_macro": round(
            float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            4,
        ),
        "f1_macro": round(
            float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
    }


def save_confusion_matrix(y_true, y_pred, labels: list[str], filename: str) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, axis = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=axis,
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title(filename.replace("_", " ").replace(".png", "").title())
    fig.tight_layout()
    fig.savefig(REPORT_DIR / filename, dpi=150)
    plt.close(fig)


def train_task(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    model_filename: str,
    matrix_filename: str,
) -> dict[str, object]:
    train = train.dropna(subset=["processed_text", target])
    test = test.dropna(subset=["processed_text", target])
    model = build_pipeline()
    model.fit(train["processed_text"], train[target])
    prediction = model.predict(test["processed_text"])
    labels = sorted(train[target].unique().tolist())
    result = metrics(test[target], prediction)
    result["train_rows"] = int(len(train))
    result["test_rows"] = int(len(test))
    result["labels"] = labels
    with (settings.model_dir / model_filename).open("wb") as file:
        pickle.dump(model, file)
    save_confusion_matrix(test[target], prediction, labels, matrix_filename)
    print(f"{target}: {result}")
    return result


def main() -> None:
    if not INPUT_FILE.exists():
        raise SystemExit(
            f"Missing {INPUT_FILE}. Run python scripts/prepare_data.py first."
        )
    frame = pd.read_csv(INPUT_FILE)
    train = frame[frame["split"] == "train"].copy()
    test = frame[frame["split"].isin(["validation", "test"])].copy()
    if train.empty or test.empty:
        raise SystemExit("Dataset must contain train and validation/test splits.")

    settings.model_dir.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "sentiment": train_task(
            train,
            test,
            "sentiment",
            "sentiment_model.pkl",
            "sentiment_confusion_matrix.png",
        ),
        "topic": train_task(
            train,
            test,
            "topic",
            "topic_model.pkl",
            "topic_confusion_matrix.png",
        ),
        "algorithm": "TfidfVectorizer + LogisticRegression",
        "random_state": RANDOM_STATE,
    }
    output = REPORT_DIR / "metrics.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Models saved in {settings.model_dir}")
    print(f"Metrics saved -> {output}")


if __name__ == "__main__":
    main()
