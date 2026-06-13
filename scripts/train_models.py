from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402

INPUT_FILE = settings.data_dir / "processed" / "feedbacks.csv"
REPORT_DIR = settings.data_dir / "reports"
RANDOM_STATE = 42


def build_pipeline(model_name: str) -> Pipeline:
    tfidf = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    if model_name == "nb":
        classifier = MultinomialNB()
    elif model_name == "lr":
        classifier = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    elif model_name == "svm":
        classifier = CalibratedClassifierCV(
            estimator=LinearSVC(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                dual=False,
            ),
            method="sigmoid",
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return Pipeline([("tfidf", tfidf), ("classifier", classifier)])


def compute_metrics(y_true: pd.Series, y_pred) -> dict[str, float]:
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
        "f1_weighted": round(
            float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4
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

    train = train.dropna(subset=["processed_text", "sentiment", "topic"])
    test = test.dropna(subset=["processed_text", "sentiment", "topic"])

    settings.model_dir.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    targets = ["sentiment", "topic"]
    models_to_compare = ["nb", "lr", "svm"]
    results = {}

    for target in targets:
        print(f"\n--- Training and evaluating models for target: {target} ---")
        results[target] = {}
        best_f1 = -1.0
        best_model_name = ""
        best_pipeline = None
        best_predictions = None

        labels = sorted(train[target].unique().tolist())

        for name in models_to_compare:
            print(f"Huấn luyện {name.upper()}...")
            pipeline = build_pipeline(name)
            pipeline.fit(train["processed_text"], train[target])

            predictions = pipeline.predict(test["processed_text"])
            metrics_dict = compute_metrics(test[target], predictions)
            print(f"Kết quả {name.upper()}: {metrics_dict}")

            results[target][name] = metrics_dict

            if metrics_dict["f1_macro"] > best_f1:
                best_f1 = metrics_dict["f1_macro"]
                best_model_name = name
                best_pipeline = pipeline
                best_predictions = predictions

        print(f"==> Mô hình tốt nhất cho '{target}': {best_model_name.upper()} (F1 Macro: {best_f1:.4f})")

        # Lưu mô hình tốt nhất cho tác vụ
        model_filename = f"{target}_model.pkl"
        with (settings.model_dir / model_filename).open("wb") as file:
            pickle.dump(best_pipeline, file)

        # Lưu confusion matrix cho mô hình tốt nhất
        matrix_filename = f"{target}_confusion_matrix.png"
        save_confusion_matrix(test[target], best_predictions, labels, matrix_filename)

        results[target]["selected_model"] = best_model_name

    # Xuất báo cáo so sánh
    report = {
        "comparison": results,
        "algorithm": "TfidfVectorizer + [MultinomialNB, LogisticRegression, LinearSVC]",
        "random_state": RANDOM_STATE,
    }

    # Lưu file metrics.json
    output_metrics = REPORT_DIR / "metrics.json"
    with output_metrics.open("w", encoding="utf-8") as file:
        json.dumps(report, ensure_ascii=False, indent=2)
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(f"\nMetrics saved -> {output_metrics}")

    # Cập nhật file phase3_summary.json để đồng bộ dữ liệu
    summary_data = {
        "nb_test": results["sentiment"]["nb"],
        "lr_test": results["sentiment"]["lr"],
        "svm_test": results["sentiment"]["svm"],
        "nb_topic_test": results["topic"]["nb"],
        "lr_topic_test": results["topic"]["lr"],
        "svm_topic_test": results["topic"]["svm"],
        "selected_sentiment_model": results["sentiment"]["selected_model"],
        "selected_topic_model": results["topic"]["selected_model"],
    }

    output_summary = settings.data_dir / "phase3_summary.json"
    with output_summary.open("w", encoding="utf-8") as file:
        json.dump(summary_data, file, ensure_ascii=False, indent=2)
    print(f"Summary saved -> {output_summary}")

    # Dọn dẹp các tệp mô hình dư thừa cũ trong models/
    redundant_files = ["lr_model.pkl", "nb_model.pkl", "tfidf_vectorizer.pkl"]
    print("\n--- Dọn dẹp tệp mô hình cũ/dư thừa ---")
    for filename in redundant_files:
        filepath = settings.model_dir / filename
        if filepath.exists():
            try:
                filepath.unlink()
                print(f"Đã xóa tệp dư thừa: {filepath.name}")
            except Exception as exc:
                print(f"Lỗi khi xóa tệp {filepath.name}: {exc}")

    print("\nQuá trình huấn luyện, đánh giá và chuẩn hóa mô hình hoàn tất!")


if __name__ == "__main__":
    main()
