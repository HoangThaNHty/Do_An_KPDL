from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "data" / "reports"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["negative", "neutral", "positive"]
RANDOM_STATE = 42


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(DATA_DIR / "train_clean.parquet")
    val = pd.read_parquet(DATA_DIR / "val_clean.parquet")
    test = pd.read_parquet(DATA_DIR / "test_clean.parquet")
    return train, val, test


def build_vectorizer(
    train_texts: pd.Series,
    val_texts: pd.Series,
    test_texts: pd.Series,
) -> tuple[TfidfVectorizer, np.ndarray, np.ndarray, np.ndarray]:
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
    X_train = tfidf.fit_transform(train_texts)
    X_val = tfidf.transform(val_texts)
    X_test = tfidf.transform(test_texts)
    print(f"Vocabulary size: {len(tfidf.vocabulary_)}")
    print(f"Train shape: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return tfidf, X_train, X_val, X_test


def train_nb(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_val: np.ndarray,
    y_val: pd.Series,
) -> dict:
    from sklearn.naive_bayes import MultinomialNB

    nb = MultinomialNB(alpha=1.0)
    nb.fit(X_train, y_train)

    val_pred = nb.predict(X_val)
    val_metrics = _compute_metrics(y_val, val_pred)
    print(f"\n=== Naive Bayes (Val) ===")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  F1 Macro: {val_metrics['f1_macro']:.4f}")
    print(classification_report(y_val, val_pred, labels=LABELS, zero_division=0))

    return {"model": nb, "metrics": val_metrics, "val_pred": val_pred}


def train_lr(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_val: np.ndarray,
    y_val: pd.Series,
) -> dict:
    lr = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    lr.fit(X_train, y_train)

    val_pred = lr.predict(X_val)
    val_metrics = _compute_metrics(y_val, val_pred)
    print(f"\n=== Logistic Regression (Val) ===")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  F1 Macro: {val_metrics['f1_macro']:.4f}")
    print(classification_report(y_val, val_pred, labels=LABELS, zero_division=0))

    return {"model": lr, "metrics": val_metrics, "val_pred": val_pred}


def evaluate_on_test(
    model, name: str, X_test: np.ndarray, y_test: pd.Series,
) -> dict:
    test_pred = model.predict(X_test)
    test_metrics = _compute_metrics(y_test, test_pred)
    print(f"\n=== {name} (Test) ===")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1 Macro: {test_metrics['f1_macro']:.4f}")
    print(classification_report(y_test, test_pred, labels=LABELS, zero_division=0))
    return {"metrics": test_metrics, "pred": test_pred}


def save_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    title: str,
    filename: str,
) -> Path:
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    path = REPORT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved: {path}")
    return path


def save_model(model, filename: str) -> Path:
    path = MODEL_DIR / filename
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def _compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def main() -> None:
    print("=== Phase 3: Model Training ===\n")

    train, val, test = load_splits()
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    text_col = "text_clean" if "text_clean" in train.columns else "clean_text"
    tfidf, X_train, X_val, X_test = build_vectorizer(
        train[text_col], val[text_col], test[text_col],
    )

    y_train = train["sentiment"]
    y_val = val["sentiment"]
    y_test = test["sentiment"]

    nb_result = train_nb(X_train, y_train, X_val, y_val)
    lr_result = train_lr(X_train, y_train, X_val, y_val)

    nb_test = evaluate_on_test(nb_result["model"], "Naive Bayes (Test)", X_test, y_test)
    lr_test = evaluate_on_test(lr_result["model"], "Logistic Regression (Test)", X_test, y_test)

    save_confusion_matrix(y_test, nb_test["pred"], "Naive Bayes — Test", "nb_confusion_matrix.png")
    save_confusion_matrix(y_test, lr_test["pred"], "Logistic Regression — Test", "lr_confusion_matrix.png")

    save_model(tfidf, "tfidf_vectorizer.pkl")
    save_model(nb_result["model"], "nb_model.pkl")
    save_model(lr_result["model"], "lr_model.pkl")
    print("\nSaved models: tfidf_vectorizer.pkl, nb_model.pkl, lr_model.pkl")

    comparison = {
        "nb_val": nb_result["metrics"],
        "lr_val": lr_result["metrics"],
        "nb_test": nb_test["metrics"],
        "lr_test": lr_test["metrics"],
        "selected_model": "lr" if lr_test["metrics"]["f1_macro"] >= nb_test["metrics"]["f1_macro"] else "nb",
        "selected_reason": "Higher F1 Macro on test set",
    }
    print(f"\nSelected model for deployment: {comparison['selected_model'].upper()}")
    print(f"  NB test F1: {nb_test['metrics']['f1_macro']:.4f}")
    print(f"  LR test F1: {lr_test['metrics']['f1_macro']:.4f}")

    (DATA_DIR / "phase3_summary.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
