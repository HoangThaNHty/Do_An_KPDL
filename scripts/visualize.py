from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORT_DIR = DATA_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["font.size"] = 12

COLORS_SENTIMENT = {"negative": "#d9534f", "neutral": "#5bc0de", "positive": "#5cb85c"}
LABELS_SENTIMENT = ["negative", "neutral", "positive"]
TOPIC_COLORS = {"lecturer": "#2196F3", "training_program": "#FF9800", "facility": "#f44336", "others": "#9C27B0"}


def load_data() -> dict[str, pd.DataFrame]:
    return {
        "train": pd.read_parquet(DATA_DIR / "train_clean.parquet"),
        "val": pd.read_parquet(DATA_DIR / "val_clean.parquet"),
        "test": pd.read_parquet(DATA_DIR / "test_clean.parquet"),
    }


def plot_sentiment_pie(df: pd.DataFrame, title: str, filename: str) -> None:
    counts = df["sentiment"].value_counts()
    colors = [COLORS_SENTIMENT.get(s, "#999999") for s in counts.index]
    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 13},
    )
    for t in autotexts:
        t.set_fontsize(11)
        t.set_color("white")
        t.set_fontweight("bold")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=20)
    path = REPORT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_sentiment_bar(df: pd.DataFrame, title: str, filename: str) -> None:
    counts = df["sentiment"].value_counts()
    colors = [COLORS_SENTIMENT.get(s, "#999999") for s in counts.index]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                str(val), ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    path = REPORT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_topic_bar(df: pd.DataFrame, filename: str) -> None:
    topic_df = df[df["topic"].notna()]
    if "topic" not in topic_df.columns or topic_df["topic"].nunique() == 0:
        print("  No topic data found, skipping topic chart")
        return
    counts = topic_df["topic"].value_counts()
    colors = [TOPIC_COLORS.get(t, "#999999") for t in counts.index]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(counts.index, counts.values, color=colors, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height() / 2,
                str(val), ha="left", va="center", fontweight="bold", fontsize=11)
    ax.set_title("Phân bố Topic", fontsize=15, fontweight="bold")
    ax.set_xlabel("Count")
    ax.invert_yaxis()
    path = REPORT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_word_clouds(df: pd.DataFrame, filename_prefix: str) -> None:
    text_col = "text_clean" if "text_clean" in df.columns else "clean_text"
    sentiments = ["negative", "neutral", "positive"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, sentiment in zip(axes, sentiments):
        texts = df[df["sentiment"] == sentiment][text_col].dropna().str.cat(sep=" ")
        if not texts.strip():
            ax.set_title(f"{sentiment} (no data)")
            ax.axis("off")
            continue
        wc = WordCloud(
            width=800, height=400,
            background_color="white",
            colormap="coolwarm" if sentiment == "negative" else "viridis" if sentiment == "neutral" else "YlGn",
            max_words=80,
        ).generate(texts)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(sentiment.upper(), fontsize=14, fontweight="bold",
                     color=COLORS_SENTIMENT.get(sentiment, "#333"))
        ax.axis("off")
    fig.suptitle("Word Clouds theo Sentiment", fontsize=16, fontweight="bold", y=1.02)
    path = REPORT_DIR / f"{filename_prefix}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_sentiment_by_source(df: pd.DataFrame, filename: str) -> None:
    if "source" not in df.columns:
        return
    cross = pd.crosstab(df["source"], df["sentiment"], normalize="index") * 100
    colors = [COLORS_SENTIMENT.get(s, "#999") for s in cross.columns]
    fig, ax = plt.subplots(figsize=(10, 6))
    cross[["negative", "neutral", "positive"]].plot(
        kind="bar", stacked=True, color=colors, edgecolor="white", ax=ax,
    )
    ax.set_title("Sentiment Distribution theo Dataset", fontsize=15, fontweight="bold")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Percentage (%)")
    ax.legend(title="Sentiment")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    path = REPORT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def extract_insights(df: pd.DataFrame) -> dict:
    text_col = "text_clean" if "text_clean" in df.columns else "clean_text"
    insights: dict = {}

    total = len(df)
    sentiment_pct = (df["sentiment"].value_counts() / total * 100).round(1).to_dict()
    insights["sentiment_percentages"] = sentiment_pct

    if "topic" in df.columns and df["topic"].notna().any():
        topic_sentiment = df[df["topic"].notna()].groupby("topic")["sentiment"].value_counts(normalize=True) * 100
        top_topic_by_sentiment = {}
        for sentiment in LABELS_SENTIMENT:
            try:
                subset = topic_sentiment.xs(sentiment, level="sentiment")
                top = subset.idxmax()
                top_pct = subset.max()
                top_topic_by_sentiment[sentiment] = {"topic": top, "pct": round(top_pct, 1)}
            except KeyError:
                pass
        insights["top_topic_by_sentiment"] = top_topic_by_sentiment

    if text_col in df.columns:
        negative_texts = df[df["sentiment"] == "negative"][text_col].dropna()
        word_freq = pd.Series(
            " ".join(negative_texts).split()
        ).value_counts().head(20)
        insights["top_negative_words"] = word_freq.to_dict()

    positive_texts = df[df["sentiment"] == "positive"][text_col].dropna() if text_col in df.columns else pd.Series()
    positive_words = pd.Series(" ".join(positive_texts).split()).value_counts().head(20)
    insights["top_positive_words"] = positive_words.to_dict() if len(positive_texts) > 0 else {}

    return insights


def main() -> None:
    print("=== Phase 4: Visualization ===\n")

    data = load_data()
    all_data = pd.concat(data.values(), ignore_index=True)
    print(f"Total rows: {len(all_data)}")

    print("\n1. Sentiment Distribution")
    plot_sentiment_pie(all_data, "Sentiment Distribution (Overall)", "sentiment_pie.png")
    plot_sentiment_bar(all_data, "Sentiment Count", "sentiment_bar.png")

    print("\n2. Topic Distribution")
    uit_only = all_data[all_data["source"] == "uit_vsfc"].copy()
    plot_topic_bar(uit_only, "topic_bar.png")

    print("\n3. Word Clouds")
    plot_word_clouds(all_data, "word_clouds")

    print("\n4. Sentiment by Source")
    plot_sentiment_by_source(all_data, "sentiment_by_source.png")

    print("\n5. Extracting Insights")
    insights = extract_insights(all_data)

    print(f"\n  Sentiment %: {insights.get('sentiment_percentages', {})}")
    print(f"  Top topic by sentiment: {insights.get('top_topic_by_sentiment', {})}")
    print(f"  Top negative words: {list(insights.get('top_negative_words', {}).keys())[:10]}")
    print(f"  Top positive words: {list(insights.get('top_positive_words', {}).keys())[:10]}")

    (DATA_DIR / "phase4_insights.json").write_text(
        json.dumps(insights, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    print(f"\n  All charts saved to: {REPORT_DIR}")
    print("  Insights saved to: phase4_insights.json")


if __name__ == "__main__":
    main()
