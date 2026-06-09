from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.cleaning import extract_keywords
from app.config import settings
from app.db import build_feedback_document, check_connection, init_database, upsert_feedback


INPUT_FILE = settings.data_dir / "processed" / "feedbacks.csv"


def main() -> None:
    connected, error = check_connection()
    if not connected:
        raise SystemExit(error)
    if not INPUT_FILE.exists():
        raise SystemExit(
            f"Missing {INPUT_FILE}. Run python scripts/prepare_data.py first."
        )
    init_database()
    frame = pd.read_csv(INPUT_FILE)
    imported = skipped = 0
    for row in frame.itertuples(index=False):
        document = build_feedback_document(
            sentence=row.sentence,
            sentiment_label=row.sentiment,
            topic_label=row.topic,
            processed_text=row.processed_text,
            keywords=extract_keywords(row.sentence),
            source=row.source,
            split=row.split,
        )
        if upsert_feedback(document):
            imported += 1
        else:
            skipped += 1
    print(f"Import completed. Inserted: {imported:,}; skipped duplicates: {skipped:,}.")


if __name__ == "__main__":
    main()
