from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.cleaning import clean_text, extract_keywords  # noqa: E402
from app.config import settings  # noqa: E402
from app.db import build_feedback_document, check_connection, init_database, upsert_feedback  # noqa: E402


SAMPLE_FILE = settings.data_dir / "sample_feedbacks.csv"


def main() -> None:
    connected, error = check_connection()
    if not connected:
        raise SystemExit(error)
    init_database()
    frame = pd.read_csv(SAMPLE_FILE)
    imported = skipped = 0
    for row in frame.itertuples(index=False):
        document = build_feedback_document(
            sentence=row.sentence,
            sentiment_label=row.sentiment,
            topic_label=row.topic,
            processed_text=clean_text(row.sentence, do_segment=True),
            keywords=extract_keywords(row.sentence),
            source="sample",
            split="manual",
        )
        if upsert_feedback(document):
            imported += 1
        else:
            skipped += 1
    print(f"Sample seed completed. Inserted: {imported}; skipped: {skipped}.")


if __name__ == "__main__":
    main()
