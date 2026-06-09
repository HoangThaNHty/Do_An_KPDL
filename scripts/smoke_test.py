from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.db import check_connection, get_stats
from app.main import app
from app.ml import model_service


def main() -> None:
    print(f"[OK] FastAPI imported: {app.title}")
    connected, error = check_connection()
    print(f"[{'OK' if connected else 'WARN'}] MongoDB: {'connected' if connected else error}")
    model_service.load()
    print(f"[{'OK' if model_service.ready else 'WARN'}] Models: {model_service.warning or 'loaded'}")
    if model_service.ready:
        result = model_service.predict("giảng viên dạy dễ hiểu và nhiệt tình")
        print(
            "[OK] Predict:",
            result["sentiment"]["label"],
            "/",
            result["topic"]["label"],
        )
    if connected:
        print("[OK] Stats:", get_stats()["total"], "feedbacks")


if __name__ == "__main__":
    main()
