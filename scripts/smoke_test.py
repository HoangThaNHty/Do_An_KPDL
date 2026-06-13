from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.db import check_connection, get_stats  # noqa: E402
from app.main import app  # noqa: E402
from app.ml import model_service  # noqa: E402


def main() -> None:
    print(f"[OK] FastAPI imported: {app.title}")
    connected, error = check_connection()
    print(f"[{'OK' if connected else 'WARN'}] MongoDB: {'connected' if connected else error}")
    model_service.load()
    print(f"[{'OK' if model_service.ready else 'WARN'}] Models: {model_service.warning or 'loaded'}")
    if model_service.ready:
        result = model_service.predict("giảng viên dạy dễ hiểu và nhiệt tình")
        print(
            "[OK] Predict sentiment:",
            result["sentiment"]["label"],
            f"({result['sentiment']['confidence']:.2%})",
        )
        print(
            "[OK] Predict topic:",
            result["topic"]["label"],
            f"({result['topic']['confidence']:.2%})",
        )
    if connected:
        stats = get_stats()
        print("[OK] Stats:", stats["total"], "feedbacks")
        print(
            "[OK] Data origin:",
            stats["data_origin"]["uit_vsfc"],
            "UIT-VSFC /",
            stats["data_origin"]["manual_demo"],
            "manual-demo",
        )


if __name__ == "__main__":
    main()
