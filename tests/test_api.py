from fastapi.testclient import TestClient

import app.main as main


def fake_prediction(text: str) -> dict:
    return {
        "input": text,
        "processed_text": "giảng_viên dạy dễ hiểu",
        "keywords": ["giảng viên", "dạy"],
        "sentiment": {
            "label": "positive",
            "confidence": 0.91,
            "probabilities": {"positive": 0.91},
        },
        "topic": {
            "label": "lecturer",
            "confidence": 0.88,
            "probabilities": {"lecturer": 0.88},
        },
    }


def test_predict_api_valid_and_empty(monkeypatch) -> None:
    monkeypatch.setattr(main, "predict_feedback", fake_prediction)
    with TestClient(main.app) as client:
        response = client.post("/api/predict", json={"text": "Thầy dạy dễ hiểu"})
        assert response.status_code == 200
        assert response.json()["sentiment"]["label"] == "positive"
        assert response.json()["topic"]["label"] == "lecturer"

        empty_response = client.post("/api/predict", json={"text": "   "})
        assert empty_response.status_code == 422


def test_feedback_crud_api(monkeypatch) -> None:
    monkeypatch.setattr(main, "insert_feedback", lambda document: "507f1f77bcf86cd799439011")
    monkeypatch.setattr(main, "update_feedback", lambda feedback_id, fields: True)
    monkeypatch.setattr(main, "delete_feedback", lambda feedback_id: True)

    payload = {
        "sentence": "Phòng học sạch sẽ",
        "sentiment_label": "positive",
        "topic_label": "facility",
        "source": "test",
        "split": "manual",
    }
    with TestClient(main.app) as client:
        create_response = client.post("/api/feedbacks", json=payload)
        assert create_response.status_code == 201
        feedback_id = create_response.json()["id"]

        update_response = client.put(
            f"/api/feedbacks/{feedback_id}",
            json={"sentence": "Phòng học rất sạch"},
        )
        assert update_response.status_code == 200

        delete_response = client.delete(f"/api/feedbacks/{feedback_id}")
        assert delete_response.status_code == 200


def test_feedback_api_rejects_invalid_labels() -> None:
    with TestClient(main.app) as client:
        response = client.post(
            "/api/feedbacks",
            json={
                "sentence": "Nội dung",
                "sentiment_label": "unknown",
                "topic_label": "facility",
            },
        )
        assert response.status_code == 422


def test_export_csv_uses_utf8_bom(monkeypatch) -> None:
    monkeypatch.setattr(
        main,
        "get_all_feedbacks",
        lambda skip, limit: (
            [
                {
                    "sentence": "Giảng viên dạy rất dễ hiểu",
                    "sentiment": {"label": "positive"},
                    "topic": {"label": "lecturer"},
                    "processed_text": "giảng_viên dạy dễ hiểu",
                    "keywords": ["giảng viên", "dễ hiểu"],
                    "source": "UIT-VSFC",
                    "split": "test",
                    "created_at": "2026-06-09T00:00:00",
                }
            ],
            1,
        ),
    )
    with TestClient(main.app) as client:
        response = client.get("/api/export")
        assert response.status_code == 200
        assert response.content.startswith(b"\xef\xbb\xbf")
        assert "Giảng viên dạy rất dễ hiểu" in response.content.decode("utf-8-sig")
