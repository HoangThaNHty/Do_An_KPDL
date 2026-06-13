from __future__ import annotations

import json
import io
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import jinja2
import pandas as pd
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.cleaning import clean_text, extract_keywords
from app.config import ROOT_DIR, settings
from app.db import (
    DatabaseUnavailable,
    build_feedback_document,
    check_connection,
    delete_feedback,
    empty_stats,
    get_all_feedbacks,
    get_stats,
    init_database,
    insert_feedback,
    update_feedback,
    upsert_feedback,
    get_top_keywords_by_sentiment,
)
from app.ml import ModelNotReadyError, model_service
from app.schemas import FeedbackCreate, FeedbackUpdate, PredictRequest


TEMPLATE_DIR = ROOT_DIR / "app" / "templates"
STATIC_DIR = ROOT_DIR / "app" / "static"
SENTIMENT_MAP = {
    "negative": "Tiêu cực",
    "neutral": "Trung tính",
    "positive": "Tích cực",
}
TOPIC_MAP = {
    "lecturer": "Giảng viên",
    "training_program": "Chương trình đào tạo",
    "facility": "Cơ sở vật chất",
    "others": "Khác",
}
SPLIT_MAP = {
    "train": "Huấn luyện",
    "validation": "Xác thực",
    "test": "Kiểm thử",
    "manual": "Nhập thủ công",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service.load()
    app.state.model_warning = model_service.warning
    connected, database_warning = check_connection()
    app.state.database_warning = database_warning
    if connected:
        try:
            init_database()
        except DatabaseUnavailable as exc:
            app.state.database_warning = f"Không thể khởi tạo MongoDB: {exc}"
    yield


app = FastAPI(
    title="SFAS - Student Feedback Analytics System",
    description="Quản lý và khai thác dữ liệu phản hồi sinh viên tiếng Việt.",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape=jinja2.select_autoescape(["html", "xml"]),
)


def render_template(template_name: str, **context: Any) -> HTMLResponse:
    context.setdefault("SENTIMENT_MAP", SENTIMENT_MAP)
    context.setdefault("TOPIC_MAP", TOPIC_MAP)
    context.setdefault("SPLIT_MAP", SPLIT_MAP)
    context.setdefault("model_warning", getattr(app.state, "model_warning", None))
    context.setdefault("database_warning", getattr(app.state, "database_warning", None))
    return HTMLResponse(jinja_env.get_template(template_name).render(**context))


def safe_stats() -> tuple[dict[str, Any], str | None]:
    try:
        return get_stats(), None
    except DatabaseUnavailable as exc:
        return empty_stats(), f"MongoDB chưa sẵn sàng: {exc}"


def predict_feedback(text: str) -> dict[str, Any]:
    return model_service.predict(text)


def document_from_prediction(
    sentence: str,
    prediction: dict[str, Any],
    source: str = "manual",
    split: str = "manual",
) -> dict[str, Any]:
    return build_feedback_document(
        sentence=sentence,
        sentiment_label=prediction["sentiment"]["label"],
        topic_label=prediction["topic"]["label"],
        processed_text=prediction["processed_text"],
        keywords=prediction.get("keywords", []),
        source=source,
        split=split,
        prediction={
            "sentiment_label": prediction["sentiment"]["label"],
            "sentiment_confidence": prediction["sentiment"]["confidence"],
            "topic_label": prediction["topic"]["label"],
            "topic_confidence": prediction["topic"]["confidence"],
        },
    )


def document_from_payload(payload: FeedbackCreate) -> dict[str, Any]:
    prediction: dict[str, Any] | None = None
    if payload.sentiment_label is None or payload.topic_label is None:
        prediction = predict_feedback(payload.sentence)
    sentiment_label = payload.sentiment_label or prediction["sentiment"]["label"]
    topic_label = payload.topic_label or prediction["topic"]["label"]
    processed_text = (
        prediction["processed_text"]
        if prediction
        else clean_text(payload.sentence, do_segment=True)
    )
    return build_feedback_document(
        sentence=payload.sentence,
        sentiment_label=sentiment_label,
        topic_label=topic_label,
        processed_text=processed_text,
        keywords=extract_keywords(payload.sentence),
        source=payload.source,
        split=payload.split,
        prediction={
            "sentiment_label": (
                prediction["sentiment"]["label"] if prediction else sentiment_label
            ),
            "sentiment_confidence": (
                prediction["sentiment"]["confidence"] if prediction else 0.0
            ),
            "topic_label": prediction["topic"]["label"] if prediction else topic_label,
            "topic_confidence": (
                prediction["topic"]["confidence"] if prediction else 0.0
            ),
        },
    )


def redirect_with_message(path: str, message: str, level: str = "success"):
    return RedirectResponse(
        url=f"{path}?{urlencode({'message': message, 'level': level})}",
        status_code=303,
    )


# Pages
@app.get("/", response_class=HTMLResponse)
async def home(
    message: str | None = None,
    level: str = "info",
):
    stats, database_warning = safe_stats()
    return render_template(
        "index.html",
        stats=stats,
        message=message,
        level=level,
        database_warning=database_warning
        or getattr(app.state, "database_warning", None),
        result=None,
        text="",
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict_page(text: str = Form(...)):
    text = text.strip()
    stats, database_warning = safe_stats()
    if not text:
        return render_template(
            "index.html",
            stats=stats,
            result=None,
            text="",
            message="Vui lòng nhập nội dung phản hồi.",
            level="warning",
            database_warning=database_warning,
        )
    try:
        result = predict_feedback(text)
    except (ModelNotReadyError, ValueError) as exc:
        return render_template(
            "index.html",
            stats=stats,
            result=None,
            text=text,
            message=str(exc),
            level="warning",
            database_warning=database_warning,
        )
    return render_template(
        "index.html",
        stats=stats,
        result=result,
        text=text,
        message=None,
        level="info",
        database_warning=database_warning,
    )


@app.post("/feedback/save-prediction")
async def save_prediction(text: str = Form(...)):
    text = text.strip()
    if not text:
        return redirect_with_message("/", "Phản hồi không được để trống.", "warning")
    try:
        prediction = predict_feedback(text)
        insert_feedback(document_from_prediction(text, prediction))
        return redirect_with_message("/", "Đã lưu phản hồi vào MongoDB.")
    except (ModelNotReadyError, DatabaseUnavailable, ValueError) as exc:
        return redirect_with_message("/", str(exc), "danger")


@app.get("/feedbacks", response_class=HTMLResponse)
async def feedback_list(
    page: int = Query(1, ge=1),
    sentiment: str | None = Query(None),
    topic: str | None = Query(None),
    split: str | None = Query(None),
    search: str | None = Query(None),
    message: str | None = None,
    level: str = "info",
):
    per_page = 15
    try:
        feedbacks, total = get_all_feedbacks(
            skip=(page - 1) * per_page,
            limit=per_page,
            sentiment_filter=sentiment,
            topic_filter=topic,
            split_filter=split,
            search=search,
        )
        database_warning = None
    except DatabaseUnavailable as exc:
        feedbacks, total = [], 0
        database_warning = f"MongoDB chưa sẵn sàng: {exc}"
    return render_template(
        "list.html",
        feedbacks=feedbacks,
        total=total,
        page=page,
        total_pages=max(1, (total + per_page - 1) // per_page),
        sentiment_filter=sentiment,
        topic_filter=topic,
        split_filter=split,
        search=search,
        message=message,
        level=level,
        database_warning=database_warning,
    )


@app.post("/feedback/add")
async def add_feedback(text: str = Form(...)):
    text = text.strip()
    if not text:
        return redirect_with_message(
            "/feedbacks", "Phản hồi không được để trống.", "warning"
        )
    try:
        prediction = predict_feedback(text)
        insert_feedback(document_from_prediction(text, prediction))
        return redirect_with_message("/feedbacks", "Đã thêm phản hồi.")
    except (ModelNotReadyError, DatabaseUnavailable, ValueError) as exc:
        return redirect_with_message("/feedbacks", str(exc), "danger")


@app.post("/feedback/{feedback_id}/update")
async def edit_feedback(
    feedback_id: str,
    text: str = Form(...),
    sentiment: str = Form(...),
    topic: str = Form(...),
    split: str = Form(...),
):
    if sentiment not in SENTIMENT_MAP or topic not in TOPIC_MAP or split not in SPLIT_MAP:
        return redirect_with_message("/feedbacks", "Nhãn cập nhật không hợp lệ.", "danger")
    text = text.strip()
    if not text:
        return redirect_with_message(
            "/feedbacks", "Phản hồi không được để trống.", "warning"
        )
    try:
        changed = update_feedback(
            feedback_id,
            {
                "sentence": text,
                "processed_text": clean_text(text, do_segment=True),
                "keywords": extract_keywords(text),
                "sentiment": {
                    "label": sentiment,
                    "code": {"negative": 0, "neutral": 1, "positive": 2}[sentiment],
                },
                "topic": {
                    "label": topic,
                    "code": {
                        "lecturer": 0,
                        "training_program": 1,
                        "facility": 2,
                        "others": 3,
                    }[topic],
                },
                "split": split,
            },
        )
    except (DatabaseUnavailable, ValueError) as exc:
        return redirect_with_message("/feedbacks", str(exc), "danger")
    if not changed:
        return redirect_with_message("/feedbacks", "Không tìm thấy phản hồi.", "danger")
    return redirect_with_message("/feedbacks", "Đã cập nhật phản hồi.")


@app.post("/feedback/{feedback_id}/delete")
async def remove_feedback(feedback_id: str):
    try:
        deleted = delete_feedback(feedback_id)
    except DatabaseUnavailable as exc:
        return redirect_with_message("/feedbacks", str(exc), "danger")
    if not deleted:
        return redirect_with_message("/feedbacks", "ID phản hồi không hợp lệ.", "danger")
    return redirect_with_message("/feedbacks", "Đã xóa phản hồi.")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    stats, database_warning = safe_stats()
    pos_keywords = []
    neg_keywords = []
    if not database_warning:
        try:
            pos_keywords = get_top_keywords_by_sentiment("positive", limit=15)
            neg_keywords = get_top_keywords_by_sentiment("negative", limit=15)
        except Exception:
            pass
    return render_template(
        "dashboard.html",
        stats=stats,
        database_warning=database_warning,
        pos_keywords=pos_keywords,
        neg_keywords=neg_keywords,
    )


@app.get("/about", response_class=HTMLResponse)
async def about():
    metrics_path = settings.data_dir / "reports" / "metrics.json"
    comparison = None
    if metrics_path.exists():
        try:
            with metrics_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
                comparison = data.get("comparison")
        except Exception:
            pass
    return render_template("about.html", comparison=comparison)


# JSON API
@app.get("/api/stats")
async def api_stats():
    try:
        return {"success": True, "data": get_stats()}
    except DatabaseUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"MongoDB chưa sẵn sàng: {exc}")


@app.get("/api/feedbacks")
async def api_feedbacks(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    sentiment: str | None = None,
    topic: str | None = None,
    split: str | None = None,
    search: str | None = None,
    sort: str = Query("desc", pattern="^(asc|desc)$"),
):
    try:
        documents, total = get_all_feedbacks(
            skip=(page - 1) * limit,
            limit=limit,
            sentiment_filter=sentiment,
            topic_filter=topic,
            split_filter=split,
            search=search,
            sort_order=sort,
        )
        return {
            "success": True,
            "data": documents,
            "pagination": {"page": page, "limit": limit, "total": total},
        }
    except DatabaseUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"MongoDB chưa sẵn sàng: {exc}")


@app.post("/api/feedbacks", status_code=201)
async def api_create_feedback(payload: FeedbackCreate):
    try:
        feedback_id = insert_feedback(document_from_payload(payload))
        return {
            "success": True,
            "message": "Đã tạo phản hồi.",
            "id": feedback_id,
        }
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except DatabaseUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"MongoDB chưa sẵn sàng: {exc}")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.put("/api/feedbacks/{feedback_id}")
async def api_update_feedback(feedback_id: str, payload: FeedbackUpdate):
    values = payload.model_dump(exclude_none=True)
    update_fields: dict[str, Any] = {}
    if "sentence" in values:
        update_fields["sentence"] = values["sentence"]
        update_fields["processed_text"] = clean_text(values["sentence"], do_segment=True)
        update_fields["keywords"] = extract_keywords(values["sentence"])
    if "sentiment_label" in values:
        label = values["sentiment_label"]
        update_fields["sentiment"] = {
            "label": label,
            "code": {"negative": 0, "neutral": 1, "positive": 2}[label],
        }
    if "topic_label" in values:
        label = values["topic_label"]
        update_fields["topic"] = {
            "label": label,
            "code": {
                "lecturer": 0,
                "training_program": 1,
                "facility": 2,
                "others": 3,
            }[label],
        }
    for field in ("source", "split"):
        if field in values:
            update_fields[field] = values[field]
    if not update_fields:
        raise HTTPException(status_code=400, detail="Không có trường nào để cập nhật")
    try:
        if not update_feedback(feedback_id, update_fields):
            raise HTTPException(status_code=404, detail="Không tìm thấy phản hồi")
        return {"success": True, "message": "Đã cập nhật phản hồi."}
    except DatabaseUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"MongoDB chưa sẵn sàng: {exc}")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.delete("/api/feedbacks/{feedback_id}")
async def api_delete_feedback(feedback_id: str):
    try:
        if not delete_feedback(feedback_id):
            raise HTTPException(status_code=404, detail="Không tìm thấy phản hồi")
        return {"success": True, "message": "Đã xóa phản hồi."}
    except DatabaseUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"MongoDB chưa sẵn sàng: {exc}")


@app.post("/api/predict")
async def api_predict(payload: PredictRequest):
    try:
        return predict_feedback(payload.text)
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _normalize_label(value: Any, mapping: dict[int, str]) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, str) and value in mapping.values():
        return value
    try:
        return mapping.get(int(value))
    except (TypeError, ValueError):
        return None


@app.post("/api/import")
async def api_import(file: UploadFile = File(...)):
    content = await file.read()
    suffix = Path(file.filename or "").suffix.lower()
    try:
        if suffix == ".csv":
            frame = pd.read_csv(io.BytesIO(content))
        elif suffix == ".parquet":
            frame = pd.read_parquet(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Chỉ hỗ trợ CSV hoặc Parquet")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Không thể đọc file: {exc}")

    text_column = next(
        (
            name
            for name in ("sentence", "text", "feedback", "comment")
            if name in frame.columns
        ),
        None,
    )
    if text_column is None:
        raise HTTPException(
            status_code=400,
            detail="File cần có cột sentence, text, feedback hoặc comment",
        )

    imported = skipped = failed = 0
    errors: list[str] = []
    for row_number, row in frame.iterrows():
        sentence = str(row.get(text_column, "")).strip()
        if not sentence or sentence.lower() == "nan":
            skipped += 1
            continue
        sentiment = _normalize_label(
            row.get("sentiment"), {0: "negative", 1: "neutral", 2: "positive"}
        )
        topic = _normalize_label(
            row.get("topic"),
            {0: "lecturer", 1: "training_program", 2: "facility", 3: "others"},
        )
        split = str(row.get("split", "manual")).strip().lower()
        if split not in SPLIT_MAP:
            split = "manual"
        source = str(row.get("source", "import")).strip() or "import"
        try:
            document = document_from_payload(
                FeedbackCreate(
                    sentence=sentence,
                    sentiment_label=sentiment,
                    topic_label=topic,
                    source=source,
                    split=split,
                )
            )
            if upsert_feedback(document):
                imported += 1
            else:
                skipped += 1
        except (ModelNotReadyError, DatabaseUnavailable) as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        except Exception as exc:
            failed += 1
            if len(errors) < 5:
                errors.append(f"Dòng {row_number + 2}: {exc}")
    return {
        "success": True,
        "total_rows": len(frame),
        "imported": imported,
        "skipped": skipped,
        "failed": failed,
        "errors": errors,
    }


@app.get("/api/export")
async def api_export(
    sentiment: str | None = None,
    topic: str | None = None,
    split: str | None = None,
    search: str | None = None,
):
    try:
        documents, _ = get_all_feedbacks(
            skip=0,
            limit=100_000,
            sentiment_filter=sentiment,
            topic_filter=topic,
            split_filter=split,
            search=search,
        )
    except DatabaseUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"MongoDB chưa sẵn sàng: {exc}")
    rows = [
        {
            "sentence": item.get("sentence"),
            "sentiment": item.get("sentiment", {}).get("label"),
            "topic": item.get("topic", {}).get("label"),
            "processed_text": item.get("processed_text"),
            "keywords": ", ".join(item.get("keywords", [])),
            "source": item.get("source"),
            "split": item.get("split"),
            "created_at": item.get("created_at"),
        }
        for item in documents
    ]
    buffer = io.BytesIO()
    pd.DataFrame(rows).to_csv(buffer, index=False, encoding="utf-8-sig")
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": (
                'attachment; filename="sfas_feedbacks_utf8.csv"'
            )
        },
    )
