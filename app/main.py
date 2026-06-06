from __future__ import annotations

import io
import json
import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from bson import ObjectId
from fastapi import FastAPI, Form, Request, UploadFile, File, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import jinja2

from app.cleaning import clean_text
from app.db import (
    delete_feedback,
    get_all_feedbacks,
    get_feedback_by_id,
    get_stats,
    insert_feedback,
    update_feedback,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"
TEMPLATE_DIR = ROOT_DIR / "app" / "templates"

models: dict = {}

SENTIMENT_MAP = {"negative": "Tiêu cực", "neutral": "Trung tính", "positive": "Tích cực"}
TOPIC_MAP = {
    "lecturer": "Giảng viên",
    "training_program": "Chương trình đào tạo",
    "facility": "Cơ sở vật chất",
    "others": "Khác",
    "academic": "Học thuật",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open(MODEL_DIR / "tfidf_vectorizer.pkl", "rb") as f:
        models["tfidf"] = pickle.load(f)
    with open(MODEL_DIR / "lr_model.pkl", "rb") as f:
        models["lr"] = pickle.load(f)
    with open(MODEL_DIR / "nb_model.pkl", "rb") as f:
        models["nb"] = pickle.load(f)
    print("Models loaded successfully")
    yield
    models.clear()


app = FastAPI(title="SFAS", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(ROOT_DIR / "app" / "static")), name="static")

jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape=jinja2.select_autoescape(),
)


def render_template(template_name: str, **context: Any) -> HTMLResponse:
    template = jinja_env.get_template(template_name)
    content = template.render(**context)
    return HTMLResponse(content)


def predict_sentiment(text: str) -> dict:
    cleaned = clean_text(text, do_segment=True)
    X = models["tfidf"].transform([cleaned])
    lr_pred = models["lr"].predict(X)[0]
    lr_proba = models["lr"].predict_proba(X)[0]
    classes = models["lr"].classes_
    lr_confidence = {cls: round(float(prob) * 100, 1) for cls, prob in zip(classes, lr_proba)}
    return {
        "text": text,
        "cleaned": cleaned,
        "sentiment": lr_pred,
        "sentiment_vi": SENTIMENT_MAP.get(lr_pred, lr_pred),
        "confidence": lr_confidence,
        "max_confidence": round(max(lr_proba) * 100, 1),
    }


# ─── PAGES ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(message: str | None = None):
    stats = get_stats()
    return render_template("index.html", stats=stats, message=message, result=None, text="")


@app.post("/predict", response_class=HTMLResponse)
async def predict(text: str = Form(...)):
    if not text.strip():
        return RedirectResponse(url="/?message=Vui+long+nhap+van+ban", status_code=302)
    result = predict_sentiment(text.strip())
    insert_feedback(
        text=text.strip(),
        sentiment=result["sentiment"],
        confidence=result["max_confidence"],
        source="predict",
    )
    stats = get_stats()
    return render_template("index.html", stats=stats, message=None, result=result, text=text.strip())


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    stats = get_stats()
    return render_template("dashboard.html", stats=stats)


@app.get("/feedbacks", response_class=HTMLResponse)
async def feedback_list(
    page: int = Query(1, ge=1),
    sentiment: str | None = Query(None),
    search: str | None = Query(None),
):
    per_page = 20
    skip = (page - 1) * per_page
    docs, total = get_all_feedbacks(
        skip=skip, limit=per_page,
        sentiment_filter=sentiment, search=search,
    )
    total_pages = max(1, (total + per_page - 1) // per_page)
    return render_template(
        "list.html",
        feedbacks=docs, page=page, total_pages=total_pages, total=total,
        sentiment_filter=sentiment, search=search,
        SENTIMENT_MAP=SENTIMENT_MAP, TOPIC_MAP=TOPIC_MAP,
    )


# ─── CRUD API ─────────────────────────────────────────────────────────────────

@app.post("/feedback/add")
async def add_feedback(text: str = Form(...)):
    result = predict_sentiment(text.strip())
    insert_feedback(
        text=text.strip(), sentiment=result["sentiment"],
        confidence=result["max_confidence"], source="manual",
    )
    return RedirectResponse(url="/feedbacks", status_code=302)


@app.post("/feedback/{feedback_id}/update")
async def edit_feedback(feedback_id: str, text: str = Form(...), sentiment: str = Form(None)):
    update_fields: dict = {"text": text.strip()}
    if sentiment and sentiment in SENTIMENT_MAP:
        update_fields["sentiment"] = sentiment
    update_feedback(feedback_id, update_fields)
    return RedirectResponse(url="/feedbacks", status_code=302)


@app.post("/feedback/{feedback_id}/delete")
async def remove_feedback(feedback_id: str):
    delete_feedback(feedback_id)
    return RedirectResponse(url="/feedbacks", status_code=302)


# ─── API ─────────────────────────────────────────────────────────────────────

@app.get("/api/stats")
async def api_stats():
    return get_stats()


@app.get("/api/predict")
async def api_predict(text: str = Query(...)):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
    return predict_sentiment(text.strip())


@app.post("/api/import")
async def api_import(file: UploadFile = File(...)):
    content = await file.read()
    try:
        if file.filename and file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif file.filename and file.filename.endswith(".parquet"):
            df = pd.read_parquet(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="CSV or Parquet only")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Parse error: {e}")

    text_col = None
    for col in ["text", "text_clean", "clean_text", "sentence", "feedback", "comment"]:
        if col in df.columns:
            text_col = col
            break
    if text_col is None:
        raise HTTPException(status_code=400, detail="No text column found")

    imported = 0
    for _, row in df.iterrows():
        text_val = str(row[text_col]).strip()
        if not text_val:
            continue
        result = predict_sentiment(text_val)
        insert_feedback(
            text=text_val, sentiment=result["sentiment"],
            confidence=result["max_confidence"], source="import",
        )
        imported += 1
    return {"imported": imported, "total_rows": len(df)}


@app.get("/api/export")
async def api_export():
    docs, total = get_all_feedbacks(skip=0, limit=10000)
    for doc in docs:
        doc.pop("_id", None)
    df = pd.DataFrame(docs)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, encoding="utf-8-sig")
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sfas_feedbacks.csv"},
    )
