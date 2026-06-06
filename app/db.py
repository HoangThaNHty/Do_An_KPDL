from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient, DESCENDING

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT_DIR, ".env"))

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB", "sfas")

_client: Optional[MongoClient] = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URI)
    return _client


def get_db():
    return get_client()[DB_NAME]


def get_collection():
    return get_db()["feedbacks"]


def insert_feedback(
    text: str,
    sentiment: str,
    topic: str | None = None,
    confidence: float | None = None,
    source: str = "manual",
) -> str:
    doc = {
        "text": text,
        "sentiment": sentiment,
        "topic": topic,
        "confidence": confidence,
        "source": source,
        "timestamp": datetime.now(timezone.utc),
    }
    result = get_collection().insert_one(doc)
    return str(result.inserted_id)


def get_all_feedbacks(
    skip: int = 0,
    limit: int = 50,
    sentiment_filter: str | None = None,
    topic_filter: str | None = None,
    search: str | None = None,
) -> tuple[list[dict], int]:
    query: dict = {}
    if sentiment_filter:
        query["sentiment"] = sentiment_filter
    if topic_filter:
        query["topic"] = topic_filter
    if search:
        query["text"] = {"$regex": search, "$options": "i"}

    coll = get_collection()
    total = coll.count_documents(query)
    cursor = (
        coll.find(query)
        .sort("timestamp", DESCENDING)
        .skip(skip)
        .limit(limit)
    )
    docs = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        if isinstance(doc.get("timestamp"), datetime):
            doc["timestamp"] = doc["timestamp"].isoformat()
        docs.append(doc)
    return docs, total


def get_feedback_by_id(feedback_id: str) -> dict | None:
    try:
        doc = get_collection().find_one({"_id": ObjectId(feedback_id)})
    except Exception:
        return None
    if doc is None:
        return None
    doc["_id"] = str(doc["_id"])
    if isinstance(doc.get("timestamp"), datetime):
        doc["timestamp"] = doc["timestamp"].isoformat()
    return doc


def update_feedback(feedback_id: str, update_fields: dict) -> bool:
    try:
        result = get_collection().update_one(
            {"_id": ObjectId(feedback_id)},
            {"$set": update_fields},
        )
    except Exception:
        return False
    return result.modified_count > 0


def delete_feedback(feedback_id: str) -> bool:
    try:
        result = get_collection().delete_one({"_id": ObjectId(feedback_id)})
    except Exception:
        return False
    return result.deleted_count > 0


def get_stats() -> dict:
    coll = get_collection()

    pipeline_sentiment = [
        {"$group": {"_id": "$sentiment", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    sentiment_dist = {d["_id"]: d["count"] for d in coll.aggregate(pipeline_sentiment)}

    pipeline_topic = [
        {"$match": {"topic": {"$ne": None}}},
        {"$group": {"_id": "$topic", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    topic_dist = {d["_id"]: d["count"] for d in coll.aggregate(pipeline_topic)}

    pipeline_trend = [
        {
            "$group": {
                "_id": {
                    "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}},
                    "sentiment": "$sentiment",
                },
                "count": {"$sum": 1},
            }
        },
        {"$sort": {"_id.date": 1}},
    ]
    trend_raw = list(coll.aggregate(pipeline_trend))
    trend: dict[str, dict[str, int]] = {}
    for item in trend_raw:
        date = item["_id"]["date"]
        sentiment = item["_id"]["sentiment"]
        if date not in trend:
            trend[date] = {}
        trend[date][sentiment] = item["count"]

    total = coll.count_documents({})

    return {
        "total": total,
        "sentiment": sentiment_dist,
        "topic": topic_dist,
        "trend": trend,
    }
