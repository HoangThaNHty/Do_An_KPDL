from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import DuplicateKeyError, PyMongoError

from app.config import settings


SENTIMENT_CODES = {"negative": 0, "neutral": 1, "positive": 2}
TOPIC_CODES = {"lecturer": 0, "training_program": 1, "facility": 2, "others": 3}
DATABASE_ERROR_MESSAGE = (
    "Không thể kết nối MongoDB. Hãy bật dịch vụ MongoDB và kiểm tra "
    "MONGO_URI trong file .env"
)


class DatabaseUnavailable(RuntimeError):
    def __init__(self, detail: str | None = None) -> None:
        super().__init__(DATABASE_ERROR_MESSAGE)


_client: MongoClient | None = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(
            settings.mongo_uri,
            serverSelectionTimeoutMS=settings.mongo_timeout_ms,
            connectTimeoutMS=settings.mongo_timeout_ms,
        )
    return _client


def get_collection():
    return get_client()[settings.mongo_db][settings.feedback_collection]


def check_connection() -> tuple[bool, str | None]:
    try:
        get_client().admin.command("ping")
        return True, None
    except PyMongoError:
        return False, DATABASE_ERROR_MESSAGE


def init_database() -> None:
    collection = get_collection()
    try:
        _migrate_legacy_documents(collection)
        collection.create_index([("sentence", ASCENDING)])
        collection.create_index([("sentiment.label", ASCENDING)])
        collection.create_index([("topic.label", ASCENDING)])
        collection.create_index([("split", ASCENDING)])
        collection.create_index([("created_at", DESCENDING)])
        collection.create_index(
            [("feedback_key", ASCENDING)],
            unique=True,
            sparse=True,
        )
    except PyMongoError as exc:
        raise DatabaseUnavailable(str(exc)) from exc


def _migrate_legacy_documents(collection) -> None:
    legacy_query = {
        "$or": [
            {"sentence": {"$exists": False}},
            {"sentiment.label": {"$exists": False}},
            {"topic.label": {"$exists": False}},
        ]
    }
    for document in collection.find(legacy_query):
        sentence = str(document.get("sentence") or document.get("text") or "").strip()
        if not sentence:
            continue
        sentiment_value = document.get("sentiment", "neutral")
        sentiment_label = (
            sentiment_value.get("label")
            if isinstance(sentiment_value, dict)
            else sentiment_value
        )
        if sentiment_label not in SENTIMENT_CODES:
            sentiment_label = "neutral"
        topic_value = document.get("topic", "others")
        topic_label = topic_value.get("label") if isinstance(topic_value, dict) else topic_value
        if topic_label not in TOPIC_CODES:
            topic_label = "others"
        split = document.get("split") or "manual"
        created_at = document.get("created_at") or document.get("timestamp")
        if not isinstance(created_at, datetime):
            created_at = datetime.now(timezone.utc)
        confidence = float(document.get("confidence") or 0.0)
        if confidence > 1:
            confidence /= 100
        normalized = " ".join(sentence.lower().split())
        legacy_key = hashlib.sha256(
            f"{normalized}|{split}|legacy:{document['_id']}".encode("utf-8")
        ).hexdigest()
        collection.update_one(
            {"_id": document["_id"]},
            {
                "$set": {
                    "sentence": sentence,
                    "sentiment": {
                        "label": sentiment_label,
                        "code": SENTIMENT_CODES[sentiment_label],
                    },
                    "topic": {
                        "label": topic_label,
                        "code": TOPIC_CODES[topic_label],
                    },
                    "processed_text": document.get("processed_text")
                    or document.get("cleaned")
                    or sentence.lower(),
                    "keywords": document.get("keywords") or [],
                    "source": document.get("source") or "legacy",
                    "split": split,
                    "prediction": document.get("prediction")
                    or {
                        "sentiment_label": sentiment_label,
                        "sentiment_confidence": confidence,
                        "topic_label": topic_label,
                        "topic_confidence": 0.0,
                    },
                    "feedback_key": document.get("feedback_key") or legacy_key,
                    "created_at": created_at,
                    "updated_at": document.get("updated_at") or created_at,
                }
            },
        )


def _feedback_key(sentence: str, split: str) -> str:
    normalized = " ".join(sentence.lower().split())
    return hashlib.sha256(f"{normalized}|{split}".encode("utf-8")).hexdigest()


def build_feedback_document(
    sentence: str,
    sentiment_label: str,
    topic_label: str,
    processed_text: str,
    keywords: list[str] | None = None,
    source: str = "manual",
    split: str = "manual",
    prediction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    prediction = prediction or {}
    return {
        "sentence": sentence.strip(),
        "sentiment": {
            "label": sentiment_label,
            "code": SENTIMENT_CODES[sentiment_label],
        },
        "topic": {
            "label": topic_label,
            "code": TOPIC_CODES[topic_label],
        },
        "processed_text": processed_text,
        "keywords": keywords or [],
        "source": source,
        "split": split,
        "prediction": {
            "sentiment_label": prediction.get("sentiment_label", sentiment_label),
            "sentiment_confidence": float(
                prediction.get("sentiment_confidence", 0.0)
            ),
            "topic_label": prediction.get("topic_label", topic_label),
            "topic_confidence": float(prediction.get("topic_confidence", 0.0)),
        },
        "feedback_key": _feedback_key(sentence, split),
        "created_at": now,
        "updated_at": now,
    }


def _serialize(document: dict[str, Any] | None) -> dict[str, Any] | None:
    if document is None:
        return None
    output = dict(document)
    output["_id"] = str(output["_id"])
    for field in ("created_at", "updated_at"):
        if isinstance(output.get(field), datetime):
            output[field] = output[field].isoformat()
    return output


def insert_feedback(document: dict[str, Any]) -> str:
    try:
        result = get_collection().insert_one(document)
        return str(result.inserted_id)
    except DuplicateKeyError as exc:
        raise ValueError("Phản hồi này đã tồn tại trong cùng tập dữ liệu") from exc
    except PyMongoError as exc:
        raise DatabaseUnavailable(str(exc)) from exc


def upsert_feedback(document: dict[str, Any]) -> bool:
    try:
        result = get_collection().update_one(
            {"feedback_key": document["feedback_key"]},
            {"$setOnInsert": document},
            upsert=True,
        )
        return result.upserted_id is not None
    except PyMongoError as exc:
        raise DatabaseUnavailable(str(exc)) from exc


def get_all_feedbacks(
    skip: int = 0,
    limit: int = 50,
    sentiment_filter: str | None = None,
    topic_filter: str | None = None,
    split_filter: str | None = None,
    search: str | None = None,
    sort_order: str = "desc",
) -> tuple[list[dict[str, Any]], int]:
    query: dict[str, Any] = {}
    if sentiment_filter:
        query["sentiment.label"] = sentiment_filter
    if topic_filter:
        query["topic.label"] = topic_filter
    if split_filter:
        query["split"] = split_filter
    if search:
        query["sentence"] = {"$regex": re.escape(search), "$options": "i"}
    sort_direction = ASCENDING if sort_order == "asc" else DESCENDING
    try:
        collection = get_collection()
        total = collection.count_documents(query)
        cursor = (
            collection.find(query)
            .sort("created_at", sort_direction)
            .skip(skip)
            .limit(limit)
        )
        return [_serialize(document) for document in cursor], total
    except PyMongoError as exc:
        raise DatabaseUnavailable(str(exc)) from exc


def get_feedback_by_id(feedback_id: str) -> dict[str, Any] | None:
    if not ObjectId.is_valid(feedback_id):
        return None
    try:
        return _serialize(get_collection().find_one({"_id": ObjectId(feedback_id)}))
    except PyMongoError as exc:
        raise DatabaseUnavailable(str(exc)) from exc


def update_feedback(feedback_id: str, update_fields: dict[str, Any]) -> bool:
    if not ObjectId.is_valid(feedback_id):
        return False
    update_fields = dict(update_fields)
    update_fields["updated_at"] = datetime.now(timezone.utc)
    if "sentence" in update_fields or "split" in update_fields:
        current = get_feedback_by_id(feedback_id)
        if current is None:
            return False
        sentence = update_fields.get("sentence", current["sentence"])
        split = update_fields.get("split", current["split"])
        update_fields["feedback_key"] = _feedback_key(sentence, split)
    try:
        result = get_collection().update_one(
            {"_id": ObjectId(feedback_id)},
            {"$set": update_fields},
        )
        return result.matched_count > 0
    except DuplicateKeyError as exc:
        raise ValueError("Phản hồi trùng với dữ liệu đã có") from exc
    except PyMongoError as exc:
        raise DatabaseUnavailable(str(exc)) from exc


def delete_feedback(feedback_id: str) -> bool:
    if not ObjectId.is_valid(feedback_id):
        return False
    try:
        result = get_collection().delete_one({"_id": ObjectId(feedback_id)})
        return result.deleted_count > 0
    except PyMongoError as exc:
        raise DatabaseUnavailable(str(exc)) from exc


def _distribution(field: str) -> dict[str, int]:
    pipeline = [
        {"$match": {field: {"$ne": None}}},
        {"$group": {"_id": f"${field}", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    return {
        str(item["_id"]): int(item["count"])
        for item in get_collection().aggregate(pipeline)
    }


def get_stats() -> dict[str, Any]:
    try:
        collection = get_collection()
        sentiment = _distribution("sentiment.label")
        topic = _distribution("topic.label")
        split = _distribution("split")
        data_origin = {
            "uit_vsfc": collection.count_documents(
                {"split": {"$in": ["train", "validation", "test"]}}
            ),
            "manual_demo": collection.count_documents({"split": "manual"}),
        }
        negative_by_topic = {
            str(item["_id"]): int(item["count"])
            for item in collection.aggregate(
                [
                    {"$match": {"sentiment.label": "negative"}},
                    {"$group": {"_id": "$topic.label", "count": {"$sum": 1}}},
                    {"$sort": {"count": -1}},
                ]
            )
            if item["_id"]
        }
        recent_negative = [
            _serialize(document)
            for document in collection.find({"sentiment.label": "negative"})
            .sort("created_at", DESCENDING)
            .limit(5)
        ]
        recommendations = []
        recommendation_text = {
            "facility": "Cần ưu tiên xem xét cơ sở vật chất.",
            "lecturer": (
                "Cần xem xét chất lượng và phương pháp giảng dạy của giảng viên."
            ),
            "training_program": "Cần rà soát chương trình đào tạo.",
            "others": "Cần phân tích thêm các phản hồi tiêu cực thuộc nhóm khác.",
        }
        for label, count in sorted(
            negative_by_topic.items(), key=lambda item: item[1], reverse=True
        )[:3]:
            recommendations.append(
                {"topic": label, "count": count, "message": recommendation_text[label]}
            )
        return {
            "total": collection.count_documents({}),
            "sentiment": sentiment,
            "topic": topic,
            "split": split,
            "data_origin": data_origin,
            "negative_by_topic": negative_by_topic,
            "recent_negative": recent_negative,
            "recommendations": recommendations,
        }
    except PyMongoError as exc:
        raise DatabaseUnavailable(str(exc)) from exc


def empty_stats() -> dict[str, Any]:
    return {
        "total": 0,
        "sentiment": {},
        "topic": {},
        "split": {},
        "data_origin": {"uit_vsfc": 0, "manual_demo": 0},
        "negative_by_topic": {},
        "recent_negative": [],
        "recommendations": [],
    }


def get_top_keywords_by_sentiment(sentiment_label: str, limit: int = 15) -> list[dict[str, Any]]:
    try:
        collection = get_collection()
        pipeline = [
            {"$match": {"sentiment.label": sentiment_label, "keywords": {"$exists": True, "$ne": []}}},
            {"$unwind": "$keywords"},
            {"$group": {"_id": "$keywords", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": limit}
        ]
        return [{"keyword": str(item["_id"]), "count": int(item["count"])} for item in collection.aggregate(pipeline)]
    except PyMongoError as exc:
        raise DatabaseUnavailable(str(exc)) from exc
