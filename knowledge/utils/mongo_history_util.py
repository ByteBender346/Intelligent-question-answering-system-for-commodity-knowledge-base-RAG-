"""会话历史读写：以 session_id 为单位在 MongoDB 中存取问答消息。"""

import logging
from typing import List, Dict, Any
from datetime import datetime
from bson import ObjectId
from pymongo.collection import Collection
from pymongo import DESCENDING

from knowledge.utils.clients.storage_clients import StorageClients

logger = logging.getLogger(__name__)


def _get_collection() -> Collection:
    """获取 chat_message 集合。"""
    return StorageClients.get_mongo_db()["chat_message"]


def save_chat_message(
        session_id: str,
        role: str,
        text: str,
        rewritten_query: str = "",
        item_names: List[str] = None,
        message_id: str = None,
) -> str:
    """
    写入一条会话消息：message_id 为空则新增，否则按 _id 更新。

    Returns:
        消息 id（新增时为新生成的 ObjectId 字符串，更新时为入参 message_id）
    """
    ts = datetime.now().timestamp()

    document = {
        "session_id": session_id,
        "role": role,
        "text": text,
        "rewritten_query": rewritten_query or "",
        "item_names": item_names or [],
        "ts": ts,
    }

    collection = _get_collection()
    if message_id:
        collection.update_one(
            {"_id": ObjectId(message_id)},
            {"$set": document},
        )
        return message_id
    else:
        result = collection.insert_one(document)
        return str(result.inserted_id)


def get_recent_messages(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """按时间倒序拉取某会话最近的消息，异常时返回空列表。"""
    try:
        cursor = (
            _get_collection()
            .find({"session_id": session_id})
            .sort("ts", DESCENDING)
            .limit(limit)
        )
        return list(cursor)
    except Exception as e:
        logger.error(f"Error getting recent messages: {e}")
        return []


def clear_history(session_id: str) -> int:
    """清空某会话全部历史消息，返回删除条数。"""
    try:
        result = _get_collection().delete_many({"session_id": session_id})
        logger.info(f"Deleted {result.deleted_count} messages for session {session_id}")
        return result.deleted_count
    except Exception as e:
        logger.error(f"Error clearing history for session {session_id}: {e}")
        return 0
