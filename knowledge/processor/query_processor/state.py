"""查询流程状态：GraphState 字段定义与默认状态构造。"""
import dataclasses
from typing import TypedDict, List
import copy


class QueryGraphState(TypedDict):
    """查询图在各节点间传递的状态字典。"""
    session_id: str  # 会话ID
    task_id: str  # 任务ID
    message_id: str  # 消息ID
    original_query: str  # 原始查询
    embedding_chunks: list  # 向量检索命中的 chunks
    hyde_embedding_chunks: list  # HyDE 检索命中的 chunks
    rrf_chunks: list  # RRF 融合后的 chunks
    web_search_docs: list  # 网络检索结果
    reranked_docs: list  # 精排后的文档
    prompt: str  # 组装好的提示词
    answer: str  # 最终答案
    item_names: List[str]  # 对齐确认的商品名
    rewritten_query: str  # 大模型改写后的查询
    history: list  # 历史对话
    is_stream: bool  # 是否流式输出


# 默认状态：各字段初始空值，节点按需填充
DEFAULT_STATE: QueryGraphState = {
    "session_id": "",
    "task_id": "",
    "message_id": "",
    "original_query": "",
    "embedding_chunks": [],
    "hyde_embedding_chunks": [],
    "rrf_chunks": [],
    "web_search_docs": [],
    "reranked_docs": [],
    "prompt": "",
    "answer": "",
    "item_names": [],
    "rewritten_query": "",
    "history": [],
    "is_stream": False,
}


def create_default_state(**overrides) -> QueryGraphState:
    """创建默认状态并按 overrides 覆盖字段。"""
    state = copy.deepcopy(DEFAULT_STATE)
    state.update(overrides)
    return state


def get_default_state() -> QueryGraphState:
    """返回默认状态副本，避免改动全局默认值。"""
    return copy.deepcopy(DEFAULT_STATE)


graph_default_state = DEFAULT_STATE
