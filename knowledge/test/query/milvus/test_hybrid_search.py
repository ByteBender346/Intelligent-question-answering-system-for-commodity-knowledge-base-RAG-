"""Milvus 混合检索离线测试：BGE-M3 稠密+稀疏双路召回 + WeightedRanker 融合。"""

from typing import List

from pymilvus import AnnSearchRequest, WeightedRanker
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from knowledge.utils.clients.ai_clients import AIClients
from knowledge.utils.clients.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors


def get_beg_m3_embedding_model():
    """获取 BGE-M3 嵌入模型实例"""
    bge_m3_ef = AIClients.get_bge_m3_client()
    return bge_m3_ef


def generate_hybrid_embeddings(bge_m3_embedding_model:BGEM3EmbeddingFunction,document_list:List[str]):
    embedding_result = generate_bge_m3_hybrid_vectors(bge_m3_embedding_model, document_list)
    return embedding_result

def create_hybrid_search_requests(
    dense_vector,
    sparse_vector,
    dense_params=None,
    sparse_params=None,
    expr=None,
    limit=5
):
    """
    创建混合检索请求

    Args:
        dense_vector: 稠密向量 [0.1, 0.2, ...]
        sparse_vector: 稀疏向量 {token_id: weight}
        limit: 每路检索返回数量

    Returns:
        [dense_req, sparse_req] 检索请求列表
    """
    # 默认参数：稠密用余弦，稀疏用内积
    if dense_params is None:
        dense_params = {"metric_type": "COSINE"}
    if sparse_params is None:
        sparse_params = {"metric_type": "IP"}

    # 创建稠密向量检索请求
    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",      # Collection 中的稠密向量字段
        param=dense_params,
        expr=expr,                       # 过滤表达式（可选）
        limit=limit
    )

    # 创建稀疏向量检索请求
    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",      # Collection 中的稀疏向量字段
        param=sparse_params,
        expr=expr,
        limit=limit
    )

    return [dense_req, sparse_req]

def execute_hybrid_search_query(
    milvus_client,
    collection_name,
    search_requests,
    ranker_weights=(0.5, 0.5),
    norm_score=True,
    limit=5,
    output_fields=None
):
    """
    执行混合检索

    Args:
        collection_name: 集合名称
        search_requests: 检索请求 [dense_req, sparse_req]
        ranker_weights: 融合权重 (稠密权重, 稀疏权重)
        norm_score: 是否归一化分数
        limit: 最终返回数量
        output_fields: 返回字段

    Returns:
        检索结果 [[hit1, hit2, ...], ...]
    """
    # 创建融合排序器
    rerank = WeightedRanker(
        ranker_weights[0],    # 稠密向量权重
        ranker_weights[1],    # 稀疏向量权重
        norm_score=norm_score
    )

    # 执行混合检索
    results = milvus_client.hybrid_search(
        collection_name=collection_name,
        reqs=search_requests,      # 检索请求列表
        ranker=rerank,              # 融合排序器
        limit=limit,
        output_fields=output_fields
    )

    return results
