"""Milvus 向量检索工具：组装稠密 + 稀疏混合检索请求，并执行加权融合排序。"""

import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from dotenv import load_dotenv

load_dotenv()

from typing import Optional, List, Tuple, Any, Dict
from pymilvus import MilvusClient, WeightedRanker, AnnSearchRequest

milvus_client: Optional[MilvusClient] = None


def create_hybrid_search_requests(dense_vector,
                                  sparse_vector,
                                  dense_params=None,
                                  sparse_params=None,
                                  expr=None,
                                  expr_params=None,
                                  limit=5) -> List[AnnSearchRequest]:
    """
    创建混合检索请求：分别构造稠密向量与稀疏向量两个 AnnSearchRequest。

    :param dense_vector: 稠密向量
    :param sparse_vector: 稀疏向量
    :param dense_params: 稠密向量搜索参数，默认 COSINE
    :param sparse_params: 稀疏向量搜索参数，默认 IP
    :param expr: 标量过滤表达式
    :param expr_params: 表达式变量
    :param limit: 返回结果数量，默认 5
    :return: [dense_req, sparse_req]
    :raises ValueError: 向量参数无效
    :raises RuntimeError: 创建请求失败
    """
    if dense_vector is None or sparse_vector is None:
        raise ValueError("dense_vector 和 sparse_vector 不能为 None")

    try:
        # 默认检索参数：稠密用余弦，稀疏用内积
        if dense_params is None:
            dense_params = {"metric_type": "COSINE"}
        if sparse_params is None:
            sparse_params = {"metric_type": "IP"}

        # 稠密向量检索请求
        dense_req = AnnSearchRequest(
            data=[dense_vector],
            anns_field="dense_vector",
            param=dense_params,
            expr=expr,  # 标量过滤条件（表达式写法）
            expr_params=expr_params,
            limit=limit
        )

        # 稀疏向量检索请求
        sparse_req = AnnSearchRequest(
            data=[sparse_vector],
            anns_field="sparse_vector",
            param=sparse_params,
            expr=expr,
            expr_params=expr_params,
            limit=limit
        )

        return [dense_req, sparse_req]
    except Exception as e:
        raise RuntimeError(f"创建混合搜索请求失败: {e}") from e


def execute_hybrid_search_query(milvus_client: MilvusClient,
                                collection_name,
                                search_requests,
                                ranker_weights=(0.5, 0.5),
                                norm_score=True,
                                limit=5,
                                output_fields=None,
                                search_params=None):
    """
    执行混合检索：用 WeightedRanker 对稠密与稀疏两路结果加权融合。

    :param milvus_client: Milvus 客户端
    :param collection_name: 集合名称
    :param search_requests: [dense_req, sparse_req]
    :param ranker_weights: 两路权重，默认各 0.5
    :param norm_score: 是否归一化分数
    :param limit: 返回结果数量，默认 5
    :param output_fields: 返回字段，默认 ["item_name"]
    :param search_params: 搜索参数
    :return: 融合后的搜索结果
    :raises ValueError: 参数无效
    :raises RuntimeError: 搜索执行失败
    """
    if milvus_client is None:
        raise ValueError("milvus_client 不能为 None")
    if search_requests is None or len(search_requests) == 0:
        raise ValueError("search_requests 不能为 None 或空列表")

    try:
        # 加权融合排序器：按权重合并稠密与稀疏两路分数
        rerank = WeightedRanker(ranker_weights[0], ranker_weights[1], norm_score=norm_score)

        # 默认返回商品名字段
        if output_fields is None:
            output_fields = ["item_name"]

        # 执行混合检索
        res = milvus_client.hybrid_search(
            collection_name=collection_name,
            reqs=search_requests,
            ranker=rerank,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params
        )

        total_hits = sum(len(hits) for hits in res) if res else 0
        logger.info(f"Milvus 混合搜索完成，共处理 {len(res) if res else 0} 个查询，总计找到 {total_hits} 个结果")
        return res
    except Exception as e:
        raise RuntimeError(f"执行Milvus混合搜索失败 (collection={collection_name}): {e}") from e


def _item_names_filter(item_names: List[str]) -> Tuple[str, Dict[str, Any]]:
    """构造按商品名过滤的标量表达式与参数。"""
    expr = "item_name in {item_names}"
    expr_params = {"item_names": item_names}
    return expr, expr_params
