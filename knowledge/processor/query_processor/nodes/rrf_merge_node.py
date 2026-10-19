from typing import List, Any, Dict, Tuple

from knowledge.processor.query_processor.base import BaseNode, T
from knowledge.processor.query_processor.state import QueryGraphState


class RrfMergeNode(BaseNode):
    """RRF 融合节点：将向量检索与 HyDE 检索两路结果按倒数排名融合，统一排序后输出。

    RRF 公式：某文档得分 = Σ 权重 / (k + 该路排名)。
    只依赖排名而非各路原始分数，避免不同检索通道分数尺度不一致；k（默认 60）平滑常数，
    权重 weight 用于区分各路贡献，本项目两路均为 1.0。
    """
    name = "rrf_merge_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        vector_search_chunks = state.get("embedding_chunks") or []
        hyde_search_chunks = state.get("hyde_embedding_chunks") or []

        embedding_chunks = self._format_doc(vector_search_chunks)
        hyde_embedding_chunks = self._format_doc(hyde_search_chunks)

        # 两路结果等权参与融合
        rrf_inputs = [(embedding_chunks, 1.0), (hyde_embedding_chunks, 1.0)]

        rrf_merged_result = self._rrf_merge(rrf_inputs, self.config.rrf_k, self.config.rrf_max_results)

        state['rrf_chunks'] = rrf_merged_result
        return state

    def _format_doc(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Milvus 原始命中 → {chunk_id,title,content} 规整列表。"""
        formated_chunks = []
        for chunk in chunks:
            if not chunk or not isinstance(chunk, dict):
                continue
            chunk_id = chunk.get("id")
            entity = chunk.get("entity")
            if not entity or not isinstance(entity, dict):
                continue
            title = entity.get("title", "")
            content = entity.get("content", "")

            if not content:
                continue

            formated_chunks.append({"chunk_id": chunk_id, "title": title, "content": content})
        return formated_chunks

    def _rrf_merge(self, rrf_inputs: List[Tuple[List[Dict[str, Any]], float]], k: int, rrf_max_results: int) -> List[Dict[str, Any]]:
        """对各路结果累计 RRF 得分，按总分降序取前 rrf_max_results。"""
        # chunk_id → 累计 RRF 得分
        chunk_score = {}
        # chunk_id → 文档内容（随得分更新）
        chunk_data = {}
        for rrf_input in rrf_inputs:
            chunks, weight = rrf_input
            for rank, chunk in enumerate(chunks, 1):
                chunk_id = chunk.get("chunk_id")
                # RRF 核心公式：weight / (k + rank)
                chunk_score[chunk_id] = chunk_score.get(chunk_id, 0) + weight / (k + rank)
                chunk_data[chunk_id] = {**chunk, "score": chunk_score[chunk_id]}

        result_chunks = chunk_data.values()
        sorted_chunks = sorted(result_chunks, key=lambda x: x["score"], reverse=True)
        return sorted_chunks[:rrf_max_results] if rrf_max_results else sorted_chunks
