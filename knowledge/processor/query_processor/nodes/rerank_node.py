import math
from numbers import Real
from typing import Any, Dict, List
from knowledge.processor.query_processor.base import BaseNode, T
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.utils.clients.ai_clients import AIClients


class RerankNode(BaseNode):
    """精排节点：合并本地 RRF 结果与网络检索结果，用 BGE-Reranker 重算相关性分数，再按断崖动态截断。"""
    name = "rerank_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        user_query = state.get("rewritten_query")
        # 合并本地 RRF 结果与网络检索结果
        rrf_docs = state.get("rrf_chunks") or []
        web_search_docs = state.get("web_search_docs") or []
        final_docs = self._format_rrf_docs(rrf_docs)
        final_docs.extend(self._format_web_docs(web_search_docs))
        # 调用 BGE-Reranker 计算 Q-D 相关性得分
        doc_score = self._refine_rerank(user_query, final_docs)
        sorted_doc_score = sorted(doc_score, key=lambda x: x["score"], reverse=True)

        # 按断崖检测动态截断（min_top_k ~ max_top_k 区间内找最大分差点）
        reranked_docs = self._cliff_cutoff(sorted_doc_score, self.config.rerank_min_top_k, self.config.rerank_max_top_k, self.config.rerank_gap_abs)

        state["reranked_docs"] = reranked_docs
        return state

    def _format_rrf_docs(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """规整本地检索结果，统一为 {chunk_id,title,content,source=local}。"""
        formated_docs = []
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            chunk_id = doc.get("chunk_id")
            title = doc.get("title")
            content = doc.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            formated_docs.append({
                "chunk_id": chunk_id,
                "title": title,
                "content": content,
                "source": "local"
            })
        return formated_docs

    def _format_web_docs(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """规整网络检索结果，统一为 {title,content,url,source=web}。"""
        formated_docs = []
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            title = doc.get("title")
            content = doc.get("snippet")
            if not isinstance(content, str) or not content.strip():
                continue
            url = doc.get("url")
            formated_docs.append({
                "title": title,
                "content": content,
                "url": url,
                "source": "web"
            })
        return formated_docs

    @staticmethod
    def _sigmoid(score: float) -> float:
        """sigmoid 归一化，将原始相关分映射到 (0, 1)。"""
        if score >= 0:
            return 1.0 / (1.0 + math.exp(-score))
        exp_score = math.exp(score)
        return exp_score / (1.0 + exp_score)

    def _refine_rerank(self, user_query: str, final_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """构造 Q-D 对调用 Reranker，分数经 sigmoid 归一化；模型不可用或调用失败时保留原顺序（score=0）。"""
        if not final_docs:
            return []
        fallback_docs = [{**doc, "score": 0.0} for doc in final_docs]
        question_document_pairs = [(user_query, doc.get("content")) for doc in final_docs]
        try:
            rerank_client = AIClients.get_bge_m3_rerank_client()
        except Exception as e:
            self.logger.warning("获取Rerank模型对象失败，保留原检索顺序: %s", e, exc_info=True)
            return fallback_docs
        try:
            scores = rerank_client.compute_score(question_document_pairs)
            scores = [scores] if isinstance(scores, Real) else list(scores)
            if len(scores) != len(final_docs):
                raise ValueError("Rerank模型返回的分数数量与文档数量不一致")
            scores = [float(score) for score in scores]
            if not all(math.isfinite(score) for score in scores):
                raise ValueError("Rerank模型返回了非有限分数")
            return [{**doc, "score": self._sigmoid(score)} for doc, score in zip(final_docs, scores)]
        except Exception as e:
            self.logger.warning("调用Rerank模型失败，保留原检索顺序: %s", e, exc_info=True)
            return fallback_docs

    def _cliff_cutoff(self, sorted_doc_score: List[Dict[str, Any]], rerank_min_top_k: int, rerank_max_top_k: int, rerank_gap_abs: float) -> List[Dict[str, Any]]:
        """断崖截断：在 [min_top_k, max_top_k] 区间内找相邻分差最大处作为切分点。"""
        upper_bound = min(rerank_max_top_k, len(sorted_doc_score))
        lower_bound = min(rerank_min_top_k, upper_bound)

        max_score_gap = 0
        cut_index = upper_bound
        for i in range(max(lower_bound - 1, 0), upper_bound - 1):
            current_score = sorted_doc_score[i].get("score")
            next_score = sorted_doc_score[i + 1].get("score")

            if current_score is None or next_score is None:
                continue
            score_gap = current_score - next_score
            # 分差达到阈值且为目前最大 → 记录为断崖点
            if score_gap >= rerank_gap_abs and score_gap > max_score_gap:
                max_score_gap = score_gap
                cut_index = i + 1

        cut_index = max(lower_bound, cut_index)
        return sorted_doc_score[:cut_index]
