import json
import re
from logging import Logger
from typing import Dict, Any, List, Tuple

from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.config import QueryConfig
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.prompts.query_prompt import ITEM_NAME_SYSTEM_EXTRACT_TEMPLATE, ITEM_NAME_USER_EXTRACT_TEMPLATE
from knowledge.utils.clients.ai_clients import AIClients
from knowledge.utils.clients.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query, milvus_client
from knowledge.utils.mongo_history_util import get_recent_messages


class _ItemNameAligner:
    """商品名对齐器：将大模型提取的商品名到 Milvus 商品名库做混合检索对齐，产出 confirmed / options 两个容器。"""

    def __init__(self, logger: Logger, config: QueryConfig):
        self._logger = logger
        self._config = config

    def search_and_align(self, item_names: List[str]) -> Tuple[List[str], List[str]]:
        # 商品名向量化后到商品名库混合检索
        search_result = self._search_vector(item_names)
        if not search_result:
            return [], []
        # 按置信度阈值把命中项分流到 confirmed（确定）与 options（待确认）
        confirmed, options = self._align(search_result)

        # 分数差异化过滤：confirmed 多于一个时剔除相对最高分差距过大者
        if len(confirmed) > 1:
            confirmed = self._item_name_score_filter(confirmed, search_result)

        return confirmed, options

    def _search_vector(self, item_names: List[str]):
        """商品名逐一向量化（BGE-M3 稠密+稀疏）后到 item_name_collection 混合检索。

        返回结构：[{"extracted_name": 提取名, "matches": [{"item_name": 库内名, "score": 分数}]}]
        """
        final_search_result = []
        try:
            embedding_client = AIClients.get_bge_m3_client()
        except Exception as e:
            self._logger.error(f"获取嵌入模型失败,{e}")
            return final_search_result

        try:
            embedding_result = generate_bge_m3_hybrid_vectors(embedding_client, item_names)
            dense_vector_list = embedding_result.get("dense")
            sparse_vector_list = embedding_result.get("sparse")
        except Exception as e:
            self._logger.error(f"向量嵌入失败,{e}")
            return final_search_result

        try:
            milvus_client = StorageClients.get_milvus_client()
        except Exception as e:
            self._logger.error(f"获取milvus客户端失败,{e}")
            return final_search_result

        for index, item_name in enumerate(item_names):
            search_requests = create_hybrid_search_requests(
                dense_vector=dense_vector_list[index],
                sparse_vector=sparse_vector_list[index],
                limit=self._config.embedding_search_limit
            )

            milvus_search_results = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=self._config.item_name_collection,
                search_requests=search_requests,
                limit=self._config.embedding_search_limit,
                output_fields=["item_name"]
            )
            matches = []
            for m in milvus_search_results[0]:
                matches.append({
                    "item_name": m["entity"]["item_name"],
                    "score": m["distance"]
                })

            final_search_result.append({
                "extracted_name": item_name,
                "matches": matches
            })

        return final_search_result

    def _align(self, search_result: List[Dict[str, Any]]):
        """按置信度阈值对齐：

        - score ≥ high_confidence 为高置信：与提取名完全一致 / 仅一条高置信 / 最高分领先第二名超过 gap → 入 confirmed；
          多条高置信且分差不足 → 前 max_options 条入 options。
        - 无高置信但 score ≥ mid_confidence 为中置信：前 max_options 条入 options。
        """
        confirmed = []
        options = []
        for result in search_result:
            extracted_name = result.get("extracted_name")
            matches = result.get("matches")
            matches = sorted(matches, key=lambda x: x["score"], reverse=True)
            # 高置信命中
            high_result = [m.get("item_name") for m in matches if m.get("score") >= self._config.item_name_high_confidence]
            if high_result:
                # 与 LLM 提取名完全一致的优先入 confirmed
                same_item_name = [h for h in high_result if h == extracted_name and h not in confirmed]
                if same_item_name:
                    confirmed.extend(same_item_name)
                elif len(high_result) == 1:
                    high = high_result[0]
                    if high not in confirmed:
                        confirmed.append(high)
                else:
                    # 多条高置信：最高分与第二名分差超过 gap 则唯一确定，否则列为待确认选项
                    max_score = matches[0]["score"]
                    if max_score - matches[1]["score"] > self._config.item_name_score_gap:
                        max_item_name = matches[0]["item_name"]
                        if max_item_name not in confirmed:
                            confirmed.append(max_item_name)
                    else:
                        for high in high_result[:self._config.item_name_max_options]:
                            if high not in confirmed and high not in options:
                                options.append(high)

            else:
                # 无高置信：收集中置信结果作为待确认选项
                middle_result = [m.get("item_name") for m in matches if m.get("score") >= self._config.item_name_mid_confidence]
                if middle_result:
                    for mid in middle_result[:self._config.item_name_max_options]:
                        if mid not in confirmed and mid not in options:
                            options.append(mid)

        return confirmed, options[:self._config.item_name_max_options]

    def _item_name_score_filter(self, confirmed: List[str], search_result: List[Dict[str, Any]]):
        """分数差异化过滤：剔除与最高分差距超过 item_name_score_gap 的商品名。"""
        # 构建 商品名 → 最高得分 映射
        item_name_score = {}
        for result in search_result:
            matches = result.get("matches")
            for m in matches:
                item_name = m.get("item_name")
                score = m.get("score")
                item_name_score[item_name] = max(item_name_score.get(item_name, score), score)

        max_score = 0
        for item_name in confirmed:
            max_score = max(max_score, item_name_score[item_name])

        final_confirmed = []
        for item_name in confirmed:
            if max_score - item_name_score[item_name] <= self._config.item_name_score_gap:
                final_confirmed.append(item_name)

        return final_confirmed


class _ItemNameExtractor:
    """商品名提取器：调用大模型从原始查询中抽取商品名并改写查询。"""

    def __init__(self, logger: Logger, name: str):
        self._logger = logger
        self._name = name

    def extract_item_name(self, original_query: str, formatted_history_str: str) -> Dict[str, Any]:
        llm_result = {
            "item_names": [],
            "rewritten_query": original_query
        }

        try:
            llm_client = AIClients.get_llm_client(response_format=True)
        except Exception as e:
            self._logger.error(f"创建大模型对象失败,{e}")
            return llm_result
        system_message = SystemMessage(content=ITEM_NAME_SYSTEM_EXTRACT_TEMPLATE)

        user_prompt = ITEM_NAME_USER_EXTRACT_TEMPLATE.format(
            history_text=formatted_history_str,
            query=original_query
        )
        human_message = HumanMessage(content=user_prompt)
        try:
            llm_result = llm_client.invoke([system_message, human_message])
        except Exception as e:
            self._logger.error(f"调用大模型对象失败,{e}")
            return llm_result
        llm_content = llm_result.content

        if not llm_content:
            return llm_result

        return self._clean_and_parse(llm_content)

    def _clean_and_parse(self, llm_content: str) -> Dict[str, Any]:
        """剥离 Markdown 代码块围栏后解析 JSON，取 item_names 与 rewritten_query。"""
        # 去掉首尾 ``` 围栏
        content = re.sub(r"^```(?:json)?\s*", "", llm_content)
        content = re.sub(r"\s*```$", "", content)

        llm_content_obj: Dict[str, Any] = json.loads(content)
        original_item_names = llm_content_obj.get("item_names")
        if not isinstance(original_item_names, list):
            item_names = []
        else:
            item_names = [item_name.strip() for item_name in original_item_names if
                          isinstance(item_name, str) and item_name.strip()]

        original_rewritten_query = llm_content_obj.get("rewritten_query")
        if not isinstance(original_rewritten_query, str):
            rewritten_query = ""
        else:
            rewritten_query = original_rewritten_query.strip()

        return {"item_names": item_names, "rewritten_query": rewritten_query}


class ItemNameConfirmedNode(BaseNode):
    """商品名确认节点：

    1. LLM 从原始查询提取商品名并改写查询；
    2. 到 Milvus 商品名库对齐，得到 confirmed（精确）与 options（多个相似候选）；
    3. 决策分流——confirmed 非空则带商品名进入三路检索；options 非空则把候选写进 answer 交用户确认；
       都为空则 answer 写兜底话术。后两种情况由主图条件边直接跳答案输出。
    """
    name = "item_name_confirmed_node"

    def __init__(self):
        super().__init__()
        self._extractor = _ItemNameExtractor(self.logger, self.name)
        self._item_name_aligner = _ItemNameAligner(self.logger, self.config)

    def process(self, state: QueryGraphState) -> QueryGraphState:
        original_query = state["original_query"]
        # 从 MongoDB 取历史对话并格式化后拼入提示词
        session_id = state.get("session_id", "")
        history_messages = get_recent_messages(
            session_id=session_id
        )

        state["history"] = history_messages

        formatted_history_str = ""
        for msg in history_messages:
            role = msg.get("role")
            content = msg.get("text", "")
            formatted_history_str += f"{role}: {content}\n"

        llm_result: Dict[str, Any] = self._extractor.extract_item_name(original_query, formatted_history_str)
        self.logger.info(f"大模型提取商品名的结果:{llm_result}")
        # 商品名对齐
        extracted_item_names = llm_result.get("item_names")
        confirmed, options = self._item_name_aligner.search_and_align(extracted_item_names)
        rewritten_query = llm_result.get("rewritten_query")
        self._dicide(confirmed, options, state, rewritten_query)
        return state

    def _dicide(self, confirmed: List[str], options: List[str], state: QueryGraphState, rewritten_query: str):
        """按对齐结果决策：精确商品名进检索，候选写 answer 待确认，无商品写兜底话术。"""
        if confirmed:
            state["item_names"] = confirmed
            state["rewritten_query"] = rewritten_query
        elif options:
            state["answer"] = (
                f"我不能确认你指的是哪款产品,您是在询问以下产品:{'、'.join(options)}吗"
            )
        else:
            state["answer"] = "抱歉，我无法识别您询问的具体产品名称，请提供更准确的产品名称或型号。"
