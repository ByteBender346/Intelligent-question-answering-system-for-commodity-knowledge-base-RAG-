"""查询流程主图：LangGraph 编排商品名确认 → 三路并行检索 → RRF 融合 → 精排 → 答案输出。"""
import logging

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from dotenv import load_dotenv
from knowledge.processor.query_processor.nodes.answer_output_node import AnswerOutputNode
from knowledge.processor.query_processor.nodes.hyde_search_node import HydeSearchNode
from knowledge.processor.query_processor.nodes.item_name_confirmed_node import ItemNameConfirmedNode
from knowledge.processor.query_processor.nodes.rerank_node import RerankNode
from knowledge.processor.query_processor.nodes.rrf_merge_node import RrfMergeNode
from knowledge.processor.query_processor.nodes.vector_search_node import VectorSearchNode
from knowledge.processor.query_processor.nodes.web_search_node import WebSearchNode
from knowledge.processor.query_processor.state import QueryGraphState

# 加载环境变量
load_dotenv()


def route_after_item_confirm(state: QueryGraphState) -> bool:
    """商品名确认后的条件路由：已有答案则跳过检索直达答案输出，否则进入三路检索。"""
    if state.get("answer"):
        return True
    return False


def create_query_graph() -> CompiledStateGraph:
    """构建并编译查询图。

    流程：item_name_confirm
        ├─ 有 answer → answer_output → END
        └─ 无 answer → multi_search（扇出三路并行）
              ├─ vector_search / hyde_search / web_search
              └─ join → rrf_merge → rerank → answer_output → END
    """
    workflow = StateGraph(QueryGraphState)  # type:ignore

    # 注册节点：multi_search 与 join 为扇出/汇聚用的虚拟节点
    nodes = {
        "item_name_confirmed_node": ItemNameConfirmedNode(),
        "multi_search": lambda x: x,
        "vector_search_node": VectorSearchNode(),
        "hyde_search_node": HydeSearchNode(),
        "web_search_node": WebSearchNode(),
        "join": lambda x: {},
        "rrf_merge_node": RrfMergeNode(),
        "rerank_node": RerankNode(),
        "answer_output_node": AnswerOutputNode()
    }
    for node_name, node in nodes.items():
        workflow.add_node(node_name, node)

    workflow.set_entry_point("item_name_confirmed_node")

    # 条件边：按 route_after_item_confirm 返回值分流
    workflow.add_conditional_edges(
        source="item_name_confirmed_node",
        path=route_after_item_confirm,
        path_map={
            True: "answer_output_node",
            False: "multi_search"
        }
    )

    # 扇出：multi_search 并行触发三路检索
    workflow.add_edge("multi_search", "vector_search_node")
    workflow.add_edge("multi_search", "hyde_search_node")
    workflow.add_edge("multi_search", "web_search_node")

    # 汇聚：三路均完成后进入 join
    workflow.add_edge("vector_search_node", "join")
    workflow.add_edge("hyde_search_node", "join")
    workflow.add_edge("web_search_node", "join")

    workflow.add_edge("join", "rrf_merge_node")
    workflow.add_edge("rrf_merge_node", "rerank_node")
    workflow.add_edge("rerank_node", "answer_output_node")
    workflow.add_edge("answer_output_node", END)

    return workflow.compile()


# 全局图实例
query_app = create_query_graph()
