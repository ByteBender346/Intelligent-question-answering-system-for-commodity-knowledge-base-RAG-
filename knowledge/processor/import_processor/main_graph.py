"""导入流程 LangGraph 编排：注册节点、连接业务边与条件边，编译为可执行图。"""
from langgraph.constants import END
from langgraph.graph import StateGraph

from knowledge.processor.import_processor.nodes.bge_embedding_chunks_node import BgeEmbeddingChunksNode
from knowledge.processor.import_processor.nodes.document_split_node import DocumentSplitNode
from knowledge.processor.import_processor.nodes.entry_node import EntryNode
from knowledge.processor.import_processor.nodes.item_name_recognition_node import ItemNameRecognitionNode
from knowledge.processor.import_processor.nodes.md_image_node import MdImageNode
from knowledge.processor.import_processor.nodes.milvus_import_node import MilvusImportNode
from knowledge.processor.import_processor.nodes.pdf_to_md_node import PdfToMdNode
from knowledge.processor.import_processor.state import ImportGraphState

def import_router(state:ImportGraphState):
    # 入口条件路由：按文件类型分流到 PDF 分支或 MD 分支，否则结束
    if state['is_pdf_read_enabled']:
        return "pdf_router"
    elif state['is_md_read_enabled']:
        return "md_router"
    return END

def import_graph():
    work_flow = StateGraph(state_schema=ImportGraphState)
    # 注册全部节点
    node_list = {
        "entry_node":EntryNode(),
        "pdf_to_md_node":PdfToMdNode(),
        "md_image_node":MdImageNode(),
        "document_split_node":DocumentSplitNode(),
        "item_name_recognition_node":ItemNameRecognitionNode(),
        "bge_embedding_chunks_node":BgeEmbeddingChunksNode(),
        "milvus_import_node":MilvusImportNode()
    }
    for node_name,node in node_list.items():
        work_flow.add_node(node_name,node)

    # 入口节点
    work_flow.set_entry_point("entry_node")

    """
    流程图：
    entry_node ----> 条件边: PDF -> pdf_to_md_node, MD -> md_image_node
    pdf_to_md_node ----> md_image_node
    md_image_node -----> document_split_node
    document_split_node ----> item_name_recognition_node
    item_name_recognition_node ----> bge_embedding_chunks_node
    bge_embedding_chunks_node ----> milvus_import_node
    milvus_import_node -----> END
    """
    # 条件边：按 import_router 返回值映射到目标节点
    work_flow.add_conditional_edges("entry_node",import_router,
                                    {
                                        "pdf_router":"pdf_to_md_node",
                                        "md_router":"md_image_node"
                                    })
    # 业务边：固定顺序串联后续节点
    work_flow.add_edge("pdf_to_md_node","md_image_node")
    work_flow.add_edge("md_image_node","document_split_node")
    work_flow.add_edge("document_split_node","item_name_recognition_node")
    work_flow.add_edge("item_name_recognition_node","bge_embedding_chunks_node")
    work_flow.add_edge("bge_embedding_chunks_node","milvus_import_node")
    work_flow.add_edge("milvus_import_node",END)

    # 编译得到可执行图
    compiled_graph = work_flow.compile()
    return compiled_graph

import_app = import_graph()
