"""导入流程状态定义：TypedDict 字段结构与默认状态工具函数。"""

from typing import TypedDict, List

import copy


class ImportGraphState(TypedDict, total=False):
    """导入流程图状态，承载全流程节点间传递的数据。"""

    # ==================== 任务标识 ====================
    task_id: str  # 任务 ID，用于任务追踪（前端实时查看节点处理日志）

    # ==================== 控制标志 ====================
    is_md_read_enabled: bool  # 是否启用 MD 读取分支
    is_pdf_read_enabled: bool  # 是否启用 PDF 读取分支

    # ==================== 路径信息 ====================
    import_file_path: str  # 导入文件路径
    file_dir: str  # 导入/输出文件目录
    pdf_path: str  # PDF 文件路径
    md_path: str  # 转换后 Markdown 文件路径

    # ==================== 文件信息 ====================
    file_title: str  # 文件标题（不含扩展名）
    item_name: str  # 识别出的商品/产品名称

    # ==================== 处理中间数据 ====================
    md_content: str  # Markdown 文档内容
    chunks: List  # 文档切片列表


GRAPH_DEFAULT_STATE: ImportGraphState = {
    "task_id": "",
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,
    "file_dir": "",
    "import_file_path": "",
    "pdf_path": "",
    "md_path": "",
    "file_title": "",
    "md_content": "",
    "chunks": [],
    "item_name": "",
}


def create_default_state(**overrides) -> ImportGraphState:
    """创建默认状态，可用关键字参数覆盖指定字段。"""
    state = copy.deepcopy(GRAPH_DEFAULT_STATE)
    state.update(overrides)
    return state


def get_default_state() -> ImportGraphState:
    """获取默认状态副本（深拷贝，避免污染全局默认值）。"""
    return copy.deepcopy(GRAPH_DEFAULT_STATE)
