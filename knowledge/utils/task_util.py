"""任务状态追踪：每个上传文件对应一个 task_id，记录其处理节点的运行/完成状态与结果。"""

from typing import Dict, List
from collections import defaultdict


# defaultdict：访问不存在的 key 时自动初始化为空列表
_tasks_running_list: Dict[str, List[str]] = defaultdict(list)
_tasks_done_list: Dict[str, List[str]] = defaultdict(list)
_tasks_duration: Dict[str, Dict[str, float]] = defaultdict(dict)

# defaultdict(dict)：访问不存在的 key 自动初始化为空字典
# 结构：{task_id: {节点名: 节点结果}}，如 {"task1": {"answer": "节点结果"}}
_tasks_result: Dict[str, Dict[str, str]] = defaultdict(dict)

_tasks_status: Dict[str, str] = {}

TASK_STATUS_PROCESSING = "processing"  # 任务处理中
TASK_STATUS_COMPLETED = "completed"  # 任务完成
TASK_STATUS_FAILED = "failed"  # 任务失败

_NODE_NAME_TO_CN: Dict[str, str] = {
    "upload_file": "上传文件",
    "entry_node": "检查文件",
    "pdf_to_md_node": "PDF转Markdown",
    "md_image_node": "Markdown图片处理",
    "document_split_node": "文档切分",
    "item_name_recognition_node": "主体名称识别",
    "bge_embedding_chunks_node": "向量生成",
    "milvus_import_node": "导入向量数据库",
    "__end__": "处理完成",
    # Query 流程节点
    "item_name_confirmed_node": "确认问题产品",
    "answer_output_node": "生成答案",
    "rerank_node": "重排序",
    "rrf_merge_node": "倒排融合",
    "web_search_node": "网络搜索",
    "vector_search_node": "切片搜索",
    "hyde_search_node": "切片搜索(假设性文档)"
}


def _to_cn(node_name: str) -> str:
    """节点英文名转中文展示名，未配置时原样返回。"""
    return _NODE_NAME_TO_CN.get(node_name, node_name)


def add_running_task(task_id: str, node_name: str) -> None:
    """把节点加入运行中列表，去重。"""
    running = _tasks_running_list[task_id]
    if node_name not in running:
        running.append(node_name)


def add_done_task(task_id: str, node_name: str) -> None:
    """节点结束：从运行列表移除，加入已完成列表，均去重。"""
    if node_name in _tasks_running_list[task_id]:
        _tasks_running_list[task_id].remove(node_name)

    done = _tasks_done_list[task_id]
    if node_name not in done:
        done.append(node_name)


def get_running_task_list(task_id: str) -> List[str]:
    """返回任务当前运行中节点的中文名列表。"""
    return [_to_cn(n) for n in _tasks_running_list.get(task_id, [])]


def get_done_task_list(task_id: str) -> List[str]:
    """返回任务已完成节点的中文名列表。"""
    return [_to_cn(n) for n in _tasks_done_list.get(task_id, [])]


def get_task_status(task_id: str) -> str:
    """返回任务总体状态，不存在则空串。"""
    return _tasks_status.get(task_id, "")


def update_task_status(task_id: str, status_name: str) -> None:
    """更新任务总体状态（processing/completed/failed）。"""
    _tasks_status[task_id] = status_name


def set_task_result(task_id: str, key: str, value: str) -> None:
    """存储任务结果字段（如 answer / error）。"""
    _tasks_result[task_id][key] = value


def get_task_result(task_id: str, key: str, default: str = "") -> str:
    """读取任务结果字段（如 answer / error）。"""
    return _tasks_result.get(task_id, {}).get(key, default)


def add_node_duration(task_id: str, node_name: str, duration: float) -> None:
    """记录节点耗时（秒）。"""
    cn_name = _to_cn(node_name)
    _tasks_duration[task_id][cn_name] = round(duration, 2)

def get_node_durations(task_id: str) -> Dict[str, float]:
    """获取所有节点的耗时。"""
    return dict(_tasks_duration.get(task_id, {}))

def get_task_info(task_id: str):
    """聚合任务全局信息：状态 + 运行中节点 + 已完成节点 + 各节点耗时。"""
    return {
        "status": get_task_status(task_id),
        "running_list": get_running_task_list(task_id),
        "done_list": get_done_task_list(task_id),
        "durations": get_node_durations(task_id)
    }
