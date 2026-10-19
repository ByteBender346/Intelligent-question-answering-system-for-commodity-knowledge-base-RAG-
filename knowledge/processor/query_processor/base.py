"""查询流程节点基类：统一节点接口，提供日志、任务追踪与异常包装。"""
from abc import ABC, abstractmethod
from typing import TypeVar, Optional
import logging
from knowledge.processor.query_processor.config import QueryConfig, get_config
from knowledge.processor.query_processor.exceptions import QueryProcessError
from knowledge.utils.sse_util import push_sse_event, SSEEvent
from knowledge.utils.task_util import add_running_task, add_done_task, get_task_status, get_running_task_list, \
    get_done_task_list

T = TypeVar("T")  # 泛型状态类型


class BaseNode(ABC):
    """查询流程节点基类：子类实现 process 方法，由 LangGraph 经 __call__ 调用。"""

    name: str = "base_node"

    def __init__(self, config: Optional[QueryConfig] = None):
        self.config = config or get_config()
        self.logger = logging.getLogger(f"query.{self.name}")

    def __call__(self, state: T) -> T:
        """节点执行入口：记录运行/完成任务，流式时推送进度，异常统一包装为 QueryProcessError。"""
        is_stream = state.get('is_stream')
        task_id = state.get('task_id')

        try:
            self.logger.info(f"--- {self.name} 开始 ---")

            if task_id:
                add_running_task(task_id, self.name)
                if is_stream:
                    self._push_progress(task_id)

            result = self.process(state)

            if task_id:
                add_done_task(task_id, self.name)
                if is_stream:
                    self._push_progress(task_id)

            self.logger.info(f"--- {self.name} 完成 ---")
            return result
        except Exception as e:
            self.logger.error(f"{self.name} 执行失败: {e}")
            raise QueryProcessError(
                message=str(e),
                node_name=self.name,
                cause=e
            )

    @abstractmethod
    def process(self, state: T) -> T:
        """节点核心处理逻辑，子类必须实现。"""
        pass

    def log_step(self, step_name: str, message: str = ""):
        """记录步骤日志。"""
        log_msg = f"[{step_name}]"
        if message:
            log_msg += f" {message}"
        self.logger.info(log_msg)

    def setup_logging(level: int = logging.INFO):
        """配置查询流程日志。"""
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def _push_progress(self, task_id: str):
        """通过 SSE 推送当前任务进度（done/running 列表）。"""
        push_sse_event(
            task_id, SSEEvent.PROGRESS, {
                "status": get_task_status(task_id),
                "done_list": get_done_task_list(task_id),
                "running_list": get_running_task_list(task_id),
            }
        )
