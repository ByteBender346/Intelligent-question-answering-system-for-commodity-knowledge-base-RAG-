"""导入流程节点基类：统一节点接口、日志、任务追踪与异常包装。"""
import time
from abc import ABC, abstractmethod
from typing import TypeVar, Optional
import logging

from google.protobuf import duration

from knowledge.processor.import_processor.config import ImportConfig, get_config
from knowledge.processor.import_processor.exceptions import ImportProcessError
from knowledge.utils.task_util import add_running_task, add_done_task, add_node_duration

T = TypeVar("T")  # 泛型状态类型


class BaseNode(ABC):
    """导入流程节点基类：子类实现 process 方法，由 LangGraph 以 __call__ 调用。"""

    name: str = "base_node"  # 节点名称，子类应覆盖

    def __init__(self, config: Optional[ImportConfig] = None):
        # 默认使用全局配置单例
        self.config = config or get_config()
        self.logger = logging.getLogger(f"import.{self.name}")

    def __call__(self, state: T) -> T:
        """节点执行入口：统一日志、任务状态流转、耗时统计与异常包装。"""
        try:
            self.logger.info(f"--- {self.name} 开始 ---")
            task_id = state.get("task_id")
            add_running_task(task_id, self.name)

            start_time = time.time()
            result = self.process(state)

            self.logger.info(f"--- {self.name} 完成 ---")
            end_time = time.time()
            add_done_task(task_id, self.name)
            add_node_duration(task_id, self.name, end_time - start_time)
            return result
        except Exception as e:
            self.logger.error(f"{self.name} 执行失败: {e}")
            raise ImportProcessError(
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
    """配置导入流程日志格式。"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
