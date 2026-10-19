"""SSE 流式推送封装：后台任务把事件投入队列，异步生成器消费队列并按 SSE 协议下发。"""

import asyncio
import json
import logging
import queue
from typing import Dict, Any, Optional, AsyncGenerator
from fastapi import Request


class SSEEvent:
    PROGRESS = "progress"  # 任务节点进度
    DELTA = "delta"  # LLM 流式输出增量
    FINAL = "final"  # 最终完整答案
    ERROR = "error"  # 查询失败原因


# 全局 SSE 任务队列：Key 为 task_id，Value 为对应队列
_task_stream: Dict[str, queue.Queue] = {}


def get_sse_queue(task_id: str) -> Optional[queue.Queue]:
    """获取指定任务的队列。"""
    return _task_stream.get(task_id)


def create_sse_queue(task_id: str) -> queue.Queue:
    """创建并注册一个新的 SSE 队列。"""
    q = queue.Queue()
    _task_stream[task_id] = q
    return q


def remove_sse_queue(task_id: str):
    """移除指定任务的队列，key 不存在时静默忽略。"""
    _task_stream.pop(task_id, None)


def _sse_pack(event: str, data: Dict[str, Any]) -> str:
    """按 SSE 协议打包一条消息。"""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def push_sse_event(task_id: str, event: str, data: Dict[str, Any]):
    """按 task_id 把事件投入对应 SSE 队列；队列不存在则忽略。"""
    stream_queue = get_sse_queue(task_id)
    if stream_queue:
        stream_queue.put({"event": event, "data": data})


async def sse_generator(task_id: str, request: Request) -> AsyncGenerator:
    """
    SSE 消费者：循环从任务队列取数据，打包成 SSE 协议后 yield 下发。

    连接管理：轮询探测前端是否断连；队列空时下发心跳保活；
    收到 FINAL/ERROR 或连接异常时结束，finally 中清理队列。
    """
    sse_queue = _task_stream.get(task_id)
    if sse_queue is None:
        return

    loop = asyncio.get_event_loop()

    try:
        while True:
            # 主动探测前端 SSE 连接是否已关闭
            if await request.is_disconnected():
                return
            try:
                # 队列阻塞取数放到执行器，避免阻塞事件循环（超时 1s 便于上面的断连探测）
                msg = await loop.run_in_executor(None, sse_queue.get, True, 1)
                event_type = msg.get('event')
                event_data = msg.get('data')
                yield _sse_pack(event_type, event_data)
                # 终态事件结束本次流
                if event_type in (SSEEvent.FINAL, SSEEvent.ERROR):
                    return
            except queue.Empty:
                # 队列暂无数据：下发心跳保活
                yield ": heartbeat\n\n"
                continue
    except (ConnectionResetError, BrokenPipeError) as e:
        # 客户端中断（关闭窗口或浏览器）
        return
    except asyncio.CancelledError:
        # 服务端取消协程，向上抛出由外层处理
        raise
    finally:
        remove_sse_queue(task_id)
