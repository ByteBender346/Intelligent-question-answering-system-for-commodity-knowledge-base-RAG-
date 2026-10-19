"""查询服务：组装检索图初始状态、驱动 LangGraph 检索流程，维护任务状态机与会话历史。"""

import logging
import uuid

from knowledge.processor.query_processor.main_graph import query_app
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.utils.mongo_history_util import clear_history, get_recent_messages
from knowledge.utils.sse_util import push_sse_event, SSEEvent
from knowledge.utils.task_util import get_task_result, update_task_status, TASK_STATUS_PROCESSING, \
    TASK_STATUS_COMPLETED, TASK_STATUS_FAILED, set_task_result

logger = logging.getLogger(__name__)
class QueryService:

    def generate_session_id(self):
        return uuid.uuid4()

    def generate_task_id(self):
        return uuid.uuid4().hex[:12]

    def get_query_result(self,task_id:str) -> str:
        return get_task_result(task_id=task_id,key="answer")

    def run_query_graph(self,query:str,session_id:str,task_id:str,is_stream:bool):
        # 组装检索图初始 state
        query_graph_state = QueryGraphState()
        query_graph_state['original_query'] = query
        query_graph_state['session_id'] = session_id
        query_graph_state['task_id'] = task_id
        query_graph_state['is_stream'] = is_stream
        try:
            # 任务状态机：进行中 → 已完成
            update_task_status(task_id=task_id, status_name=TASK_STATUS_PROCESSING)
            query_app.invoke(query_graph_state)
            update_task_status(task_id=task_id, status_name=TASK_STATUS_COMPLETED)
        except Exception as e:
            logger.exception("运行查询流程出现异常")
            error = str(e)
            set_task_result(task_id=task_id, key="error", value=error)
            update_task_status(task_id=task_id, status_name=TASK_STATUS_FAILED)
            if is_stream:
                # 流式请求已提前返回，后台异常必须通过 SSE 通知前端
                push_sse_event(task_id, SSEEvent.ERROR, {"error": error, "status": TASK_STATUS_FAILED})
                return
            raise

    def clean_history(self,session_id:str):
        clear_history(session_id)

    def get_history(self,session_id:str):
        history_list = get_recent_messages(session_id)

        # 按时间正序返回历史
        sorted_history_list = sorted(history_list, key=lambda k: k['ts'], reverse=False)
        return [
            {
                "_id": str(history.get("_id", "")),
                "session_id": history.get("session_id") or "",
                "role": history.get("role") or "",
                "text": history.get("text") or "",
                "rewritten_query": history.get("rewritten_query") or "",
                "item_names": history.get("item_names") or [],
                "ts": history.get("ts"),
            } for history in sorted_history_list
        ]
