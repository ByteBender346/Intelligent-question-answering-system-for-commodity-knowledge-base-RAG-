"""知识库查询 API：接收检索请求，支持 SSE 流式返回与会话历史管理。"""

import asyncio
from typing import Union

import uvicorn
from fastapi import FastAPI, Depends, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from knowledge.core.deps import get_query_service
from knowledge.core.paths import get_front_page_dir
from starlette.staticfiles import StaticFiles

from knowledge.schema.query_schema import QueryRequest, QueryResponse, StreamSubmitResponse, HistoryResponse
from knowledge.service.query_service import QueryService
from knowledge.utils.sse_util import create_sse_queue, sse_generator
from knowledge.utils.task_util import get_done_task_list


def register_router(app: FastAPI):

    @app.post("/query")
    async def query(query_request: QueryRequest,
                    background_tasks: BackgroundTasks,
                    query_service: QueryService = Depends(get_query_service)) -> Union[QueryResponse, StreamSubmitResponse]:
        # 提取会话与任务信息
        session_id = query_request.session_id or query_service.generate_session_id()
        task_id = query_service.generate_task_id()
        query = query_request.query
        is_stream = query_request.is_stream

        if is_stream:
            # 流式：建立 SSE 队列，后台执行检索，立即返回任务 ID 供前端订阅
            create_sse_queue(task_id)
            background_tasks.add_task(query_service.run_query_graph, query, session_id, task_id, is_stream)
            return StreamSubmitResponse(message="查询请求已经提交", session_id=session_id, task_id=task_id)
        else:
            # 非流式：在线程池中执行检索，完成后同步返回答案与已完成节点
            event_loop = asyncio.get_running_loop()
            await event_loop.run_in_executor(
                None, query_service.run_query_graph, query, session_id, task_id, is_stream
            )
            answer = query_service.get_query_result(task_id=task_id)
            done_list = get_done_task_list(task_id=task_id)
            return QueryResponse(message="查询请求已处理完成", session_id=session_id, answer=answer, done_list=done_list)

    @app.get("/stream/{task_id}")
    async def stream(task_id: str, request: Request):
        # SSE 订阅：向前端推送流式生成结果
        return StreamingResponse(content=sse_generator(task_id=task_id, request=request), media_type="text/event-stream")

    @app.delete("/history/{session_id}")
    def clean_history(session_id: str, query_service: QueryService = Depends(get_query_service)):
        query_service.clean_history(session_id)
        return {"message": f"会话 {session_id} 的历史记录已清空"}

    @app.get("/history/{session_id}")
    def get_history(session_id: str, query_service: QueryService = Depends(get_query_service)):
        history_list = query_service.get_history(session_id)

        return HistoryResponse(session_id=session_id, items=history_list)


def create_app():
    app = FastAPI(
        description="商品知识智能问答检索流程API",
        version="1.0"
    )

    # 跨域配置：allow_origins="*" 与 allow_credentials=True 冲突，故 allow_credentials 保持 False
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 挂载前端静态目录：请求 /front/chat.html 时从挂载目录查找对应文件
    front_dir = get_front_page_dir()
    app.mount("/front", StaticFiles(directory=front_dir))

    register_router(app)

    return app


if __name__ == '__main__':
    # 本地开发启动入口：python -m knowledge.api.query_api
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8001)
