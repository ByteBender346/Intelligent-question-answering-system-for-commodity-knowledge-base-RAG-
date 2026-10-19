"""知识库文件导入 API：接收上传文件并后台执行导入工作流。"""

import uvicorn
from fastapi import FastAPI, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import Depends
from starlette.staticfiles import StaticFiles

from knowledge.core.deps import get_file_process_service
from knowledge.core.paths import get_front_page_dir
from knowledge.processor.import_processor.main_graph import import_app
from knowledge.schema.upload_schema import UploadResponse, TaskStatusResponse
from knowledge.service.file_process_service import FileProcessService
from knowledge.utils.task_util import get_task_info


def register_router(app: FastAPI):
    @app.get("/hello")
    def hello_world():
        return "Hello world"

    @app.post("/upload", response_model=UploadResponse)
    def upload_file(file: UploadFile,
                    background_tasks: BackgroundTasks,
                    file_process_service: FileProcessService = Depends(get_file_process_service)):
        # 保存上传文件并生成任务 ID，随后在后台执行导入主流程
        import_file_path, file_dir, task_id = file_process_service.process_upload_file(file)
        background_tasks.add_task(file_process_service.run_main_graph, import_file_path, file_dir, task_id)

        return UploadResponse(message="上传成功", task_id=task_id)

    @app.get("/status/{task_id}", response_model=TaskStatusResponse)
    def get_task_status(task_id: str):
        task_info = get_task_info(task_id)

        return TaskStatusResponse(**task_info)


def create_app():
    app = FastAPI(
        description="商品知识智能问答导入流程API",
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

    # 挂载前端静态目录：请求 /front/import.html 时从挂载目录查找对应文件
    front_dir = get_front_page_dir()
    app.mount("/front", StaticFiles(directory=front_dir))

    register_router(app)

    return app


if __name__ == '__main__':
    # 本地开发启动入口：python -m knowledge.api.import_api
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
