"""文件上传处理服务：保存上传文件、生成任务 ID，并后台执行导入工作流。"""

import logging
import os.path
import shutil
import time
import uuid
from datetime import datetime

from fastapi import UploadFile

from knowledge.core.paths import get_local_base_dir
from knowledge.processor.import_processor.exceptions import FileProcessingError
from knowledge.processor.import_processor.main_graph import import_app
from knowledge.utils.clients.storage_clients import StorageClients
from knowledge.utils.task_util import update_task_status, TASK_STATUS_PROCESSING, TASK_STATUS_FAILED, \
    TASK_STATUS_COMPLETED, add_running_task, add_done_task, add_node_duration

logger = logging.getLogger(__name__)

class FileProcessService:
    def run_main_graph(self,import_file_path:str,file_dir:str,task_id:str):
        logging.basicConfig(level=logging.INFO)
        # 导入工作流初始 state
        state = {
            "task_id": task_id,
            "import_file_path":import_file_path,
            "file_dir":file_dir
        }
        try:
            for event in import_app.stream(state):
                for node_name, state in event.items():
                    print(f"{node_name}正在执行......")

            update_task_status(task_id,TASK_STATUS_COMPLETED)
        except Exception as e:
            logger.error(e)
            update_task_status(task_id,TASK_STATUS_FAILED)
            return

    def _get_base_dir(self):
        local_base_dir = get_local_base_dir()
        return os.path.join(local_base_dir,datetime.now().strftime("%Y%m%d"))

    def process_upload_file(self,file:UploadFile):
        # 生成任务 ID 并登记为运行中
        task_id = uuid.uuid4().hex[:8]

        add_running_task(task_id,"upload_file")
        update_task_status(task_id,TASK_STATUS_PROCESSING)

        start_time = time.time()
        # 保存上传文件到本地临时目录
        try:
            import_file_path,file_dir = self._save_file_to_local(file)

        except Exception as e:
            logger.error(e)
            update_task_status(task_id,TASK_STATUS_FAILED)
            return import_file_path,file_dir,task_id
        # 备份到 MinIO，失败不影响主流程
        self._save_file_to_remote(import_file_path,file.filename)

        end_time = time.time()

        add_done_task(task_id, "upload_file")
        add_node_duration(task_id,"upload_file",duration=end_time-start_time)
        return import_file_path,file_dir,task_id

    def _save_file_to_local(self, upload_file:UploadFile):
        # 按日期组织本地临时目录
        base_dir = self._get_base_dir()

        os.makedirs(base_dir, exist_ok=True)

        import_file_path = os.path.join(base_dir, upload_file.filename)
        try:
            with open(import_file_path,"wb") as f:
                shutil.copyfileobj(upload_file.file, f)
        except Exception as e:
            logger.error(f"文件{upload_file.filename}保存到本地临时目录失败,{str(e)}")
            raise FileProcessingError(message=f"文件{upload_file.filename}保存到本地临时目录失败,{str(e)}")

        return import_file_path,base_dir

    def _save_file_to_remote(self, import_file_path:str, filename:str):
        try:
            minio_client = StorageClients.get_minio_client()
        except ConnectionError as e:
            logger.warning(f"获取minio客户端失败,但不影响主流程,{str(e)}")
            return

        bucket_name = os.getenv("MINIO_BUCKET_NAME")
        object_name =  f"origin_files/{datetime.now().strftime('%Y%m%d')}/{filename}"

        try:
            minio_client.fput_object(bucket_name, object_name, import_file_path)
        except Exception as e:
            logger.warning(f"minio备份文件失败,但不影响主流程,{str(e)}")
            return
