"""入口节点：校验输入路径、识别文件类型并设置分支标志，提取文件标题。"""
import logging
from pathlib import Path

from knowledge.processor.import_processor.base import BaseNode, T
from knowledge.processor.import_processor.exceptions import StateFieldError, ValidationError
from knowledge.processor.import_processor.state import ImportGraphState


class EntryNode(BaseNode):
    name = "entry_node"
    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 校验 import_file_path 和 file_dir 非空
        self.log_step(step_name="Step1",message="import_file_path和file_dir是否为空")

        import_file_path = state.get("import_file_path")
        file_dir = state.get("file_dir")
        if not import_file_path:
            self.logger.error(f"import_file_path not found in state {state}")
            raise StateFieldError(node_name=self.name, field_name="import_file_path",expected_type=str)

        if not file_dir:
            self.logger.error(f"file_dir not found in state {state}")
            raise StateFieldError(node_name=self.name, field_name="file_dir",expected_type=str)

        # 校验路径真实存在
        self.log_step(step_name="Step2",message="import_file_path和file_dir是否为存在")

        import_file_path_obj = Path(import_file_path)
        file_dir_obj = Path(file_dir)
        if not import_file_path_obj.exists():
            self.logger.error(f"import_file_path not exist")
            raise StateFieldError(node_name=self.name, field_name="import_file_path",expected_type=Path)

        if not file_dir_obj.exists():
            self.logger.error(f"file_dir not exist")
            raise StateFieldError(node_name=self.name, field_name="file_dir",expected_type=Path)

        # 按后缀设置 PDF / MD 分支标志，不支持的格式直接报错
        self.log_step(step_name="Step3",message="识别文件后缀类型")

        suffix = import_file_path_obj.suffix
        if suffix == ".pdf":
            state['is_pdf_read_enabled'] = True
            state['pdf_path'] = import_file_path
        elif suffix == ".md":
            state['is_md_read_enabled'] = True
            state['md_path'] = import_file_path
        else:
            self.logger.error(f"Unsupported suffix {suffix}")
            raise ValidationError(message="Unsupported suffix",node_name = self.name)

        # 文件标题（去扩展名）
        self.log_step(step_name="Step4",message="获取文件标题")
        file_title = import_file_path_obj.stem
        state['file_title'] = file_title

        return state
