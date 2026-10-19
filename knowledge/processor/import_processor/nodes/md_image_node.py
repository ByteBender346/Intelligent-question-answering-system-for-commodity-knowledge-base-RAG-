"""Markdown 图片处理节点：扫描图片、VLM 生成图片摘要、上传 MinIO 并回填 Markdown。"""
import base64
import re
from logging import Logger
from pathlib import Path
from typing import Tuple, List, Dict

from openai import OpenAI

from knowledge.processor.import_processor.base import BaseNode, T
from knowledge.processor.import_processor.config import ImportConfig
from knowledge.processor.import_processor.exceptions import StateFieldError, FileProcessingError
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.clients.ai_clients import AIClients
from knowledge.utils.clients.storage_clients import StorageClients


class ImageContext:
    head_title:str
    pre_context:str
    post_context:str
    def __init__(self,head_title:str,pre_context:str,post_context:str):
        self.head_title = head_title
        self.pre_context = pre_context
        self.post_context = post_context

class ImageInfo:
    image_name:str
    image_path:str
    image_context:ImageContext
    def __init__(self,image_name:str,image_path:str,image_context:ImageContext):
        self.image_name = image_name
        self.image_path = image_path
        self.image_context = image_context

class _MdFileHandler:
    def __init__(self,logger:Logger,node_name:str):
        self.logger = logger
        self.node_name = node_name

    def read_md(self,md_path:str) -> Tuple[str,Path,Path]:
        # 校验 md_path 非空且存在
        if not md_path:
            self.logger.error("md_path 不能为空")
            raise StateFieldError(node_name=self.node_name,field_name="md_path",expected_type=str)
        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            self.logger.error(f"md_path '{md_path}' 不存在")
            raise FileProcessingError(node_name=self.node_name,message=f"md_path'{md_path}'不存在")
        # 读取 md 内容
        with open(md_path_obj, 'r', encoding='utf-8') as f:
            md_content = f.read()
        # 图片目录与 md 文件同级 images/
        image_dir_obj = md_path_obj.parent / "images"

        return md_content,md_path_obj,image_dir_obj

    def backup_md(self,md_content:str,md_path:str) -> None:
        # 备份为 原文件名_new.md，便于测试观察
        md_path_obj = Path(md_path)
        backup_md_path_obj = md_path_obj.parent / (md_path_obj.stem + "_new" + md_path_obj.suffix)
        try:
            with open(backup_md_path_obj, 'w', encoding='utf-8') as f:
                f.write(md_content)
        except Exception as e:
            self.logger.warning(f"备份md文件失败: {e},但是不影响程序的运行")
        self.logger.info(f"文件备份成功，文件路径是{str(backup_md_path_obj)}")

class _ImageScanner:
    def __init__(self,config:ImportConfig):
        self.config = config
    def scan_img_dir(self,image_dir_obj:Path,md_content:str) -> Tuple[List[ImageInfo],List[str]]:

        image_info_list = []

        for image_file in image_dir_obj.iterdir():
            # 跳过子目录与非图片文件
            if not image_file.is_file():
                continue
            if image_file.suffix not in self.config.image_extensions:
                continue
            # 在 md_content 中定位该图片所在行
            md_lines = md_content.split("\n")
            # 匹配 md 中引用该图片的 ![]() 行
            pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file.name) + r".*?\)")

            in_code_fence = False
            code_fence_pattern = re.compile(r"^\s*```")
            for index,md_line in enumerate(md_lines):
                # 代码块内的图片引用不计
                if code_fence_pattern.match(md_line):
                    in_code_fence = not in_code_fence

                if in_code_fence or not pattern.match(md_line):
                    continue
                # 命中图片行：向上找最近标题作为上文
                heading_pattern = re.compile(r'^#{1,6}\s+(.+)$')
                head_title,head_content = self._find_heading_content(md_lines,heading_pattern,index)

                # 向下找最近标题作为下文
                post_content = self._find_post_content(index,md_lines,heading_pattern)

                # 按配置长度截取上/下文
                img_content_length = self.config.img_content_length
                pre_context = self._extract_image_context(head_content,img_content_length,"up")
                post_context = self._extract_image_context(post_content,img_content_length,"down")

                image_context = ImageContext(head_title =head_title,pre_context=pre_context,post_context=post_context)

                image_info = ImageInfo(image_name=image_file.name,image_path=str(image_file),image_context=image_context)

                image_info_list.append(image_info)

        return image_info_list,md_lines




    def _find_heading_content(self,md_lines:List[str],heading_pattern,image_index:int) -> Tuple[str,List[str]]:
        # 从图片行向上找最近的标题行（跳过代码块）
        head_title_index = -1
        head_title = ""

        in_code_fence = False
        code_fence_pattern = re.compile(r"^\s*```")
        for i in range(image_index - 1,-1,-1):
            line = md_lines[i]
            if code_fence_pattern.match(line):
                in_code_fence = not in_code_fence

            if heading_pattern.match(line) and not in_code_fence:
                head_title_index = i
                head_title = line
                break
        # 标题之后到图片行之间为上文
        head_content = md_lines[head_title_index + 1:image_index]

        return head_title,head_content

    def _find_post_content(self,image_index:int,md_lines:List[str],heading_pattern:re.Pattern) -> List[str]:
        # 从图片行向下找最近的标题行（跳过代码块）
        in_code_fence = False
        code_fence_pattern = re.compile(r"^\s*```")

        post_title_index = len(md_lines)
        for i in range(image_index + 1,len(md_lines)):
            line = md_lines[i]
            if code_fence_pattern.match(line):
                in_code_fence = not in_code_fence
            if not in_code_fence and heading_pattern.match(line):
                post_title_index = i
                break

        post_content = md_lines[image_index + 1: post_title_index]
        return post_content

    def _extract_image_context(self, content:List[str], img_content_length:int,direction:str) -> str:
        image_pattern = re.compile(r'!\[.*?\]\(.*?\)')
        current_graph = []
        final_graphs = []

        # 空行或其他图片行作为段落分隔
        for line in content:
            if not line.strip() or image_pattern.match(line):
                if current_graph:
                    final_graphs.append("\n".join(current_graph))
                    current_graph = []
            else:
                current_graph.append(line)

        if current_graph:
            final_graphs.append("\n".join(current_graph))
            current_graph = []

        extract_graphs = []
        total_length = 0

        # 截取上文时从近到远收集，再反转回原顺序
        if direction == "up":
            final_graphs.reverse()

        for graph in final_graphs:
            if total_length > img_content_length:
                break
            extract_graphs.append(graph)
            total_length += len(graph) + len("\n\n")

        if direction == "up":
            extract_graphs.reverse()

        return "\n\n".join(extract_graphs)

class _VLMSummarizer:
    def __init__(self,logger:Logger,config:ImportConfig):
        self.logger = logger
        self.config = config

    def summarize_all(self,document_name,image_info_list:List[ImageInfo]):
        # key: 图片名, value: 图片摘要
        image_summaries = {}
        # VLM 不可用时，所有图片兜底为「暂无摘要信息」
        try:
            vlm_client = AIClients.get_vlm_client()
        except ConnectionError as e:
            self.logger.warning(f"无法连接到VLM模型，图片描述信息将无法生成: {e}")
            for image_info in image_info_list:
                image_summaries[image_info.image_name] = "暂无摘要信息"
            return image_summaries

        # 逐张图片生成摘要
        for image_info in image_info_list:
            image_summarize = self._summarize_single(document_name,image_info,vlm_client)
            image_summaries[image_info.image_name] = image_summarize

        return image_summaries

    def _summarize_single(self,document_name:str, image_info:ImageInfo, vlm_client:OpenAI):
        # 拼接图片的标题与上/下文
        image_context = image_info.image_context
        final_context = f"{image_context.head_title} \n {image_context.pre_context} \n {image_context.post_context}"

        # 读取图片并 base64 编码
        image_path = image_info.image_path
        with open(image_path,"rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        # 组装多模态消息：文本要求生成中文简短标题
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"任务：为Markdown文档中的图片生成一个简短的中文标题。\n"
                            f"背景信息：\n"
                            f"  1. 所属文档标题：\"{document_name}\"\n"
                            f"  2. 图片上下文：{final_context}\n"
                            f"请结合图片内容和上述上下文信息，"
                            f"用中文简要总结这张图片的内容，"
                            f"生成一个精准的中文标题摘要（不要包含图片二字）。"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        },
                    },
                ]
            }
        ]
        # 调用 VLM，失败时兜底为「暂无摘要信息」
        try:
            vlm_result = vlm_client.chat.completions.create(
                model=self.config.vl_model,
                messages=messages
            )
        except Exception as e:
            self.logger.warning(f"调用VLM模型失败，图片'{image_info.image_name}'的描述信息将无法生成: {e}")
            return "暂无摘要信息"

        return vlm_result.choices[0].message.content

class _ImageUploader:
    def __init__(self,logger:Logger,config:ImportConfig):
        self.logger = logger
        self.config = config
    def upload_and_replace(self,md_lines,document_name:str,image_info_list:List[ImageInfo],image_summarizes:Dict[str,str]) -> str:
        # 上传图片到 MinIO
        remote_urls = self._upload_all(document_name,image_info_list)
        print(remote_urls)
        # 用摘要与远程 URL 替换 md 中的图片引用
        new_md_content = self._replace_md_content(md_lines,image_summarizes,remote_urls)

        return new_md_content

    def _upload_all(self,document_name:str,image_info_list:List[ImageInfo]) -> Dict[str, str]:
        # 每张图片在 MinIO 上的访问路径
        remote_urls = {}
        # MinIO 不可用时，所有图片回退使用本地原路径
        try:
            minio_client = StorageClients.get_minio_client()
        except ConnectionError as e:
            self.logger.warning(f"无法连接到MinIO服务器，图片将无法上传: {e}")
            for image_info in image_info_list:
                remote_urls[image_info.image_name] = image_info.image_path
            return remote_urls

        for image_info in image_info_list:
            # object_name = 文档名/图片名
            object_name = document_name + "/" + image_info.image_name
            buket_name = self.config.minio_bucket
            try:
                minio_client.fput_object(
                    bucket_name=buket_name,
                    object_name=object_name,
                    file_path=image_info.image_path
                )
            except Exception as e:
                # 单张上传失败则回退原路径，不中断整体
                remote_urls[image_info.image_name] = image_info.image_path
                continue

            # 拼接 MinIO 访问 URL
            remote_url = self.config.get_minio_base_url() + "/" + buket_name + "/" + object_name

            remote_urls[image_info.image_name] = remote_url
        return remote_urls

    def _replace_md_content(self, md_lines:List[str], image_summarizes:Dict[str,str], remote_urls:Dict[str,str]) -> str:
        # 匹配 md 图片引用，分组1为标题、分组2为路径
        pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")
        in_code_fence = False
        code_fence_pattern = re.compile(r"^\s*```")

        new_md_lines = []
        for line in md_lines:
            if code_fence_pattern.match(line):
                in_code_fence = not in_code_fence
            # 代码块外的图片行才替换为摘要 + 远程 URL
            match = pattern.match(line)
            if not in_code_fence and match:
                image_name = Path(match[2]).name
                new_md_lines.append(f"![{image_summarizes[image_name]}]({remote_urls[image_name]})")
            else:
                new_md_lines.append(line)

        return "\n".join(new_md_lines)


class MdImageNode(BaseNode):
    name = "md_image_node"

    def __init__(self):
        super().__init__()
        self.md_file_handler = _MdFileHandler(logger=self.logger, node_name=self.name)
        self.image_scanner = _ImageScanner(self.config)
        self.vlm_summarizer = _VLMSummarizer(self.logger, self.config)
        self.image_uploader = _ImageUploader(logger=self.logger, config=self.config)

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 读取 md 内容与图片目录
        md_path = state.get("md_path")
        md_content,md_path_obj,image_dir_obj = self.md_file_handler.read_md(md_path)

        # 扫描图片目录并与 md 中的图片行匹配，提取上下文
        image_info_list,md_lines = self.image_scanner.scan_img_dir(image_dir_obj,md_content)

        # VLM 为每张图片生成中文摘要
        image_summarizes = self.vlm_summarizer.summarize_all(document_name=md_path_obj.stem,image_info_list=image_info_list)

        # 上传图片到 MinIO，并把摘要与 URL 回填到 md
        new_md_content = self.image_uploader.upload_and_replace(md_lines=md_lines,document_name=md_path_obj.stem,image_info_list=image_info_list,image_summarizes=image_summarizes)

        # 回填后的 md 备份为新文件
        self.md_file_handler.backup_md(new_md_content,md_path)

        state["md_content"] = new_md_content
        return state
