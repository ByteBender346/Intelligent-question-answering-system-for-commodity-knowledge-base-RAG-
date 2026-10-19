"""文档切分节点：按 Markdown 标题层级切分文档，超长段二次切分、过短段同源合并，产出检索用 chunks。"""

import json
import re
from multiprocessing.process import parent_process
from pathlib import Path
from typing import Tuple, List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from knowledge.processor.import_processor.base import BaseNode, T
from knowledge.processor.import_processor.exceptions import StateFieldError
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.markdown_util import MarkdownTableLinearizer


class DocumentSplitNode(BaseNode):
    """文档切分节点：按标题层级切分 Markdown 文档。"""
    name = "document_split_node"
    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 校验输入参数
        new_md_content,file_title,max_content_length,min_content_length = self._validate_state(state)
        # 按标题层级切分
        sections_by_head = self._split_by_head(new_md_content,file_title)
        # 超长二次切分、过短同源合并
        final_sections = self._split_and_merge(sections_by_head,max_content_length,min_content_length)
        # 组装为后续节点可直接使用的 chunks
        chunks = self._assemble_chunks(final_sections)
        # 备份 chunks 便于排查
        self._backup_chunks(chunks,state)

        state["chunks"] = chunks
        return state

    def _validate_state(self, state:ImportGraphState) -> Tuple[str, str,int,int]:
        # 统一换行符为 \n
        md_content = state.get("md_content","")
        new_md_content = md_content.replace("\r","\n").replace("\r\n","\n\n")
        # 校验切分长度配置（非法配置直接中止）
        max_content_length = self.config.max_content_length
        min_content_length = self.config.min_content_length
        if not max_content_length or max_content_length <= 0 or not min_content_length or min_content_length <= 0 or max_content_length < min_content_length :
            self.logger.error(f"max_content_length={max_content_length} 或者 min_content_length={min_content_length}配置错误,请检查")
            raise StateFieldError(node_name=self.name,field_name="max_content_length or min_content_length",expected_type=int)
        # 文档标题：供无标题段落兜底
        file_title = state.get("file_title")
        return new_md_content,file_title,max_content_length,min_content_length

    def _split_by_head(self, md_content:str, file_title:str) -> List[Dict[str,Any]]:
        """
        按标题行切分文档，返回 section 列表：
        {"body": 正文, "title": 标题, "parent_title": 父标题, "file_title": 文档标题}
        """
        md_lines = md_content.split("\n")
        # 匹配 Markdown 标题行
        heading_re = re.compile(r"^\s*(#{1,6})\s+(.+)")
        code_fence_pattern = re.compile(r"^\s*```")
        in_code_fence = False
        # hierarchy[i] 记录当前第 i 级标题，用于识别父标题
        hierarchy = [""]*7
        # 逐行扫描标题，代码块内的 # 不计为标题
        body_lines = []
        current_title = ""
        current_level = 0
        final_sections = []
        # 收集当前标题块并写入 final_sections
        def _flush():
            """
            section 结构：{"body", "title", "parent_title", "file_title"}
            """
            body = "\n".join(body_lines)
            title = current_title

            # 无标题且无正文的段落跳过
            if not body and not title:
                return

            # 标题为空时用文档标题兜底
            title = title if title else file_title


            # 父标题：从当前层级向上找最近的非空标题
            parent_title = ""
            for i in range(current_level - 1,0,-1):
                if hierarchy[i] != "":
                    parent_title = hierarchy[i]
                    break

            section = {
                "body":body,
                "title":title,
                "parent_title":parent_title if parent_title else title,
                "file_title":file_title
            }

            final_sections.append(section)


        for md_line in md_lines:
            if code_fence_pattern.match(md_line):
                in_code_fence = not in_code_fence

            match = heading_re.match(md_line)
            if match and not in_code_fence:
                # 标题行：先收起上一段，再切换当前标题与层级
                _flush()
                current_title = md_line
                body_lines = []
                # 标题层级 = # 的个数，并清空更低层级的记录
                current_level = len(match.group(1))
                hierarchy[current_level] = current_title
                for i in range(current_level+1,7):
                    hierarchy[i] = ""
            else:
                # 非标题行：追加到当前段正文
                body_lines.append(md_line)

        # 收尾：收集最后一段
        _flush()

        return final_sections

    def _split_and_merge(self, sections:List[Dict[str,Any]], max_content_length:int, min_content_length:int) -> List[Dict[str,Any]]:
        """超长 section 二次切分，过短 section 同源合并。"""
        current_sections = []
        # 二次切分超长 section
        for section in sections:
            split_sections = self._split_long_section(section,max_content_length)
            current_sections.extend(split_sections)

        # 合并过短的 section
        final_sections = self._merge_short_section(current_sections,min_content_length)
        return final_sections

    def _split_long_section(self, section:Dict[str,Any], max_content_length:int) -> List[Dict[str,Any]]:
        """section 超过 max_content_length 时按最大长度二次切分。"""
        title = section.get("title")
        # title 过长时截断前 80 字符
        if len(title) > 80:
            title = title[:80]

        title_prefix = title + "\n\n"
        body = section.get("body")

        # 表格先线性化，避免被文本切分器切碎
        if "<table>" in body:
            body = MarkdownTableLinearizer.process(body)
            section["body"] = body

        total_length = len(body) + len(title_prefix)

        if total_length <= max_content_length:
            # 未超长则原样返回
            return [section]

        # 可切分长度 = 最大长度 - 标题前缀
        split_content_length = max_content_length - len(title_prefix)

        # 超长时用 RecursiveCharacterTextSplitter 二次切分
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", "？", "！", "；", ".", "?", "!", ';', " ", ""],
            chunk_size=split_content_length,
            keep_separator=True,
            chunk_overlap=0
        )

        split_body_list = text_splitter.split_text(body)
        if len(split_body_list) == 1:
            # 切分后仍为一段，视为无需切分
            return [section]
        # 每个分片生成带序号标题的 section
        result_sections = []
        for index,split_body in enumerate(split_body_list):
            split_section = {
                "body":split_body,
                "title":f"{title}_{index + 1}",
                "parent_title": section.get("parent_title"),
                "file_title": section.get("file_title")
            }

            result_sections.append(split_section)

        return result_sections

    def _merge_short_section(self, current_sections:List[Dict[str,Any]], min_content_length:int) -> List[Dict[str,Any]]:
        """合并过短的 section：同源且不足最小长度时与下一段拼接。"""
        current_section = current_sections[0]
        final_sections = []
        for i in range(1,len(current_sections)):
            next_section = current_sections[i]
            # 同源（父标题一致）且当前段过短 → 合并
            same_parent = current_section.get("parent_title") == next_section.get("parent_title")
            current_section_body = current_section.get("body")
            is_short = len(current_section_body) < min_content_length
            if is_short and same_parent:
                # 拼接两段正文，合并后标题取父标题
                merged_body = current_section_body + "\n\n" + next_section.get("body")
                current_section["body"] = merged_body
                current_section["title"] = current_section.get("parent_title")


            else:
                # 不可合并：当前段定稿，指针后移
                final_sections.append(current_section)
                current_section = next_section

        # 收尾：追加最后一段
        final_sections.append(current_section)
        return final_sections

    def _assemble_chunks(self, final_sections:List[Dict[str,Any]]) -> List[Dict[str,Any]]:
        """
        section → chunk：content 为标题与正文拼接，其余字段保留。
        """
        chunks = []
        # 每个 section 生成一个 chunk
        for section in final_sections:
            title = section.get("title")
            body = section.get("body")
            chunk = {
                "content": f"{title}\n\n{body}",
                "title": title,
                "parent_title": section.get("parent_title"),
                "file_title": section.get("file_title")
            }
            chunks.append(chunk)
        return chunks

    def _backup_chunks(self, chunks:List[Dict[str,Any]],state:ImportGraphState):
        # 备份到 输出目录/<md 文件名>/chunks.json，失败不影响主流程
        md_path = state.get("md_path")
        md_path_obj = Path(md_path)
        file_dir = state.get("file_dir")
        file_dir_obj = Path(file_dir)
        backup_dir = file_dir_obj / md_path_obj.stem
        backup_dir.mkdir(parents=True,exist_ok=True)
        backup_file_path = backup_dir / "chunks.json"

        try:
            with open(backup_file_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=4)
        except Exception as e:
            self.logger.warning(f"{md_path_obj.stem}文件备份成chunks.json失败,但是不影响主流程")