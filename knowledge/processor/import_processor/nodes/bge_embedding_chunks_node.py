"""BGE 向量化节点：为每个 chunk 生成稠密 + 稀疏混合向量并备份。"""
import json
from pathlib import Path
from typing import Any, Dict, List

from knowledge.processor.import_processor.base import BaseNode, T
from knowledge.processor.import_processor.exceptions import StateFieldError, ValidationError, EmbeddingError
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.clients.ai_clients import AIClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors


class BgeEmbeddingChunksNode(BaseNode):
    name = "bge_embedding_chunks_node"
    def process(self, state: ImportGraphState) -> ImportGraphState:
        chunks = self._validate(state)
        # 每个切片生成 BGE-M3 稠密 + 稀疏向量
        final_chunks = self.embedding_chunks(chunks)
        state['chunks'] = final_chunks
        # 备份为 json，便于后续节点单独测试
        self._backup_chunks(final_chunks,state)

        return state

    def _validate(self, state:ImportGraphState):
        chunks = state.get("chunks")
        # chunks 必须为非空 list，且每个元素为 dict
        if not chunks or not isinstance(chunks, list):
            self.logger.error("chunks 为空或者不是list")
            raise StateFieldError(node_name=self.name, field_name="chunks",expected_type=list)
        for chunk in chunks:
            if not isinstance(chunk, dict):
                self.logger.error("chunk 类型不是dict")
                raise ValidationError(node_name=self.name,message="chunk 类型不是dict")

        return chunks

    def embedding_chunks(self, chunks:List[Dict[str,Any]]):
        # 获取 BGE-M3 嵌入模型
        try:
            embedding_client = AIClients.get_bge_m3_client()
        except Exception as e:
            self.logger.error(f"获取嵌入模型对象失败,{e}")
            raise EmbeddingError(node_name=self.name,message=f"获取嵌入模型对象失败,{e}")

        # 按批次批量嵌入：步长为 embedding_batch_size
        embedding_batch_size = self.config.embedding_batch_size
        total_length = len(chunks)
        final_chunks = []
        for i in range(0,total_length,embedding_batch_size):
            batch_chunks = chunks[i:i+embedding_batch_size]
            # 将每个 chunk 的 item_name 与 content 拼接后向量化
            document_list = [f"{chunk.get('item_name')}\n{chunk.get('content')}" for chunk in batch_chunks]
            try:
                embedding_result = generate_bge_m3_hybrid_vectors(embedding_client,document_list)
            except Exception as e:
                self.logger.error(f"嵌入向量失败,{e}")
                raise EmbeddingError(node_name=self.name, message=f"嵌入向量失败,{e}")

            # 取出稠密向量与稀疏向量
            dense_vector_list = embedding_result.get("dense")
            sparse_vector_list = embedding_result.get("sparse")

            # 为每个 chunk 附加稠密、稀疏向量
            for i,chunk in enumerate(batch_chunks):
                chunk["dense_vector"] = dense_vector_list[i]
                chunk["sparse_vector"] = sparse_vector_list[i]
                final_chunks.append(chunk)

        return final_chunks

    def _backup_chunks(self, final_chunks:List[Dict[str,Any]],state:ImportGraphState):
        # 备份目录：输出目录/<md 文件名>/
        md_path = state.get("md_path")
        md_path_obj = Path(md_path)
        file_dir = state.get("file_dir")
        file_dir_obj = Path(file_dir)
        backup_dir = file_dir_obj / md_path_obj.stem
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_file_path = backup_dir / "chunks_vector.json"

        # 写入备份文件
        try:

            with open(backup_file_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=4)
        except Exception as e:
            self.logger.warning(f"{md_path_obj.stem}文件备份成chunks_vector.json失败,但是不影响主流程")
