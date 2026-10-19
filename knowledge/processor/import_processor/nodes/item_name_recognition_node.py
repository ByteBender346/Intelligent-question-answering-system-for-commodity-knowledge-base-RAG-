"""商品名识别节点：LLM 从文档切片中提取商品名，向量化后入库 Milvus，并回写每个 chunk。"""
import json
from pathlib import Path
from typing import Tuple, Any, Dict, List

from dns.message import IndexType
from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus import DataType

from knowledge.processor.import_processor.base import BaseNode
from knowledge.processor.import_processor.exceptions import StateFieldError, ConfigurationError
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.prompts.import_prompt import ITEM_NAME_SYSTEM_PROMPT, ITEM_NAME_USER_PROMPT_TEMPLATE
from knowledge.utils.clients.ai_clients import AIClients
from knowledge.utils.clients.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors


class ItemNameRecognitionNode(BaseNode):
    name = "item_name_recognition_node"
    def process(self, state: ImportGraphState) -> ImportGraphState:
        file_title, chunks, item_name_chunk_k, item_name_chunk_size = self._validate_state(state)
        # 取前 k 个切片、累计不超过设定长度，作为 LLM 上下文
        context = self._prepare_item_name_recognition_context(chunks,item_name_chunk_k,item_name_chunk_size)
        # LLM 识别商品名
        item_name = self._recognition_item_name(context,file_title)

        # 商品名向量化
        hybrid_vectors = self._embedding_item_name(item_name)

        # 商品名向量入库 Milvus
        self._insert_to_milvus(item_name,hybrid_vectors)

        # 将商品名回写到每个 chunk，供下游节点使用
        for chunk in chunks:
            if not isinstance(chunk,dict):
                continue
            chunk["item_name"] = item_name

        state["item_name"] = item_name
        state["chunks"] = chunks

        # 备份 chunks，便于后续测试
        self._backup_chunks(chunks,state)

        return state


    def _validate_state(self, state:ImportGraphState) -> Tuple[str,list,int,int]:
        """校验 file_title、chunks 及商品名提取配置项。"""
        # file_title 作为提取失败时的兜底商品名
        file_title = state.get("file_title")
        if not file_title:
            raise StateFieldError(node_name=self.name,field_name="file_title",expected_type=str)
        # 上一步切分得到的 chunks，作为 LLM 上下文来源
        chunks = state.get("chunks")
        if not chunks or not isinstance(chunks,list):
            raise StateFieldError(node_name=self.name,field_name="chunks",expected_type=list)
        # 商品名提取可用的最大 chunk 数与最大长度
        item_name_chunk_k = self.config.item_name_chunk_k
        item_name_chunk_size = self.config.item_name_chunk_size

        if not item_name_chunk_size or item_name_chunk_size <= 0:
            raise ConfigurationError(node_name=self.name,message="item_name_chunk_size must be greater than 0")

        if not item_name_chunk_k or item_name_chunk_k <= 0:
            raise ConfigurationError(node_name=self.name,message="item_name_chunk_k must be greater than 0")
        return file_title, chunks, item_name_chunk_k, item_name_chunk_size

    def _prepare_item_name_recognition_context(self, chunks:List[Dict[str,Any]], item_name_chunk_k:int, item_name_chunk_size:int) -> str:
        """拼接前 k 个切片的 content，累计长度超过 item_name_chunk_size 即停止。"""
        final_context = []
        total_length = 0
        for index,chunk in enumerate(chunks[:item_name_chunk_k]):
            if not isinstance(chunk,dict):
                continue
            content = chunk.get("content")
            context = f"【切片】-{index+1}-{content}"
            final_context.append(context)
            total_length += len(context)
            # 超过最大长度后不再追加后续切片
            if total_length >= item_name_chunk_size:
                break

        return "\n".join(final_context)

    def _recognition_item_name(self, context:str, file_title:str):
        # 获取 LLM 客户端，失败时回退为 file_title
        try:
            llm_client = AIClients.get_llm_client(False)
        except Exception as e:
            self.logger.warning(f"获取llm对象失败,{e}")
            return file_title
        # 组装 System / Human 消息
        system_prompt = ITEM_NAME_SYSTEM_PROMPT
        system_message = SystemMessage(content=system_prompt)

        user_prompt = ITEM_NAME_USER_PROMPT_TEMPLATE.format(
            file_title = file_title,
            context = context
        )
        human_message = HumanMessage(content=user_prompt)
        # 调用 LLM 提取商品名
        try:
            llm_res = llm_client.invoke([
                system_message,
                human_message
            ])
        except Exception as e:
            self.logger.warning(f"调用llm失败,失败信息为{e}")
            return file_title
        item_name = llm_res.content.strip()

        # LLM 返回 UNKNOWN 时兜底为 file_title
        if item_name == "UNKNOWN":
            return file_title

        return item_name

    def _embedding_item_name(self, item_name:str) -> Dict[list,list]:
        # BGE-M3 为商品名生成混合向量
        try:
            embedding_client = AIClients.get_bge_m3_client()
        except Exception as e:
            self.logger.warning(f"获取嵌入模型失败,{e}")
            return None
        hybrid_vectors = generate_bge_m3_hybrid_vectors(embedding_client,[item_name])
        return hybrid_vectors

    def _insert_to_milvus(self, item_name:str, hybrid_vectors:dict):
        # 获取 Milvus 客户端，失败直接跳过
        try:
            milvus_client = StorageClients.get_milvus_client()
        except Exception as e:
            self.logger.warning(f"获取Milvus客户端失败,{e}")
            return
        # item_name 集合不存在时自动创建：含稠密向量(COSINE)与稀疏向量(IP)索引
        collection_name = self.config.item_name_collection
        if not milvus_client.has_collection(collection_name):
            # 建表 Schema
            schema = milvus_client.create_schema()
            schema.add_field(
                field_name="pk",
                datatype=DataType.INT64,
                auto_id=True,
                is_primary=True
            )
            schema.add_field(
                field_name="item_name",
                datatype=DataType.VARCHAR,
                max_length=2048,
            )
            schema.add_field(
                field_name="dense_vector",
                datatype=DataType.FLOAT_VECTOR,
                dim=1024
            )
            schema.add_field(
                field_name="sparse_vector",
                datatype=DataType.SPARSE_FLOAT_VECTOR
            )
            # 建立索引
            index_params = milvus_client.prepare_index_params()
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_vector_index",
                index_type="AUTOINDEX",
                metric_type="COSINE"
            )
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_vector_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP"
            )

            milvus_client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params
            )
        # 取出稠密、稀疏向量并插入
        dense_vector = hybrid_vectors.get("dense")[0]
        sparse_vector = hybrid_vectors.get("sparse")[0]

        if not dense_vector or not sparse_vector:
            self.logger.warning("稠密向量或者稀疏向量为空")
            return

        insert_data = {
            "item_name":item_name,
            "dense_vector":dense_vector,
            "sparse_vector":sparse_vector
        }
        milvus_client.insert(
            collection_name=collection_name,
            data=insert_data,
            timeout=30
        )

    def _backup_chunks(self, chunks:List[Dict[str,Any]],state:ImportGraphState):
        # 备份目录：输出目录/<md 文件名>/
        md_path = state.get("md_path")
        md_path_obj = Path(md_path)
        file_dir = state.get("file_dir")
        file_dir_obj = Path(file_dir)
        backup_dir = file_dir_obj / md_path_obj.stem
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_file_path = backup_dir / "chunks_item_name.json"

        try:
            with open(backup_file_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=4)
        except Exception as e:
            self.logger.warning(f"{md_path_obj.stem}文件备份成chunks_item_name.json失败,但是不影响主流程")
