"""Milvus 入库节点：将带稠密+稀疏向量的 chunks 批量写入 Milvus，回填自增主键。"""
import json
from typing import Dict, Any, List

from pymilvus import MilvusClient, DataType

from knowledge.processor.import_processor.base import BaseNode, T
from knowledge.processor.import_processor.exceptions import StateFieldError, EmbeddingError, MilvusError
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.clients.storage_clients import StorageClients

class _IndexParamsBuilder:
    @classmethod
    def build_index_params(cls,milvus_client:MilvusClient):
        # 稠密向量用 COSINE、稀疏向量用稀疏倒排索引 + IP
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
        return index_params

class _MilvusSchemaBuilder:
    @classmethod
    def build_schema(cls,milvus_client:MilvusClient,dim:int):
        # 主键 + 标量字段 + 稠密/稀疏向量字段
        schema = milvus_client.create_schema()
        schema.add_field(
            field_name="id",
            datatype=DataType.INT64,
            auto_id=True,
            is_primary=True
        )
        schema.add_field(
            field_name="item_name",
            datatype=DataType.VARCHAR,
            max_length=65535
        )
        schema.add_field(
            field_name="title",
            datatype=DataType.VARCHAR,
            max_length=65535
        )
        schema.add_field(
            field_name="parent_title",
            datatype=DataType.VARCHAR,
            max_length=65535
        )
        schema.add_field(
            field_name="file_title",
            datatype=DataType.VARCHAR,
            max_length=65535
        )
        schema.add_field(
            field_name="content",
            datatype=DataType.VARCHAR,
            max_length=65535
        )
        # BGE-M3 混合向量
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=dim
        )
        schema.add_field(
            field_name="sparse_vector",
            datatype=DataType.SPARSE_FLOAT_VECTOR
        )
        return schema

class MilvusImportNode(BaseNode):
    name = "milvus_import_node"
    def process(self, state: ImportGraphState) -> ImportGraphState:
        final_chunks,dim = self._validate_state(state)
        # 向量入库
        self.insert_data(final_chunks,dim)

        state["chunks"] = final_chunks
        return state

    def _validate_state(self, state:ImportGraphState):
        chunks = state.get("chunks")
        if not chunks or not isinstance(chunks, list):
            self.logger.error("chunks 为空，或者类型不是list")
            raise StateFieldError(node_name=self.name,field_name="chunks",expected_type=list)
        # 仅保留同时带有稠密与稀疏向量的 chunk，缺向量的丢弃
        final_chunks = []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            dense_vector = chunk.get("dense_vector")
            sparse_vector = chunk.get("sparse_vector")
            if not dense_vector or not sparse_vector:
                continue

            # dim 取自稠密向量长度，用于建集合
            dim = len(dense_vector)
            final_chunks.append(chunk)

        if not final_chunks:
            self.logger.error("所有chunk都没有向量,导入失败")
            raise EmbeddingError("所有chunk都没有向量,导入失败")



        return final_chunks,dim

    def insert_data(self, final_chunks:List[Dict[str,Any]], dim):
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"获取milvus客户端失败,{e}")
            raise MilvusError(f"获取milvus客户端失败,{e}")
        collection_name = self.config.chunks_collection
        # 集合不存在时按 Schema + 索引参数创建
        if not milvus_client.has_collection(collection_name):
            schema = _MilvusSchemaBuilder.build_schema(milvus_client,dim)
            index_params = _IndexParamsBuilder.build_index_params(milvus_client)
            milvus_client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params,
                timeout=30
            )
        # 插入并取回自增主键 id
        insert_results = milvus_client.insert(
            collection_name=collection_name,
            data=final_chunks
        )
        ids = insert_results.get("ids")
        # 将自增 id 回填到每个 chunk
        for i,chunk in enumerate(final_chunks):
            chunk["chunk_id"] = ids[i]
