"""查询流程自定义异常：按节点/状态字段/检索/LLM/存储等场景细分。"""


class QueryProcessError(Exception):
    """查询流程基础异常，携带出错节点名与原始原因。"""

    def __init__(
            self,
            message: str,
            node_name: str = "",
            cause: Exception = None
    ):
        self.node_name = node_name
        self.cause = cause
        super().__init__(message)

    def __str__(self):
        parts = []
        if self.node_name:
            parts.append(f"[{self.node_name}]")
        parts.append(super().__str__())
        if self.cause:
            parts.append(f"(原因: {self.cause})")
        return " ".join(parts)


class StateFieldError(QueryProcessError):
    """状态字段缺失、为空或类型不符时抛出。"""

    def __init__(
            self,
            node_name: str = "",
            field_name: str = "",
            expected_type: type = None,
            message: str = "",
            cause: Exception = None,
    ):
        self.field_name = field_name
        self.expected_type = expected_type
        if not message:
            message = f"状态字段 '{field_name}' 缺失或无效"
            if expected_type:
                message += f"，期望类型: {expected_type.__name__}"
        super().__init__(message, node_name=node_name, cause=cause)


class ConfigurationError(QueryProcessError):
    """环境变量缺失或配置值无效。"""
    pass


class SearchError(QueryProcessError):
    """向量检索、混合检索或网络检索失败。"""
    pass


class EmbeddingError(QueryProcessError):
    """向量化模型调用失败。"""
    pass


class LLMError(QueryProcessError):
    """大模型 API 调用或响应解析失败。"""
    pass


class StorageError(QueryProcessError):
    """数据库操作失败。"""
    pass


class MilvusError(StorageError):
    """Milvus 向量数据库操作失败。"""
    pass


class MongoDBError(StorageError):
    """MongoDB 操作失败。"""
    pass


class ValidationError(QueryProcessError):
    """输入数据不符合预期。"""
    pass


class EntityAlignmentError(QueryProcessError):
    """商品名实体对齐失败。"""
    pass


class RerankError(QueryProcessError):
    """重排序过程失败。"""
    pass


class ItemNameConfirmError(QueryProcessError):
    """商品名识别或确认失败。"""
    pass
