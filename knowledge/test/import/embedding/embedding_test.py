"""BGE-M3 嵌入向量生成离线测试：稠密 + 稀疏向量解析流程验证。"""

from pymilvus.model.hybrid import BGEM3EmbeddingFunction

# 创建嵌入模型对象
bge_m3 = BGEM3EmbeddingFunction(
    model_name=r"D:\ai_models\modelscope_cache\models\BAAI\bge-m3",

    # device 与 use_fp16 需与本地 torch 版本匹配（GPU 用 cuda:0/True，CPU 用 cpu/False）
    device="cuda:0",
    use_fp16=True,
)

# 生成嵌入
document_list = ["我是中国人","你是美国人"]

result = bge_m3.encode_documents(document_list)

# 解析稠密向量
dense = result.get("dense")
dense_vector_list = []
for item in dense:
    dense_vector = item.tolist()
    dense_vector_list.append(dense_vector)

# 解析稀疏向量（CSR 矩阵 → {token_id: weight}）
# indices: 非零维度的 token_id；data: 对应权重；indptr: 每条向量的起止下标（不含结束）
sparse_csr = result.get("sparse")
indptr = sparse_csr.indptr

sparse_vector_list = []

for i in range(0,len(document_list)):
    start_index = indptr[i]
    end_index = indptr[i+1]

    indices = sparse_csr.indices[start_index:end_index].tolist()
    data = sparse_csr.data[start_index:end_index].tolist()

    sparse_vector = dict(zip(indices, data))

    sparse_vector_list.append(sparse_vector)

print(sparse_vector_list)
