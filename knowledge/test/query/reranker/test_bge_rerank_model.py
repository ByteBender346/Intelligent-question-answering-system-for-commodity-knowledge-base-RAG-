"""下载 BGE-Reranker 模型到本地目录。"""

from modelscope import snapshot_download

local_dir = snapshot_download(model_id="BAAI/bge-reranker-large", local_dir=r"D:\ai_models\modelscope_cache\models\BAAI\bge-reranker-large")

print(local_dir)
