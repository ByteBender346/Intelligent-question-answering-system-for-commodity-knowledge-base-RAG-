"""下载 BGE-M3 模型到本地目录。"""

from modelscope import snapshot_download

local_dir = snapshot_download(model_id="BAAI/bge-m3", local_dir="你的dir")

print(local_dir)
