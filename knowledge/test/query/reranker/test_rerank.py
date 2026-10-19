"""BGE-Reranker 相关性打分离线测试：query-doc pair 打分验证。"""

from knowledge.utils.clients.ai_clients import AIClients

reranker = AIClients.get_bge_m3_rerank_client()

# 计算相关性得分
pairs = [
    ["什么是万用表？", "万用表是一种测量电压、电流、电阻的仪器"],
    ["什么是万用表？", "今天天气很好"]
]
scores = reranker.compute_score(pairs)
# 分值越高代表相关性越强（示例输出约为 [7.9, -9.5]）
print(scores)
