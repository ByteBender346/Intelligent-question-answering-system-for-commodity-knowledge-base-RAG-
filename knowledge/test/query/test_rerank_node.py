"""RerankNode 单元测试：模型加载/推理失败兜底、双路文档排序与断崖裁剪。"""

import importlib.util
from pathlib import Path
from types import ModuleType
import unittest
from unittest.mock import MagicMock, patch


class RerankNodeTests(unittest.TestCase):
    def setUp(self):
        clients = ModuleType("knowledge.utils.clients.ai_clients")
        clients.AIClients = MagicMock()
        source = Path(__file__).resolve().parents[2] / "processor/query_processor/nodes/rerank_node.py"
        spec = importlib.util.spec_from_file_location("rerank_node_under_test", source)
        module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {clients.__name__: clients}):
            spec.loader.exec_module(module)
        self.node = module.RerankNode()
        self.factory = clients.AIClients.get_bge_m3_rerank_client
        self.client = self.factory.return_value

    def state(self):
        return {
            "rewritten_query": "test question",
            "rrf_chunks": [{"chunk_id": "local", "content": "local document"}],
            "web_search_docs": [{"url": "https://example.com", "snippet": "web document"}],
        }

    def test_model_load_failure_preserves_order_with_numeric_scores(self):
        self.factory.side_effect = FileNotFoundError("model directory missing")
        with self.assertLogs(self.node.logger, level="WARNING") as logs:
            result = self.node.process(self.state())["reranked_docs"]
        self.assertEqual([doc["source"] for doc in result], ["local", "web"])
        self.assertEqual([float(doc["score"]) for doc in result], [0.0, 0.0])
        self.assertIn("model directory missing", "\n".join(logs.output))

    def test_prediction_failure_preserves_documents(self):
        self.client.compute_score.side_effect = RuntimeError("inference failed")
        with self.assertLogs(self.node.logger, level="WARNING"):
            result = self.node.process(self.state())["reranked_docs"]
        self.assertEqual(len(result), 2)
        self.assertTrue(all(doc["score"] == 0.0 for doc in result))

    def test_successful_scores_sort_both_sources(self):
        self.client.compute_score.return_value = [-2.0, 2.0]
        result = self.node.process(self.state())["reranked_docs"]
        self.assertEqual([doc["source"] for doc in result], ["web", "local"])
        self.assertTrue(all(0 <= doc["score"] <= 1 for doc in result))
        self.client.compute_score.assert_called_once_with([
            ("test question", "local document"), ("test question", "web document")])

    def test_missing_or_empty_branches_do_not_load_model(self):
        for inputs in ({}, {"rrf_chunks": None, "web_search_docs": None},
                       {"rrf_chunks": [], "web_search_docs": []}):
            with self.subTest(inputs=inputs):
                self.assertEqual(self.node.process({"rewritten_query": "test", **inputs})["reranked_docs"], [])
        self.factory.assert_not_called()

    def test_single_branch_and_scalar_score(self):
        self.client.compute_score.return_value = 2.0
        result = self.node.process({"rewritten_query": "test", "web_search_docs": [
            {"snippet": "web document", "url": "https://example.com"}]})["reranked_docs"]
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["source"], "web")
        self.assertAlmostEqual(result[0]["score"], self.node._sigmoid(2.0))

    def test_invalid_document_contents_are_skipped(self):
        invalid_docs = [None, "invalid", {}, {"content": None, "snippet": None},
                        {"content": "  ", "snippet": "  "}, {"content": 123, "snippet": 123}]
        result = self.node.process({"rewritten_query": "test", "rrf_chunks": invalid_docs,
                                    "web_search_docs": invalid_docs})
        self.assertEqual(result["reranked_docs"], [])
        self.factory.assert_not_called()

    def test_invalid_model_outputs_do_not_drop_documents(self):
        for scores in ([1.0], [1.0, 2.0, 3.0], [None, 1.0], [float("nan"), 1.0],
                       [float("inf"), 1.0], [float("-inf"), 1.0], None):
            with self.subTest(scores=scores):
                self.client.compute_score.return_value = scores
                with self.assertLogs(self.node.logger, level="WARNING"):
                    result = self.node.process(self.state())["reranked_docs"]
                self.assertEqual(len(result), 2)
                self.assertEqual([doc["score"] for doc in result], [0.0, 0.0])

    def test_extreme_scores_do_not_overflow(self):
        self.client.compute_score.return_value = [-1000, 1000]
        result = self.node.process(self.state())["reranked_docs"]
        self.assertEqual([doc["score"] for doc in result], [1.0, 0.0])
        self.assertEqual(self.node._sigmoid(0), 0.5)

    def test_cliff_retains_document_before_gap(self):
        docs = [{"score": score} for score in (0.99, 0.98, 0.97, 0.96, 0.1, 0.09)]
        self.assertEqual(self.node._cliff_cutoff(docs, 3, 10, 0.08), docs[:4])

    def test_cliff_ignores_gaps_before_minimum(self):
        docs = [{"score": score} for score in (0.99, 0.2, 0.19, 0.18, 0.02)]
        self.assertEqual(self.node._cliff_cutoff(docs, 3, 10, 0.08), docs[:4])

    def test_cutoff_respects_limits_without_a_gap(self):
        docs = [{"score": 0.0} for _ in range(12)]
        self.assertEqual(self.node._cliff_cutoff(docs, 3, 10, 0.08), docs[:10])
        self.assertEqual(self.node._cliff_cutoff(docs[:2], 3, 10, 0.08), docs[:2])
        self.assertEqual(self.node._cliff_cutoff([], 3, 10, 0.08), [])
