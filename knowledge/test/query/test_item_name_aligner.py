import importlib.util
import logging
from pathlib import Path
from types import ModuleType
import unittest
from unittest.mock import MagicMock, patch

from knowledge.processor.query_processor.config import QueryConfig


class ItemNameAlignerTests(unittest.TestCase):
    def setUp(self):
        dependencies = {
            "knowledge.utils.clients.ai_clients": ["AIClients"],
            "knowledge.utils.clients.storage_clients": ["StorageClients"],
            "knowledge.utils.embedding_util": ["generate_bge_m3_hybrid_vectors"],
            "knowledge.utils.milvus_util": [
                "create_hybrid_search_requests", "execute_hybrid_search_query", "milvus_client"],
            "knowledge.utils.mongo_history_util": ["get_recent_messages"],
        }
        modules = {}
        for name, attributes in dependencies.items():
            module = ModuleType(name)
            for attribute in attributes:
                setattr(module, attribute, MagicMock())
            modules[name] = module
        source = Path(__file__).resolve().parents[2] / "processor/query_processor/nodes/item_name_confirmed_node.py"
        spec = importlib.util.spec_from_file_location("item_name_node_under_test", source)
        node_module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", modules):
            spec.loader.exec_module(node_module)
        config = QueryConfig(item_name_high_confidence=0.75, item_name_mid_confidence=0.45,
                             item_name_score_gap=0.08, item_name_max_options=3)
        self.aligner = node_module._ItemNameAligner(logging.getLogger(__name__), config)

    def align(self, search_results):
        with patch.object(self.aligner, "_search_vector", return_value=search_results):
            return self.aligner.search_and_align([result["extracted_name"] for result in search_results])

    def hit(self, name, score):
        return {"extracted_name": name, "matches": [{"item_name": name, "score": score}]}

    def test_multiple_confirmed_are_filtered_and_returned(self):
        confirmed, options = self.align([self.hit("product-a", 0.98), self.hit("product-b", 0.80)])
        self.assertEqual(confirmed, ["product-a"])
        self.assertEqual(options, [])

    def test_close_scores_keep_both_products_in_order(self):
        confirmed, options = self.align([self.hit("product-b", 0.91), self.hit("product-a", 0.95)])
        self.assertEqual(confirmed, ["product-b", "product-a"])
        self.assertEqual(options, [])

    def test_repeated_matches_use_highest_score(self):
        results = [self.hit("product-a", 0.98), self.hit("product-b", 0.94), self.hit("product-a", 0.76)]
        confirmed, _ = self.align(results)
        self.assertEqual(confirmed, ["product-a", "product-b"])

    def test_options_are_preserved_when_confirmed_are_filtered(self):
        confirmed, options = self.align([
            self.hit("product-a", 0.98), self.hit("product-b", 0.80), self.hit("product-c", 0.60)])
        self.assertEqual(confirmed, ["product-a"])
        self.assertEqual(options, ["product-c"])

    def test_single_confirmed_skips_score_filter(self):
        with patch.object(self.aligner, "_item_name_score_filter") as score_filter:
            self.assertEqual(self.align([self.hit("product-a", 0.90)]), (["product-a"], []))
            score_filter.assert_not_called()

    def test_no_results_returns_empty_lists(self):
        self.assertEqual(self.align([]), ([], []))

    def test_only_options_skip_score_filter(self):
        with patch.object(self.aligner, "_item_name_score_filter") as score_filter:
            self.assertEqual(self.align([self.hit("product-a", 0.60)]), ([], ["product-a"]))
            score_filter.assert_not_called()


if __name__ == "__main__":
    unittest.main()
