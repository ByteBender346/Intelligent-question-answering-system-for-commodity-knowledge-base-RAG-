"""离线回归：MCP 传输协议选择及查询失败的 SSE 通知。"""
import asyncio
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from knowledge.processor.query_processor.config import QueryConfig
from knowledge.processor.query_processor.nodes.web_search_node import WebSearchNode
from knowledge.utils import sse_util, task_util


class WebSearchTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_transport_and_authorization(self):
        for path, selected in (("/WebSearch/sse", "sse"), ("/WebSearch/sse/", "sse"),
                               ("/WebSearch/sse?region=cn", "sse"), ("/WebSearch/mcp", "http")):
            for key in ("test-key", "Bearer test-key"):
                with self.subTest(path=path, key=key):
                    node = WebSearchNode(QueryConfig(
                        mcp_dashscope_base_url="https://example.com" + path,
                        openai_api_key=key,
                    ))
                    client = MagicMock()
                    client.__aenter__ = AsyncMock(return_value=client)
                    client.__aexit__ = AsyncMock(return_value=False)
                    result = object()
                    client.call_tool = AsyncMock(return_value=result)
                    legacy_client = MagicMock()
                    legacy_client.__aenter__ = AsyncMock(return_value=legacy_client)
                    legacy_client.__aexit__ = AsyncMock(return_value=False)
                    legacy_client.session.call_tool = AsyncMock(return_value=result)
                    with patch("knowledge.processor.query_processor.nodes.web_search_node.MCPServerSse",
                               return_value=client) as sse, patch(
                            "knowledge.processor.query_processor.nodes.web_search_node.MCPServerStreamableHttp",
                            return_value=client) as http, patch(
                            "knowledge.processor.query_processor.nodes.web_search_node.MCPClient",
                            return_value=legacy_client) as legacy:
                        self.assertIs(await node.mcp_web_search("测试问题"), result)
                    chosen, other = (sse, http) if selected == "sse" else (http, sse)
                    chosen.assert_called_once()
                    other.assert_not_called()
                    params = chosen.call_args.kwargs["params"]
                    self.assertEqual(params["headers"]["Authorization"], "Bearer test-key")
                    self.assertEqual("terminate_on_close" in params, selected == "http")
                    if selected == "sse":
                        legacy.assert_not_called()
                        client.call_tool.assert_awaited_once_with(
                            tool_name="bailian_web_search", arguments={"query": "测试问题", "count": 3})
                    else:
                        legacy.assert_called_once_with(
                            client.create_streams.return_value, mode="legacy", cache=None,
                            read_timeout_seconds=300)
                        client.create_streams.assert_called_once_with()
                        client.__aenter__.assert_not_awaited()
                        client.call_tool.assert_not_awaited()
                        legacy_client.session.call_tool.assert_awaited_once_with(
                            name="bailian_web_search", arguments={"query": "测试问题", "count": 3})
                        legacy_client.__aexit__.assert_awaited_once_with(None, None, None)

    async def test_http_tool_failure_closes_legacy_client(self):
        node = WebSearchNode(QueryConfig(
            mcp_dashscope_base_url="https://example.com/mcp", openai_api_key="test-key"))
        server = MagicMock()
        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        error = RuntimeError("tool failed")
        client.session.call_tool = AsyncMock(side_effect=error)
        with patch("knowledge.processor.query_processor.nodes.web_search_node.MCPServerStreamableHttp",
                   return_value=server), patch(
                "knowledge.processor.query_processor.nodes.web_search_node.MCPClient",
                return_value=client):
            with self.assertRaisesRegex(RuntimeError, "tool failed"):
                await node.mcp_web_search("test")
        client.__aexit__.assert_awaited_once()
        self.assertIs(client.__aexit__.await_args.args[1], error)
        server.__aenter__.assert_not_awaited()

    async def test_missing_credentials(self):
        for url, key in (("", "test-key"), ("https://example.com/sse", "")):
            node = WebSearchNode(QueryConfig(mcp_dashscope_base_url=url, openai_api_key=key))
            with self.assertRaisesRegex(ValueError, "MCP_DASHSCOPE_BASE_URL"):
                await node.mcp_web_search("测试")

    async def test_missing_sse_endpoint_event_has_actionable_error(self):
        node = WebSearchNode(QueryConfig(
            mcp_dashscope_base_url="https://example.com/sse", openai_api_key="test-key"))
        client = MagicMock()
        client.__aenter__ = AsyncMock(side_effect=ExceptionGroup(
            "unhandled errors in a TaskGroup", [RuntimeError(
                "Child exited without calling task_status.started()")]))
        client.__aexit__ = AsyncMock(return_value=False)
        with patch("knowledge.processor.query_processor.nodes.web_search_node.MCPServerSse",
                   return_value=client):
            with self.assertRaisesRegex(RuntimeError, "没有发送 endpoint 事件"):
                await node.mcp_web_search("测试")


class SSELifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_terminal_event_delivered_then_queue_removed(self):
        for event in (sse_util.SSEEvent.FINAL, sse_util.SSEEvent.ERROR):
            with self.subTest(event=event):
                task_id = "terminal-" + event
                sse_util.create_sse_queue(task_id)
                sse_util.push_sse_event(task_id, sse_util.SSEEvent.DELTA, {"delta": "部分答案"})
                sse_util.push_sse_event(task_id, event, {"error": "网络搜索失败"})
                request = SimpleNamespace(is_disconnected=AsyncMock(return_value=False))
                chunks = await asyncio.wait_for(self.collect(task_id, request), timeout=3)
                self.assertEqual(len(chunks), 2)
                self.assertIn("event: " + event, chunks[-1])
                self.assertIn("网络搜索失败", chunks[-1])
                self.assertIsNone(sse_util.get_sse_queue(task_id))

    async def collect(self, task_id, request):
        return [chunk async for chunk in sse_util.sse_generator(task_id, request)]

    async def test_idle_stream_emits_heartbeat(self):
        task_id = "idle-stream"
        sse_util.create_sse_queue(task_id)
        request = SimpleNamespace(is_disconnected=AsyncMock(return_value=False))
        stream = sse_util.sse_generator(task_id, request)
        try:
            self.assertEqual(await asyncio.wait_for(anext(stream), 3), ": heartbeat\n\n")
        finally:
            await stream.aclose()
        self.assertIsNone(sse_util.get_sse_queue(task_id))


class QueryFailureTests(unittest.TestCase):
    def setUp(self):
        # 不加载真实检索图及 MongoDB；执行实际 QueryService 错误处理。
        graph = ModuleType("knowledge.processor.query_processor.main_graph")
        graph.query_app = MagicMock()
        history = ModuleType("knowledge.utils.mongo_history_util")
        history.clear_history = MagicMock()
        history.get_recent_messages = MagicMock()
        service_path = Path(__file__).resolve().parents[2] / "service" / "query_service.py"
        spec = importlib.util.spec_from_file_location("query_service_under_test", service_path)
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {graph.__name__: graph, history.__name__: history}):
            spec.loader.exec_module(self.module)
        self.service = self.module.QueryService()
        graph.query_app.invoke.side_effect = RuntimeError("网络搜索连接失败")

    def test_stream_failure_is_sent_to_frontend(self):
        task_id = "stream-failure"
        q = sse_util.create_sse_queue(task_id)
        try:
            with self.assertLogs(self.module.logger, level="ERROR"):
                self.service.run_query_graph("问题", "session", task_id, True)
            self.assertEqual(task_util.get_task_status(task_id), task_util.TASK_STATUS_FAILED)
            self.assertEqual(task_util.get_task_result(task_id, "error"), "网络搜索连接失败")
            self.assertEqual(q.get_nowait(), {
                "event": "error", "data": {"error": "网络搜索连接失败", "status": "failed"}})
        finally:
            sse_util.remove_sse_queue(task_id)

    def test_non_stream_failure_still_raises(self):
        with self.assertLogs(self.module.logger, level="ERROR"), self.assertRaisesRegex(
                RuntimeError, "网络搜索连接失败"):
            self.service.run_query_graph("问题", "session", "sync-failure", False)
        self.assertEqual(task_util.get_task_status("sync-failure"), task_util.TASK_STATUS_FAILED)
