"""网络检索节点：通过 MCP 调用百炼联网搜索工具，返回网页标题/摘要/链接。"""
import asyncio
import json
from urllib.parse import urlsplit

from agents.mcp import MCPServerSse, MCPServerStreamableHttp
from mcp import Client as MCPClient

from knowledge.processor.query_processor.base import BaseNode, T
from knowledge.processor.query_processor.exceptions import StateFieldError
from knowledge.processor.query_processor.state import QueryGraphState


def _exception_leaves(error):
    """展开 ExceptionGroup 嵌套，逐个产出叶子异常。"""
    if isinstance(error, BaseExceptionGroup):
        for nested in error.exceptions:
            yield from _exception_leaves(nested)
    else:
        yield error


class WebSearchNode(BaseNode):
    """网络检索节点：MCP 调用百炼 bailian_web_search，解析返回 JSON 的 pages 为统一文档列表。"""
    name = "web_search_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        rewritten_query, item_names = self._validate_state(state)
        # 同步包装异步 MCP 调用
        execute_tool_result = asyncio.run(self.mcp_web_search(rewritten_query))
        # 工具返回的 text 为 JSON 字符串，反序列化后取 pages
        result_text = execute_tool_result.content[0].text
        json_object = json.loads(result_text)
        pages = json_object["pages"]

        web_search_docs = []
        for page in pages:
            snippet = page["snippet"]
            title = page["title"]
            url = page["url"]
            web_search_docs.append({
                "snippet": snippet,
                "title": title,
                "url": url
            })

        return {"web_search_docs": web_search_docs}

    async def mcp_web_search(self, rewritten_query: str):
        """连接百炼 MCP 服务并调用 bailian_web_search 工具（count=3）。"""
        url = self.config.mcp_dashscope_base_url.strip()
        api_key = self.config.openai_api_key.strip()
        if not url or not api_key:
            raise ValueError("联网搜索需要配置 MCP_DASHSCOPE_BASE_URL 和 OPEN_API_KEY")
        authorization = api_key if api_key.lower().startswith("bearer ") else f"Bearer {api_key}"
        # /sse 路径走旧版 HTTP+SSE 协议，其余走 Streamable HTTP
        use_sse = urlsplit(url).path.rstrip("/").endswith("/sse")
        server_class = MCPServerSse if use_sse else MCPServerStreamableHttp
        params = {
            "url": url,
            "headers": {"Authorization": authorization},
            "timeout": 300,
        }
        if not use_sse:
            params["terminate_on_close"] = True
        try:
            server = server_class(
                name="search_mcp",
                params=params,
                max_retry_attempts=2,
                cache_tools_list=True,
            )
            if not use_sse:
                async with MCPClient(
                        server.create_streams(),
                        mode="legacy",
                        cache=None,
                        read_timeout_seconds=300,
                ) as client:
                    return await client.session.call_tool(
                        name="bailian_web_search",
                        arguments={"query": rewritten_query, "count": 3},
                    )
            async with server as client:
                execute_tool_result = await client.call_tool(
                    tool_name="bailian_web_search",
                    arguments={"query": rewritten_query, "count": 3}
                )

                return execute_tool_result
        except BaseExceptionGroup as exc:
            if use_sse and any(
                isinstance(leaf, RuntimeError)
                and "Child exited without calling task_status.started()" in str(leaf)
                for leaf in _exception_leaves(exc)
            ):
                raise RuntimeError(
                    "百炼 MCP SSE 握手未完成：服务端关闭了连接，但没有发送 endpoint 事件。"
                    "请检查 MCP 服务地址、鉴权配置及服务端状态。"
                ) from None
            raise
        except RuntimeError as exc:
            if use_sse and "Child exited without calling task_status.started()" in str(exc):
                raise RuntimeError(
                    "百炼 MCP SSE 握手未完成：服务端关闭了连接，但没有发送 endpoint 事件。"
                    "请检查 MCP 服务地址、鉴权配置及服务端状态。"
                ) from None
            raise

    def _validate_state(self, state: QueryGraphState):
        rewritten_query = state.get("rewritten_query")
        item_names = state.get("item_names")
        if not rewritten_query or not isinstance(rewritten_query, str):
            self.logger.error(f"rewritten_query不能为空以及类型必须是str")
            raise StateFieldError(node_name=self.name, field_name="rewritten_query", expected_type=str)

        if not item_names or not isinstance(item_names, list):
            self.logger.error("item_names不能为空以及类型必须是list")
            raise StateFieldError(node_name=self.name, field_name="item_names", expected_type=list)

        return rewritten_query, item_names
