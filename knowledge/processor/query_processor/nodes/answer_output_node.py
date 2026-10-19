from typing import Dict, Any, List, Tuple

from langchain_openai import ChatOpenAI
from typer.cli import state

from knowledge.processor.query_processor.base import BaseNode, T
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.prompts.query_prompt import ANSWER_PROMPT
from knowledge.utils.clients.ai_clients import AIClients
from knowledge.utils.mongo_history_util import save_chat_message, get_recent_messages
from knowledge.utils.sse_util import push_sse_event, SSEEvent
from knowledge.utils.task_util import set_task_result


class AnswerOutputNode(BaseNode):
    """答案输出节点：已有答案则直接推送，否则组装上下文调用大模型生成答案，最后落库历史会话。"""
    name = "answer_output_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        user_query = state.get("rewritten_query")
        task_id = state.get("task_id")
        answer = state.get("answer")
        is_stream = state.get("is_stream")

        if answer:
            # 上游已给出答案（商品名待确认/无商品），直接输出
            self._push_exist_answer(answer, is_stream, task_id)
        else:
            # 组装提示词并调用大模型生成答案
            prompt = self._build_answer_prompt(state, self.config.max_context_chars)
            llm_answer = self._generate_answer(prompt, state)
            state["answer"] = llm_answer
            # 流式结束后通过 FINAL 事件通知前端关闭 SSE 通道
            if is_stream:
                push_sse_event(task_id=task_id, event=SSEEvent.FINAL, data={})

        # 保存本轮问答历史到 MongoDB
        self._save_history(state)

        return state

    def _push_exist_answer(self, answer: str, is_stream: bool, task_id: str):
        """流式经 SSE FINAL 推送，非流式写入任务结果。"""
        if is_stream:
            push_sse_event(task_id=task_id, event=SSEEvent.FINAL, data={"answer": answer})
        else:
            set_task_result(task_id=task_id, key="answer", value=answer)

    def _build_answer_prompt(self, state: QueryGraphState, max_context_chars: int) -> str:
        """组装答案提示词：问题 + 商品名 + 精排上下文 + 历史对话，总长度受 max_context_chars 约束。"""
        user_query = state.get("rewritten_query")
        item_names = state.get("item_names") or []
        reranked_docs = state.get("reranked_docs") or []
        # 格式化并截断检索上下文，返回 (文本, 已用字符数)
        formatted_context, usage_chars = self._format_retrieval_context(reranked_docs, max_context_chars)
        history = state.get("history") or []
        # 历史对话占用剩余字符预算
        formatted_history = self._format_history(history, max_context_chars - usage_chars)

        return ANSWER_PROMPT.format(
            question=user_query,
            context=formatted_context,
            history=formatted_history,
            item_names=",".join(item_names)
        )

    def _format_history(self, history: List[Dict[str, Any]], char_budget: int) -> Tuple[str, int]:
        """按字符预算格式化历史对话，返回 (文本, 剩余字符数)。"""
        formatted_lines = []
        used_chars = 0
        role_map = {"user": "用户", "assistant": "助手"}
        for msg in history:
            role = msg.get('role', '')
            text = msg.get('text', '')
            if not text or role not in role_map:
                continue

            formatted_line = f"{role_map[role]}: {text}"
            seperator_usage = 1 if formatted_lines else 0
            total_usage = seperator_usage + len(formatted_line)

            if used_chars + total_usage > char_budget:
                break

            formatted_lines.append(formatted_line)
            used_chars += total_usage

        return "\n".join(formatted_lines), char_budget - used_chars

    def _format_retrieval_context(self, reranked_docs: List[Dict[str, Any]], max_context_chars: int) -> Tuple[str, int]:
        """将精排结果格式化为带元信息的上下文文本，超预算即截断。

        示例：
        [1] [source=local] [chunk_id=chunk_001] [title=操作指导] [score=5.060000]
        测量直流电压时，将旋钮转到DCV档位...
        """
        final_context = []
        usage_chars = 0

        for index, reranked_doc in enumerate(reranked_docs, 1):
            metadata_content = [f"[{index}]"]
            for field_name, template in [("chunk_id", "[chunk_id={}]"),
                                         ("title", "[title={}]"),
                                         ("source", "[source={}]"),
                                         ("url", "[url={}]")]:
                field_value = reranked_doc.get(field_name)
                if field_value:
                    metadata_content.append(template.format(field_value))

            score = float(reranked_doc.get("score"))
            metadata_content.append(f"[score={score:.6f}]")
            content = reranked_doc.get("content")
            format_chunk = " ".join(metadata_content) + "\n" + content

            spec_len = 2 if final_context else 0
            chunk_length = spec_len + len(format_chunk)
            if usage_chars + chunk_length > max_context_chars:
                break

            final_context.append(format_chunk)
            usage_chars += chunk_length

        return "\n\n".join(final_context), usage_chars

    def _generate_answer(self, prompt: str, state: QueryGraphState):
        """获取大模型客户端并按流式/非流式生成答案；客户端不可用时返回兜底文案。"""
        try:
            llm_client = AIClients.get_llm_client(response_format=False)
        except Exception as e:
            return "LLM暂无任何内容输出"

        llm_result = ""
        if state.get("is_stream"):
            # 流式：边生成边推送 SSE DELTA 事件
            llm_result = self._llm_stream(prompt, llm_client, state)
        else:
            # 非流式：拿到完整答案后写入任务结果
            llm_result = self._llm_invoke(prompt, llm_client)
            set_task_result(task_id=state.get("task_id"), key="answer", value=llm_result)
        return llm_result

    def _llm_invoke(self, prompt: str, llm_client: ChatOpenAI):
        """非流式调用大模型，失败或空输出返回兜底文案。"""
        try:
            llm_res = llm_client.invoke(prompt)
        except Exception as e:
            return "LLM暂无任何内容输出"
        llm_content = llm_res.content
        if not llm_content:
            return "LLM暂无任何内容输出"
        return llm_content

    def _llm_stream(self, prompt: str, llm_client: ChatOpenAI, state: QueryGraphState):
        """流式调用大模型，逐块推送 SSE DELTA 并拼接最终答案。"""
        final_answer = ""
        for chunk in llm_client.stream(prompt):
            content = getattr(chunk, "content", "")
            if content:
                push_sse_event(task_id=state.get("task_id"), event=SSEEvent.DELTA, data={"delta": content})
            final_answer += content
        return final_answer

    def _save_history(self, state: QueryGraphState):
        """保存用户提问与助手回答两条消息到 MongoDB。"""
        save_chat_message(
            session_id=state.get("session_id"),
            role="user",
            text=state.get("original_query", ""),
            rewritten_query=state.get("rewritten_query") or "",
            item_names=state.get("item_names") or []
        )

        save_chat_message(
            session_id=state.get("session_id"),
            role="assistant",
            text=state.get("answer") or "",
            rewritten_query=state.get("rewritten_query") or "",
            item_names=state.get("item_names", [])
        )
