from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agent.models.base import (
    ChatMessage,
    ChatModel,
    ModelResponse,
    ToolCall,
)
from agent.tool_registry import ToolRegistry


DEFAULT_SYSTEM_PROMPT = """
You are a research-paper assistant.

Answer questions using evidence from the available research tools.
Use tools whenever external paper evidence is needed.
Never invent paper findings, statistics, or citations.

When citing evidence, preserve the paper IDs, chunk IDs, and page
information supplied by the tools.

Distinguish reported results from your own analysis.
If the available evidence is insufficient, say so explicitly.
""".strip()


class AgentError(RuntimeError):
    pass


class AgentProtocolError(AgentError):
    pass


class AgentStepLimitError(AgentError):
    pass


@dataclass(frozen=True)
class ToolExecution:
    call_id: str
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any] | None
    error: str | None = None


@dataclass(frozen=True)
class AgentResult:
    answer: str
    tool_trace: tuple[ToolExecution, ...]
    messages: tuple[ChatMessage, ...]
    model_turns: int


class AgentOrchestrator:
    """Run a bounded, provider-independent tool-calling loop."""

    def __init__(
        self,
        *,
        model: ChatModel,
        tools: ToolRegistry,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 5,
    ) -> None:
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")

        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system_prompt cannot be empty")

        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt.strip()
        self.max_steps = max_steps

    @staticmethod
    def _validate_tool_call(call: ToolCall) -> None:
        if not isinstance(call, ToolCall):
            raise AgentProtocolError("Model returned an invalid tool call")

        if not isinstance(call.id, str) or not call.id.strip():
            raise AgentProtocolError("Tool call is missing its ID")

        if not isinstance(call.name, str) or not call.name.strip():
            raise AgentProtocolError("Tool call is missing its name")

        if not isinstance(call.arguments, dict):
            raise AgentProtocolError(
                f"Arguments for {call.name} must be an object"
            )

    def _execute_tool(
        self, call: ToolCall
    ) -> tuple[dict[str, Any] | None, str | None]:
        try:
            result = self.tools.execute(call.name, call.arguments)

            if not isinstance(result, dict):
                raise TypeError("Tool must return a dictionary")

            # Ensure the result can be passed back to the model as JSON.
            json.dumps(result, ensure_ascii=False)

            return result, None

        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    def run(self, query: str) -> AgentResult:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(role="user", content=query.strip()),
        ]
        trace: list[ToolExecution] = []

        for step in range(1, self.max_steps + 1):
            response = self.model.generate(
                messages=tuple(messages),
                tools=self.tools.definitions(),
            )

            if not isinstance(response, ModelResponse):
                raise AgentProtocolError(
                    "Model must return a ModelResponse"
                )

            if response.tool_calls:
                if not isinstance(response.tool_calls, tuple):
                    raise AgentProtocolError(
                        "tool_calls must be a tuple"
                    )

                messages.append(
                    ChatMessage(
                        role="assistant",
                        content=response.content,
                        tool_calls=response.tool_calls,
                    )
                )

                for call in response.tool_calls:
                    self._validate_tool_call(call)

                    result, error = self._execute_tool(call)

                    trace.append(
                        ToolExecution(
                            call_id=call.id,
                            name=call.name,
                            arguments=call.arguments,
                            result=result,
                            error=error,
                        )
                    )

                    tool_content = (
                        json.dumps(result, ensure_ascii=False)
                        if error is None
                        else json.dumps(
                            {"error": error},
                            ensure_ascii=False,
                        )
                    )

                    messages.append(
                        ChatMessage(
                            role="tool",
                            name=call.name,
                            tool_call_id=call.id,
                            content=tool_content,
                        )
                    )

                continue

            if isinstance(response.content, str) and response.content.strip():
                messages.append(
                    ChatMessage(
                        role="assistant",
                        content=response.content,
                    )
                )

                return AgentResult(
                    answer=response.content.strip(),
                    tool_trace=tuple(trace),
                    messages=tuple(messages),
                    model_turns=step,
                )

            raise AgentProtocolError(
                "Model returned neither an answer nor tool calls"
            )

        raise AgentStepLimitError(
            f"Agent reached the maximum of {self.max_steps} "
            "model turns without producing a final answer"
        )