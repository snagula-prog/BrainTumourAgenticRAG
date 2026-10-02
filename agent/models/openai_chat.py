
from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from agent.models.base import ChatMessage, ModelResponse, ToolCall


class OpenAIChatModel:
    """OpenAI Chat Completions adapter for the ChatModel contract."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        client: Any | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be positive")

        self.model = model.strip()

        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "Install the OpenAI SDK with "
                    "'uv pip install openai'"
                ) from exc

            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=float(timeout),
            )

        self._client = client

    @staticmethod
    def _serialize_message(message: ChatMessage) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": message.role}

        if message.content is not None:
            payload["content"] = message.content

        if message.role == "assistant" and message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(
                            call.arguments, ensure_ascii=False
                        ),
                    },
                }
                for call in message.tool_calls
            ]

        if message.role == "tool":
            if not message.tool_call_id:
                raise ValueError("Tool message requires tool_call_id")
            payload["tool_call_id"] = message.tool_call_id
            if message.name:
                payload["name"] = message.name

        return payload

    @staticmethod
    def _serialize_tool(tool: dict[str, Any]) -> dict[str, Any]:
        name = tool.get("name")
        description = tool.get("description")
        schema = tool.get("input_schema")

        if not isinstance(name, str) or not name.strip():
            raise ValueError("Tool requires a non-empty name")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"Tool {name} requires a description")
        if not isinstance(schema, dict):
            raise ValueError(f"Tool {name} requires an input schema")

        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": schema,
            },
        }

    @staticmethod
    def _parse_tool_calls(raw_calls: Any) -> tuple[ToolCall, ...]:
        parsed: list[ToolCall] = []

        for raw in raw_calls or []:
            try:
                arguments = json.loads(raw.function.arguments)
            except (AttributeError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    "Model returned invalid tool-call arguments"
                ) from exc

            if not isinstance(arguments, dict):
                raise ValueError(
                    "Tool-call arguments must decode to an object"
                )

            if not isinstance(raw.id, str) or not raw.id:
                raise ValueError("Model returned a tool call without an ID")

            if not isinstance(raw.function.name, str):
                raise ValueError("Model returned a tool call without a name")

            parsed.append(
                ToolCall(
                    id=raw.id,
                    name=raw.function.name,
                    arguments=arguments,
                )
            )

        return tuple(parsed)

    def generate(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[dict[str, Any]],
    ) -> ModelResponse:
        payload_messages = [
            self._serialize_message(message) for message in messages
        ]
        payload_tools = [
            self._serialize_tool(tool) for tool in tools
        ]

        request: dict[str, Any] = {
            "model": self.model,
            "messages": payload_messages,
        }

        if payload_tools:
            request["tools"] = payload_tools
            request["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**request)

        if not response.choices:
            raise RuntimeError("Model returned no choices")

        message = response.choices[0].message
        content = message.content

        if content is not None and not isinstance(content, str):
            raise RuntimeError("Model returned unsupported content")

        return ModelResponse(
            content=content,
            tool_calls=self._parse_tool_calls(message.tool_calls),
        )
