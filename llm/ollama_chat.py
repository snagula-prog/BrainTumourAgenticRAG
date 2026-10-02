
from __future__ import annotations

import json
import math
import time
from typing import TypeVar

from pydantic import BaseModel

from llm.base import LLMResponse

T = TypeVar("T", bound=BaseModel)


class OllamaChatModel:
    def __init__(
        self,
        *,
        model: str = "qwen3:4b-instruct",
        host: str = "http://localhost:11434",
        timeout: float = 180.0,
        num_ctx: int = 4096,
        max_tokens: int = 512,
        structured_max_tokens: int = 2048,
        temperature: float = 0.0,
        client=None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be non-empty")
        if not isinstance(host, str) or not host.strip():
            raise ValueError("host must be non-empty")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ValueError("timeout must be numeric")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        if type(num_ctx) is not int or num_ctx < 1:
            raise ValueError("num_ctx must be positive")
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or temperature < 0
        ):
            raise ValueError("temperature must be non-negative and finite")

        if client is None:
            try:
                from ollama import Client
            except ImportError as exc:
                raise RuntimeError(
                    "Install the Ollama SDK with: "
                    "uv pip install ollama"
                ) from exc

            client = Client(host=host, timeout=float(timeout))

        self.model = model.strip()
        self.num_ctx = num_ctx
        self.max_tokens = max_tokens
        self.temperature = float(temperature)
        self._client = client
        self.structured_max_tokens = structured_max_tokens

    def _call(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        schema: type[BaseModel] | None = None,
    ) -> LLMResponse:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be non-empty")

        messages = []
        if system_prompt is not None:
            if not isinstance(system_prompt, str) or not system_prompt.strip():
                raise ValueError("system_prompt must be non-empty")
            messages.append({
                "role": "system",
                "content": system_prompt.strip(),
            })

        user_prompt = prompt.strip()
        kwargs = {}

        if schema is not None:
            kwargs["format"] = schema.model_json_schema()
            user_prompt += (
                "\n\nReturn a JSON object matching the required schema. "
                "Do not include Markdown."
            )

        messages.append({"role": "user", "content": user_prompt})

        started = time.perf_counter()
        response = self._client.chat(
            model=self.model,
            messages=messages,
            stream=False,
            options={
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": (
                    self.structured_max_tokens
                    if schema is not None
                    else self.max_tokens
                ),
            },
            **kwargs,
        )
        elapsed = time.perf_counter() - started

        message = getattr(response, "message", None)
        content = getattr(message, "content", None)

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Ollama returned empty content")

        return LLMResponse(
            text=content,
            model=getattr(response, "model", None) or self.model,
            elapsed_seconds=elapsed,
            prompt_tokens=getattr(response, "prompt_eval_count", None),
            output_tokens=getattr(response, "eval_count", None),
            total_duration_ns=getattr(response, "total_duration", None),
        )

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        return self._call(prompt, system_prompt=system_prompt)

    def complete_structured(
        self,
        prompt: str,
        *,
        schema: type[T],
        system_prompt: str | None = None,
    ) -> tuple[T, LLMResponse]:
        if not isinstance(schema, type) or not issubclass(schema, BaseModel):
            raise TypeError("schema must be a Pydantic model class")

        response = self._call(
            prompt,
            system_prompt=system_prompt,
            schema=schema,
        )
        parsed = schema.model_validate_json(response.text)
        return parsed, response
