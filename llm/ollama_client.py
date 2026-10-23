"""Ollama implementation of the project's provider-neutral LLM interface.

Expected interface: ``llm.base.ChatModel`` and ``llm.base.LLMResponse``.
Uses only the Python standard library for HTTP calls.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, TypeVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ValidationError

from .base import LLMResponse

T = TypeVar("T", bound=BaseModel)


class OllamaError(RuntimeError):
    """Base exception for local Ollama request failures."""


class OllamaAPIError(OllamaError):
    """Ollama rejected a request or returned an unusable response."""


class StructuredOutputError(OllamaError):
    """The model response could not be validated against the requested schema."""

    def __init__(self, message: str, *, response: LLMResponse | None = None) -> None:
        super().__init__(message)
        self.response = response


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class OllamaConfig:
    """Runtime settings; defaults match the V3 benchmark configuration."""

    base_url: str = "http://localhost:11434"
    model: str = "qwen3:4b-instruct"
    timeout_seconds: float = 120.0
    num_ctx: int = 4096
    num_predict: int = 512
    temperature: float = 0.0
    seed: int = 42
    keep_alive: str = "5m"
    disable_thinking: bool = True

    @classmethod
    def from_env(cls) -> "OllamaConfig":
        """Load settings from environment variables without requiring a dotenv dependency."""
        return cls(
            base_url=os.getenv("OLLAMA_BASE_URL", cls.base_url).rstrip("/"),
            model=os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct"),
            timeout_seconds=float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120")),
            num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "4096")),
            num_predict=int(os.getenv("OLLAMA_NUM_PREDICT", "512")),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0")),
            seed=int(os.getenv("OLLAMA_SEED", "42")),
            keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "5m"),
            disable_thinking=_env_bool("OLLAMA_DISABLE_THINKING", True),
        )

    def __post_init__(self) -> None:
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        if not self.model.strip():
            raise ValueError("model cannot be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.num_ctx < 1 or self.num_predict < 1:
            raise ValueError("num_ctx and num_predict must be positive integers")


class OllamaChatModel:
    """Synchronous Ollama chat client implementing ``llm.base.ChatModel``."""

    def __init__(self, config: OllamaConfig | None = None) -> None:
        self.config = config or OllamaConfig.from_env()

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Generate a normal text response."""
        return self._chat(prompt, system_prompt=system_prompt, schema=None)

    def complete_structured(
        self,
        prompt: str,
        *,
        schema: type[T],
        system_prompt: str | None = None,
    ) -> tuple[T, LLMResponse]:
        """Generate output constrained by a Pydantic JSON schema and validate it."""
        response = self._chat(prompt, system_prompt=system_prompt, schema=schema)
        if response.termination_reason in {"length", "max_tokens"}:
            raise StructuredOutputError(
                f"Model {response.model!r} stopped before completing structured output "
                f"({response.termination_reason}).",
                response=response,
            )
        try:
            if hasattr(schema, "model_validate_json"):  # Pydantic v2
                parsed = schema.model_validate_json(response.text)
            else:  # Pydantic v1 compatibility
                parsed = schema.parse_raw(response.text)
        except (ValidationError, ValueError, TypeError) as exc:
            raise StructuredOutputError(
                f"Model {response.model!r} returned output that does not match "
                f"{schema.__name__}: {exc}. Output prefix: {response.text[:500]!r}",
                response=response,
            ) from exc
        return parsed, response

    def _chat(
        self,
        prompt: str,
        *,
        system_prompt: str | None,
        schema: type[BaseModel] | None,
    ) -> LLMResponse:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise TypeError("system_prompt must be a string or None")

        messages: list[dict[str, str]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": prompt.strip()})

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.config.keep_alive,
            "options": {
                "temperature": self.config.temperature,
                "seed": self.config.seed,
                "num_ctx": self.config.num_ctx,
                "num_predict": self.config.num_predict,
            },
        }
        if self.config.disable_thinking:
            payload["think"] = False
        if schema is not None:
            if hasattr(schema, "model_json_schema"):  # Pydantic v2
                payload["format"] = schema.model_json_schema()
            else:  # Pydantic v1
                payload["format"] = schema.schema()

        started = time.perf_counter()
        data = self._post_chat_with_compatibility_fallback(payload)
        elapsed = time.perf_counter() - started

        message = data.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise OllamaAPIError("Ollama response is missing message.content")

        content = message["content"]
        if not content.strip():
            raise OllamaAPIError("Ollama returned an empty message")

        return LLMResponse(
            text=content,
            model=str(data.get("model") or self.config.model),
            elapsed_seconds=elapsed,
            prompt_tokens=_optional_int(data.get("prompt_eval_count")),
            output_tokens=_optional_int(data.get("eval_count")),
            total_duration_ns=_optional_int(data.get("total_duration")),
            termination_reason=(str(data["done_reason"]) if data.get("done_reason") else None),
        )

    def _post_chat_with_compatibility_fallback(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Retry without Ollama's optional ``think`` field if an older server rejects it."""
        try:
            return self._post_json("/api/chat", payload)
        except OllamaAPIError as exc:
            if "think" not in payload or not _looks_like_unsupported_think(str(exc)):
                raise
            fallback_payload = dict(payload)
            fallback_payload.pop("think", None)
            return self._post_json("/api/chat", fallback_payload)

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.config.base_url}{path}"
        request = Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaAPIError(f"Ollama HTTP {exc.code}: {detail[:1200]}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise OllamaAPIError(
                f"Could not reach Ollama at {url}. Confirm Ollama is running and the "
                f"model is installed. Details: {exc}"
            ) from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OllamaAPIError(f"Ollama returned invalid JSON: {raw[:500]!r}") from exc
        if not isinstance(data, dict):
            raise OllamaAPIError("Ollama response must be a JSON object")
        if data.get("error"):
            raise OllamaAPIError(str(data["error"]))
        return data


def _optional_int(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _looks_like_unsupported_think(message: str) -> bool:
    lowered = message.lower()
    return "think" in lowered and any(
        marker in lowered
        for marker in ("unknown field", "unsupported", "invalid parameter", "bad request")
    )
