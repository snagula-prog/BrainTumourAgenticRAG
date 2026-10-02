
from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import TypeVar

from pydantic import BaseModel

from llm.base import LLMResponse

T = TypeVar("T", bound=BaseModel)


class FakeChatModel:
    """Deterministic model for tests; never accesses Ollama."""

    def __init__(
        self,
        responses: Iterable[str] = (),
    ) -> None:
        self._responses = deque(responses)
        self.calls: list[dict[str, str]] = []

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        self.calls.append({
            "method": "complete",
            "prompt": prompt,
            "system_prompt": system_prompt or "",
        })

        if not self._responses:
            raise RuntimeError("FakeChatModel has no queued response")

        return LLMResponse(
            text=self._responses.popleft(),
            model="fake",
            elapsed_seconds=0.0,
        )

    def complete_structured(
        self,
        prompt: str,
        *,
        schema: type[T],
        system_prompt: str | None = None,
    ) -> tuple[T, LLMResponse]:
        self.calls.append({
            "method": "complete_structured",
            "prompt": prompt,
            "system_prompt": system_prompt or "",
        })

        if not self._responses:
            raise RuntimeError("FakeChatModel has no queued response")

        response = LLMResponse(
            text=self._responses.popleft(),
            model="fake",
            elapsed_seconds=0.0,
        )
        return schema.model_validate_json(response.text), response
