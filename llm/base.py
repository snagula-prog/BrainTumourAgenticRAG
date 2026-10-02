
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    elapsed_seconds: float
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    total_duration_ns: int | None = None


class ChatModel(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        ...

    def complete_structured(
        self,
        prompt: str,
        *,
        schema: type[T],
        system_prompt: str | None = None,
    ) -> tuple[T, LLMResponse]:
        ...
