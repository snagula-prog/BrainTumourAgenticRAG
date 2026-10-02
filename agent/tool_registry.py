from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from typing import Any, Protocol


class AgentTool(Protocol):
    name: str
    description: str
    input_schema: dict[str, Any]

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        ...


class ToolRegistry:
    """Registers tools and exposes their definitions to the agent."""

    def __init__(self, tools: Iterable[AgentTool] = ()) -> None:
        self._tools: dict[str, AgentTool] = {}

        for tool in tools:
            self.register(tool)

    def register(self, tool: AgentTool) -> None:
        name = getattr(tool, "name", None)
        description = getattr(tool, "description", None)
        schema = getattr(tool, "input_schema", None)

        if not isinstance(name, str) or not name.strip():
            raise ValueError("Tool must have a non-empty name")

        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")

        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"Tool {name} needs a description")

        if not isinstance(schema, dict):
            raise ValueError(f"Tool {name} needs a JSON schema")

        if not callable(getattr(tool, "execute", None)):
            raise ValueError(f"Tool {name} has no execute method")

        self._tools[name] = tool

    def get(self, name: str) -> AgentTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def definitions(self) -> list[dict[str, Any]]:
        """Return provider-neutral tool definitions."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": deepcopy(tool.input_schema),
            }
            for tool in self._tools.values()
        ]

    def execute(
        self, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        return self.get(name).execute(arguments)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)