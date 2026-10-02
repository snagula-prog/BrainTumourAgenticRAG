
from types import SimpleNamespace as NS

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from llm import FakeChatModel, OllamaChatModel


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str
    supported: bool


class FakeClient:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return NS(
            model=kwargs["model"],
            message=NS(content=self.content),
            prompt_eval_count=10,
            eval_count=20,
            total_duration=1_000_000_000,
        )


def test_fake_completion():
    model = FakeChatModel(["A concise response."])
    result = model.complete("Answer this")

    assert result.text == "A concise response."
    assert result.model == "fake"
    assert len(model.calls) == 1


def test_fake_structured_completion():
    model = FakeChatModel([
        '{"answer": "Supported", "supported": true}'
    ])
    value, result = model.complete_structured(
        "Answer using evidence",
        schema=Answer,
    )

    assert value.answer == "Supported"
    assert value.supported is True
    assert result.model == "fake"


def test_ollama_completion_controls():
    client = FakeClient("Test answer")
    model = OllamaChatModel(
        model="qwen3:4b",
        client=client,
        num_ctx=4096,
        max_tokens=256,
        temperature=0.0,
    )

    result = model.complete(
        "Hello",
        system_prompt="Be concise.",
    )

    call = client.calls[0]
    assert call["options"]["num_ctx"] == 4096
    assert call["options"]["num_predict"] == 256
    assert call["options"]["temperature"] == 0.0
    assert result.prompt_tokens == 10
    assert result.output_tokens == 20


def test_ollama_structured_output():
    client = FakeClient(
        '{"answer": "Evidence supports this", "supported": true}'
    )
    model = OllamaChatModel(client=client)

    value, _ = model.complete_structured(
        "Use the supplied evidence.",
        schema=Answer,
    )

    assert value.supported is True
    assert client.calls[0]["format"] == Answer.model_json_schema()


def test_invalid_structured_output_is_rejected():
    client = FakeClient('{"answer": 42}')
    model = OllamaChatModel(client=client)

    with pytest.raises(ValidationError):
        model.complete_structured("Test", schema=Answer)


def test_fake_model_exhaustion():
    model = FakeChatModel()

    with pytest.raises(RuntimeError, match="no queued response"):
        model.complete("Test")
