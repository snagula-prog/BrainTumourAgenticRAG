from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from pydantic import BaseModel

from llm.base import LLMResponse
from llm.ollama_client import (
    OllamaAPIError,
    OllamaChatModel,
    OllamaConfig,
    StructuredOutputError,
)


class AnswerSchema(BaseModel):
    answer: str
    evidence_ids: list[str]


class FakeHTTPResponse:
    def __init__(self, value: dict):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return json.dumps(self.value).encode("utf-8")


class OllamaClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = OllamaConfig(
            model="qwen3:4b-instruct",
            base_url="http://localhost:11434",
            disable_thinking=True,
        )
        self.client = OllamaChatModel(self.config)

    @patch("llm.ollama_client.urlopen")
    def test_complete_sends_selected_model_and_returns_usage(self, mock_urlopen):
        mock_urlopen.return_value = FakeHTTPResponse(
            {
                "model": "qwen3:4b-instruct",
                "message": {"role": "assistant", "content": "Grounded answer."},
                "prompt_eval_count": 17,
                "eval_count": 6,
                "total_duration": 1_500_000_000,
            }
        )
        result = self.client.complete("Question?", system_prompt="Use evidence.")
        self.assertIsInstance(result, LLMResponse)
        self.assertEqual(result.text, "Grounded answer.")
        self.assertEqual(result.model, "qwen3:4b-instruct")
        self.assertEqual(result.prompt_tokens, 17)
        self.assertEqual(result.output_tokens, 6)
        sent_payload = json.loads(mock_urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(sent_payload["model"], "qwen3:4b-instruct")
        self.assertFalse(sent_payload["think"])

    @patch("llm.ollama_client.urlopen")
    def test_structured_output_is_parsed_and_validated(self, mock_urlopen):
        mock_urlopen.return_value = FakeHTTPResponse(
            {
                "model": "qwen3:4b-instruct",
                "message": {"role": "assistant", "content": '{"answer":"Yes","evidence_ids":["E1"]}'},
            }
        )
        parsed, response = self.client.complete_structured("Return JSON", schema=AnswerSchema)
        self.assertEqual(parsed.answer, "Yes")
        self.assertEqual(parsed.evidence_ids, ["E1"])
        self.assertEqual(response.model, "qwen3:4b-instruct")
        sent_payload = json.loads(mock_urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertIn("format", sent_payload)

    @patch("llm.ollama_client.urlopen")
    def test_invalid_structured_output_raises_clear_error(self, mock_urlopen):
        mock_urlopen.return_value = FakeHTTPResponse(
            {"model": "qwen3:4b-instruct", "message": {"content": '{"answer":42}'}}
        )
        with self.assertRaises(StructuredOutputError):
            self.client.complete_structured("Return JSON", schema=AnswerSchema)

    def test_empty_prompt_is_rejected_before_network_call(self):
        with self.assertRaises(ValueError):
            self.client.complete("  ")

    def test_environment_defaults_select_qwen3_instruct(self):
        with patch.dict("os.environ", {}, clear=True):
            config = OllamaConfig.from_env()
        self.assertEqual(config.model, "qwen3:4b-instruct")
        self.assertEqual(config.num_ctx, 4096)
        self.assertEqual(config.temperature, 0.0)


if __name__ == "__main__":
    unittest.main()
