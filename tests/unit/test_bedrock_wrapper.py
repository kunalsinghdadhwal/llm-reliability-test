"""Unit tests for src.wrappers.bedrock -- all AWS calls are mocked."""

import importlib
import json
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CONVERSE_RESPONSE = {
    "output": {"message": {"content": [{"text": "mocked response"}]}},
    "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
    "stopReason": "end_turn",
}


def _make_invoke_model_response():
    body_mock = MagicMock()
    body_mock.read.return_value = json.dumps(
        {"embedding": [0.1, 0.2, 0.3], "inputTextTokenCount": 3}
    )
    return {"body": body_mock}


# ---------------------------------------------------------------------------
# call_llm tests
# ---------------------------------------------------------------------------


class TestCallLlm:
    def test_returns_text(self):
        """call_llm returns text from mocked converse response."""
        with patch("src.wrappers.bedrock._client") as mock_client:
            mock_client.converse.return_value = CONVERSE_RESPONSE
            from src.wrappers.bedrock import call_llm

            result = call_llm("hello")
            assert result == "mocked response"

    def test_passes_system_prompt(self):
        """call_llm passes system prompt when provided."""
        with patch("src.wrappers.bedrock._client") as mock_client:
            mock_client.converse.return_value = CONVERSE_RESPONSE
            from src.wrappers.bedrock import call_llm

            call_llm("hello", system="Be helpful")
            _, kwargs = mock_client.converse.call_args
            assert kwargs["system"] == [{"text": "Be helpful"}]

    def test_omits_system_when_empty(self):
        """call_llm omits system when empty string."""
        with patch("src.wrappers.bedrock._client") as mock_client:
            mock_client.converse.return_value = CONVERSE_RESPONSE
            from src.wrappers.bedrock import call_llm

            call_llm("hello", system="")
            _, kwargs = mock_client.converse.call_args
            assert "system" not in kwargs

    def test_inference_config(self):
        """call_llm uses temperature 0.0 and maxTokens 4096."""
        with patch("src.wrappers.bedrock._client") as mock_client:
            mock_client.converse.return_value = CONVERSE_RESPONSE
            from src.wrappers.bedrock import call_llm

            call_llm("hello")
            _, kwargs = mock_client.converse.call_args
            assert kwargs["inferenceConfig"] == {
                "maxTokens": 4096,
                "temperature": 0.0,
            }


# ---------------------------------------------------------------------------
# embed tests
# ---------------------------------------------------------------------------


class TestEmbed:
    def test_returns_list_of_floats(self):
        """embed returns list of floats from mocked invoke_model response."""
        with patch("src.wrappers.bedrock._client") as mock_client:
            mock_client.invoke_model.return_value = _make_invoke_model_response()
            from src.wrappers.bedrock import embed

            result = embed("some text")
            assert result == [0.1, 0.2, 0.3]
            assert all(isinstance(v, float) for v in result)

    def test_sends_correct_body(self):
        """embed sends correct body with inputText."""
        with patch("src.wrappers.bedrock._client") as mock_client:
            mock_client.invoke_model.return_value = _make_invoke_model_response()
            from src.wrappers.bedrock import embed

            embed("some text")
            _, kwargs = mock_client.invoke_model.call_args
            assert json.loads(kwargs["body"]) == {"inputText": "some text"}
            assert kwargs["accept"] == "application/json"
            assert kwargs["contentType"] == "application/json"


# ---------------------------------------------------------------------------
# Module-level client test
# ---------------------------------------------------------------------------


class TestModuleLevelClient:
    def test_client_created_once(self):
        """Module-level client created once -- patch boto3.client, reload module."""
        with patch("boto3.client") as mock_boto_client:
            mock_boto_client.return_value = MagicMock()
            import src.wrappers.bedrock as bedrock_mod

            importlib.reload(bedrock_mod)
            mock_boto_client.assert_called_once_with("bedrock-runtime", region_name="us-east-1")
