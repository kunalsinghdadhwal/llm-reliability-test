"""Tests for run_model agent."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.agents.run_model import call_target_llm, run_model


def _make_state(prompts, provider="bedrock", model_id="test-model"):
    return {
        "prompts": prompts,
        "config": {
            "model": {"provider": provider, "model_id": model_id},
        },
    }


class TestCallTargetLlm:
    @patch("src.agents.run_model.call_llm", return_value="bedrock response")
    def test_bedrock_provider(self, mock_llm):
        result = call_target_llm("hello", {"provider": "bedrock", "model_id": "x"})
        assert result == "bedrock response"
        mock_llm.assert_called_once_with("hello")

    def test_ollama_provider(self):
        mock_client_instance = MagicMock()
        mock_client_instance.chat.return_value = SimpleNamespace(
            message=SimpleNamespace(content="ollama response")
        )
        mock_ollama_module = MagicMock()
        mock_ollama_module.Client = MagicMock(return_value=mock_client_instance)

        with patch.dict(sys.modules, {"ollama": mock_ollama_module}):
            # Need to reimport to pick up the mocked module
            import importlib

            import src.agents.run_model as mod

            importlib.reload(mod)
            result = mod.call_target_llm("hello", {"provider": "ollama", "model_id": "llama3"})

        assert result == "ollama response"
        mock_client_instance.chat.assert_called_once_with(
            model="llama3",
            messages=[{"role": "user", "content": "hello"}],
        )

    def test_unknown_provider_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Unknown provider: foobar"):
            call_target_llm("hello", {"provider": "foobar", "model_id": "x"})


class TestRunModel:
    @patch("src.agents.run_model.call_llm", side_effect=["resp1", "resp2"])
    def test_responses_match_prompts(self, mock_llm):
        state = _make_state(["p1", "p2"], provider="bedrock")
        run_model(state)
        assert len(state["responses"]) == 2
        assert state["responses"][0] == {"prompt": "p1", "response": "resp1"}
        assert state["responses"][1] == {"prompt": "p2", "response": "resp2"}

    @patch("src.agents.run_model.call_llm")
    def test_each_response_has_keys(self, mock_llm):
        mock_llm.return_value = "answer"
        state = _make_state(["q1"], provider="bedrock")
        run_model(state)
        entry = state["responses"][0]
        assert "prompt" in entry
        assert "response" in entry

    @patch("src.agents.run_model.call_llm")
    def test_empty_prompts(self, mock_llm):
        state = _make_state([], provider="bedrock")
        run_model(state)
        assert state["responses"] == []
        mock_llm.assert_not_called()
