"""Tests for generate_prompts agent."""

from unittest.mock import patch

from src.agents.generate_prompts import generate_prompts, parse_prompts


def _make_state(num_prompts=3):
    return {
        "config": {
            "use_case": "customer support chatbot",
            "evaluation": {
                "num_prompts": num_prompts,
                "prompt_categories": ["returns", "shipping", "pricing"],
            },
        },
    }


NUMBERED_RESPONSE = (
    "1. What is your return policy for electronics?\n"
    "2. Can I return items after 30 days?\n"
    "3. Do you offer free shipping?"
)

BULLET_RESPONSE = (
    "- What is your return policy for electronics?\n"
    "- Can I return items after 30 days?\n"
    "- Do you offer free shipping?"
)


class TestParsePrompts:
    def test_numbered_list(self):
        result = parse_prompts(NUMBERED_RESPONSE)
        assert len(result) == 3
        assert result[0] == "What is your return policy for electronics?"

    def test_bullet_list(self):
        result = parse_prompts(BULLET_RESPONSE)
        assert len(result) == 3
        assert result[0] == "What is your return policy for electronics?"

    def test_strips_numbering(self):
        result = parse_prompts("1. First prompt\n2. Second prompt")
        assert result[0] == "First prompt"
        assert result[1] == "Second prompt"

    def test_empty_response(self):
        result = parse_prompts("")
        assert result == []

    def test_filters_empty_lines(self):
        result = parse_prompts("1. First\n\n2. Second\n\n")
        assert len(result) == 2


class TestGeneratePrompts:
    @patch("src.agents.generate_prompts.call_llm", return_value=NUMBERED_RESPONSE)
    def test_parses_numbered_response(self, mock_llm):
        state = _make_state(num_prompts=3)
        generate_prompts(state)
        assert len(state["prompts"]) == 3
        assert state["prompts"][0] == "What is your return policy for electronics?"

    @patch("src.agents.generate_prompts.call_llm", return_value=NUMBERED_RESPONSE)
    def test_truncates_to_num_prompts(self, mock_llm):
        state = _make_state(num_prompts=2)
        generate_prompts(state)
        assert len(state["prompts"]) == 2

    @patch("src.agents.generate_prompts.call_llm", return_value="")
    def test_empty_response_produces_empty_list(self, mock_llm):
        state = _make_state(num_prompts=3)
        generate_prompts(state)
        assert state["prompts"] == []

    @patch("src.agents.generate_prompts.call_llm", return_value=NUMBERED_RESPONSE)
    def test_call_llm_called_once(self, mock_llm):
        state = _make_state(num_prompts=3)
        generate_prompts(state)
        mock_llm.assert_called_once()

    @patch("src.agents.generate_prompts.call_llm", return_value=NUMBERED_RESPONSE)
    def test_prompts_is_list_of_strings(self, mock_llm):
        state = _make_state(num_prompts=3)
        generate_prompts(state)
        assert isinstance(state["prompts"], list)
        assert all(isinstance(p, str) for p in state["prompts"])
