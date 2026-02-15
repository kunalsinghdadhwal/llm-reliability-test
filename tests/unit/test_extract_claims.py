"""Tests for extract_claims agent."""

from unittest.mock import patch

from src.agents.extract_claims import extract_claims, parse_claims

CLAIMS_RESPONSE = (
    "1. The return policy allows returns within 30 days.\n2. Shipping is free for orders over $50."
)


class TestParseClaims:
    def test_numbered_list(self):
        result = parse_claims(CLAIMS_RESPONSE)
        assert len(result) == 2
        assert result[0] == "The return policy allows returns within 30 days."
        assert result[1] == "Shipping is free for orders over $50."

    def test_no_claims(self):
        result = parse_claims("NO CLAIMS")
        assert result == []

    def test_no_claims_case_insensitive(self):
        result = parse_claims("no claims")
        assert result == []

    def test_strips_numbering(self):
        result = parse_claims("1. First claim\n2. Second claim")
        assert result[0] == "First claim"

    def test_empty_string(self):
        result = parse_claims("")
        assert result == []

    def test_filters_empty_lines(self):
        result = parse_claims("1. First\n\n2. Second\n\n")
        assert len(result) == 2

    def test_parenthesis_numbering(self):
        result = parse_claims("1) First claim\n2) Second claim")
        assert len(result) == 2
        assert result[0] == "First claim"
        assert result[1] == "Second claim"


class TestExtractClaims:
    @patch("src.agents.extract_claims.call_llm", return_value=CLAIMS_RESPONSE)
    def test_two_claims_from_one_response(self, mock_llm):
        state = {
            "responses": [
                {"prompt": "test prompt", "response": "test response"},
            ],
        }
        extract_claims(state)
        assert len(state["claims"]) == 2

    @patch("src.agents.extract_claims.call_llm", return_value=CLAIMS_RESPONSE)
    def test_claim_has_required_keys(self, mock_llm):
        state = {
            "responses": [
                {"prompt": "test prompt", "response": "test response"},
            ],
        }
        extract_claims(state)
        claim = state["claims"][0]
        assert "text" in claim
        assert "source_prompt" in claim
        assert "source_response" in claim

    @patch("src.agents.extract_claims.call_llm", return_value=CLAIMS_RESPONSE)
    def test_traceability(self, mock_llm):
        state = {
            "responses": [
                {"prompt": "my prompt", "response": "my response"},
            ],
        }
        extract_claims(state)
        claim = state["claims"][0]
        assert claim["source_prompt"] == "my prompt"
        assert claim["source_response"] == "my response"

    @patch("src.agents.extract_claims.call_llm", return_value="NO CLAIMS")
    def test_no_claims_produces_empty_list(self, mock_llm):
        state = {
            "responses": [
                {"prompt": "test", "response": "test"},
            ],
        }
        extract_claims(state)
        assert state["claims"] == []

    @patch("src.agents.extract_claims.call_llm", return_value=CLAIMS_RESPONSE)
    def test_call_llm_called_once_per_response(self, mock_llm):
        state = {
            "responses": [
                {"prompt": "p1", "response": "r1"},
                {"prompt": "p2", "response": "r2"},
            ],
        }
        extract_claims(state)
        assert mock_llm.call_count == 2

    @patch("src.agents.extract_claims.call_llm")
    def test_empty_responses_no_llm_call(self, mock_llm):
        state = {"responses": []}
        extract_claims(state)
        assert state["claims"] == []
        mock_llm.assert_not_called()

    @patch(
        "src.agents.extract_claims.call_llm",
        side_effect=[CLAIMS_RESPONSE, "1. Third claim."],
    )
    def test_multiple_responses_flat_list(self, mock_llm):
        state = {
            "responses": [
                {"prompt": "p1", "response": "r1"},
                {"prompt": "p2", "response": "r2"},
            ],
        }
        extract_claims(state)
        assert len(state["claims"]) == 3
        assert state["claims"][2]["source_prompt"] == "p2"
