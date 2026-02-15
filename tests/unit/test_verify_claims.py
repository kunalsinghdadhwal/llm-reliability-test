"""Tests for verify_claims agent."""

from unittest.mock import patch

from src.agents.verify_claims import parse_verdict, verify_claims

SUPPORTED_RESPONSE = (
    "LABEL: supported\nJUSTIFICATION: The evidence directly confirms the 30-day return policy."
)
UNSUPPORTED_RESPONSE = "LABEL: unsupported\nJUSTIFICATION: No evidence mentions this pricing."
WEAKLY_RESPONSE = "LABEL: weakly_supported\nJUSTIFICATION: Evidence partially addresses the claim."


def _make_state(evidence_entries):
    return {"evidence": evidence_entries}


def _make_entry(claim_text, doc_contents):
    return {
        "claim": {"text": claim_text, "source_prompt": "p", "source_response": "r"},
        "documents": [{"content": c} for c in doc_contents],
    }


class TestParseVerdict:
    def test_supported(self):
        label, justification = parse_verdict(SUPPORTED_RESPONSE)
        assert label == "supported"
        assert "30-day" in justification

    def test_unsupported(self):
        label, _justification = parse_verdict(UNSUPPORTED_RESPONSE)
        assert label == "unsupported"

    def test_weakly_supported(self):
        label, _justification = parse_verdict(WEAKLY_RESPONSE)
        assert label == "weakly_supported"

    def test_malformed_response(self):
        label, justification = parse_verdict("This is not a valid response")
        assert label == "unsupported"
        assert justification == "Could not parse verification response."

    def test_label_always_lowercase(self):
        label, _ = parse_verdict("LABEL: SUPPORTED\nJUSTIFICATION: ok")
        assert label == "supported"

    def test_invalid_label_defaults_unsupported(self):
        label, _ = parse_verdict("LABEL: maybe\nJUSTIFICATION: not sure")
        assert label == "unsupported"

    def test_missing_justification(self):
        label, justification = parse_verdict("LABEL: supported")
        assert label == "supported"
        assert justification == "Could not parse verification response."


class TestVerifyClaims:
    @patch("src.agents.verify_claims.call_llm", return_value=SUPPORTED_RESPONSE)
    def test_supported_verdict(self, mock_llm):
        entry = _make_entry("claim A", ["evidence text"])
        state = _make_state([entry])
        verify_claims(state)
        assert state["verdicts"][0]["verdict"] == "supported"
        assert state["verdicts"][0]["confidence"] == 1.0
        assert state["verdicts"][0]["evidence_snippet"] == "evidence text"

    @patch("src.agents.verify_claims.call_llm")
    def test_empty_docs_skips_llm(self, mock_llm):
        entry = {
            "claim": {"text": "claim A", "source_prompt": "p", "source_response": "r"},
            "documents": [],
        }
        state = _make_state([entry])
        verify_claims(state)
        mock_llm.assert_not_called()
        assert state["verdicts"][0]["verdict"] == "unsupported"
        assert state["verdicts"][0]["evidence_snippet"] == ""
        assert state["verdicts"][0]["confidence"] == 0.0

    @patch("src.agents.verify_claims.call_llm", return_value=SUPPORTED_RESPONSE)
    def test_one_verdict_per_entry(self, mock_llm):
        entries = [
            _make_entry("claim 1", ["doc1"]),
            _make_entry("claim 2", ["doc2"]),
        ]
        state = _make_state(entries)
        verify_claims(state)
        assert len(state["verdicts"]) == 2
        assert mock_llm.call_count == 2

    @patch("src.agents.verify_claims.call_llm", return_value=SUPPORTED_RESPONSE)
    def test_claim_text_in_verdict(self, mock_llm):
        entry = _make_entry("Returns are allowed within 30 days", ["evidence"])
        state = _make_state([entry])
        verify_claims(state)
        assert state["verdicts"][0]["claim"] == "Returns are allowed within 30 days"

    @patch("src.agents.verify_claims.call_llm", return_value=SUPPORTED_RESPONSE)
    def test_prompt_contains_only_claim_and_evidence(self, mock_llm):
        entry = _make_entry("test claim", ["evidence content"])
        state = _make_state([entry])
        verify_claims(state)
        prompt_arg = mock_llm.call_args[0][0]
        assert "test claim" in prompt_arg
        assert "evidence content" in prompt_arg

    @patch("src.agents.verify_claims.call_llm", return_value="gibberish output")
    def test_malformed_llm_response(self, mock_llm):
        entry = _make_entry("claim", ["evidence"])
        state = _make_state([entry])
        verify_claims(state)
        assert state["verdicts"][0]["verdict"] == "unsupported"
        assert state["verdicts"][0]["confidence"] == 0.0

    @patch(
        "src.agents.verify_claims.call_llm",
        side_effect=[SUPPORTED_RESPONSE, UNSUPPORTED_RESPONSE],
    )
    def test_mixed_verdicts(self, mock_llm):
        entries = [
            _make_entry("claim 1", ["doc1"]),
            _make_entry("claim 2", ["doc2"]),
        ]
        state = _make_state(entries)
        verify_claims(state)
        assert state["verdicts"][0]["verdict"] == "supported"
        assert state["verdicts"][0]["confidence"] == 1.0
        assert state["verdicts"][1]["verdict"] == "unsupported"
        assert state["verdicts"][1]["confidence"] == 0.0

    @patch("src.agents.verify_claims.call_llm", return_value=SUPPORTED_RESPONSE)
    def test_multiple_docs_concatenated(self, mock_llm):
        entry = _make_entry("claim", ["doc part 1", "doc part 2"])
        state = _make_state([entry])
        verify_claims(state)
        prompt_arg = mock_llm.call_args[0][0]
        assert "doc part 1" in prompt_arg
        assert "doc part 2" in prompt_arg

    @patch("src.agents.verify_claims.call_llm")
    def test_empty_evidence_no_llm_call(self, mock_llm):
        state = _make_state([])
        verify_claims(state)
        assert state["verdicts"] == []
        mock_llm.assert_not_called()

    @patch("src.agents.verify_claims.call_llm", return_value=SUPPORTED_RESPONSE)
    def test_evidence_snippet_truncated_to_200_chars(self, mock_llm):
        long_content = "x" * 500
        entry = _make_entry("claim", [long_content])
        state = _make_state([entry])
        verify_claims(state)
        assert len(state["verdicts"][0]["evidence_snippet"]) == 200

    @patch("src.agents.verify_claims.call_llm", return_value=WEAKLY_RESPONSE)
    def test_weakly_supported_confidence(self, mock_llm):
        entry = _make_entry("claim", ["evidence"])
        state = _make_state([entry])
        verify_claims(state)
        assert state["verdicts"][0]["verdict"] == "weakly_supported"
        assert state["verdicts"][0]["confidence"] == 0.5

    @patch("src.agents.verify_claims.call_llm", return_value=SUPPORTED_RESPONSE)
    def test_verdict_keys(self, mock_llm):
        entry = _make_entry("claim", ["evidence"])
        state = _make_state([entry])
        verify_claims(state)
        expected_keys = {"claim", "verdict", "evidence_snippet", "confidence"}
        assert set(state["verdicts"][0].keys()) == expected_keys
