"""Tests for workflow orchestrator."""

from unittest.mock import patch

import pytest

from src.orchestrator import build_response, run_workflow

MODULE = "src.orchestrator"

AGENT_NAMES = [
    "generate_prompts",
    "run_model",
    "extract_claims",
    "retrieve_evidence",
    "verify_claims",
    "score_risk",
]


def _name_mocks(mocks, names):
    """Set __name__ on each mock so the orchestrator can log it."""
    for mock, name in zip(mocks, names, strict=True):
        mock.__name__ = name


class TestRunWorkflow:
    @patch(f"{MODULE}.score_risk")
    @patch(f"{MODULE}.verify_claims")
    @patch(f"{MODULE}.retrieve_evidence")
    @patch(f"{MODULE}.extract_claims")
    @patch(f"{MODULE}.run_model")
    @patch(f"{MODULE}.generate_prompts")
    def test_agents_called_in_order(self, *mocks):
        # patch decorators pass args innermost-first: generate_prompts, ..., score_risk
        _name_mocks(mocks, AGENT_NAMES)
        call_order = []
        for mock in mocks:
            mock.side_effect = lambda s, m=mock: call_order.append(m)

        state = {}
        run_workflow(state)

        assert len(call_order) == 6
        assert call_order == list(mocks)

    @patch(f"{MODULE}.score_risk")
    @patch(f"{MODULE}.verify_claims")
    @patch(f"{MODULE}.retrieve_evidence")
    @patch(f"{MODULE}.extract_claims")
    @patch(f"{MODULE}.run_model")
    @patch(f"{MODULE}.generate_prompts")
    def test_state_passed_to_each_agent(self, *mocks):
        _name_mocks(mocks, AGENT_NAMES)
        state = {"config": {}}
        run_workflow(state)
        for mock in mocks:
            mock.assert_called_once_with(state)

    @patch(f"{MODULE}.score_risk")
    @patch(f"{MODULE}.verify_claims")
    @patch(f"{MODULE}.retrieve_evidence")
    @patch(f"{MODULE}.extract_claims")
    @patch(f"{MODULE}.run_model")
    @patch(f"{MODULE}.generate_prompts")
    def test_exception_stops_pipeline(
        self, mock_gen, mock_run, mock_extract, mock_retrieve, mock_verify, mock_score
    ):
        for mock, name in zip(
            [mock_gen, mock_run, mock_extract, mock_retrieve, mock_verify, mock_score],
            AGENT_NAMES,
            strict=True,
        ):
            mock.__name__ = name

        mock_extract.side_effect = RuntimeError("LLM error")

        with pytest.raises(RuntimeError, match="LLM error"):
            run_workflow({})

        mock_gen.assert_called_once()
        mock_run.assert_called_once()
        mock_extract.assert_called_once()
        mock_retrieve.assert_not_called()
        mock_verify.assert_not_called()
        mock_score.assert_not_called()

    @patch(f"{MODULE}.score_risk")
    @patch(f"{MODULE}.verify_claims")
    @patch(f"{MODULE}.retrieve_evidence")
    @patch(f"{MODULE}.extract_claims")
    @patch(f"{MODULE}.run_model")
    @patch(f"{MODULE}.generate_prompts")
    def test_logging_output(self, *mocks):
        _name_mocks(mocks, AGENT_NAMES)
        with patch(f"{MODULE}.logger") as mock_logger:
            run_workflow({})
            info_messages = [c[0][1] for c in mock_logger.info.call_args_list]
            assert "generate_prompts" in info_messages
            assert "score_risk" in info_messages


class TestBuildResponse:
    def test_extracts_correct_fields(self):
        state = {
            "config": {"model": {"provider": "bedrock"}},
            "score": {
                "hallucination_risk": 0.25,
                "reliability_score": 0.75,
                "decision": "approve",
                "total_claims": 10,
                "supported": 6,
                "unsupported": 2,
                "weakly_supported": 2,
            },
            "verdicts": [
                {
                    "claim": "test",
                    "verdict": "supported",
                    "evidence_snippet": "ev",
                    "confidence": 1.0,
                }
            ],
        }
        result = build_response(state)
        assert result["hallucination_risk"] == 0.25
        assert result["reliability_score"] == 0.75
        assert result["decision"] == "approve"
        assert result["total_claims"] == 10
        assert result["supported"] == 6
        assert result["unsupported"] == 2
        assert result["weakly_supported"] == 2
        assert result["model_under_test"] == "bedrock"
        assert "run_id" in result
        assert len(result["claims"]) == 1

    def test_missing_verdicts_handled(self):
        state = {
            "config": {"model": {"provider": "bedrock"}},
            "score": {
                "hallucination_risk": 0.0,
                "reliability_score": 1.0,
                "decision": "approve",
                "total_claims": 0,
                "supported": 0,
                "unsupported": 0,
                "weakly_supported": 0,
            },
        }
        result = build_response(state)
        assert result["claims"] == []

    def test_response_has_all_keys(self):
        state = {
            "config": {"model": {"provider": "bedrock"}},
            "score": {
                "hallucination_risk": 0.5,
                "reliability_score": 0.5,
                "decision": "reject",
                "total_claims": 4,
                "supported": 1,
                "unsupported": 2,
                "weakly_supported": 1,
            },
            "verdicts": [],
        }
        result = build_response(state)
        expected_keys = {
            "run_id",
            "model_under_test",
            "total_claims",
            "supported",
            "weakly_supported",
            "unsupported",
            "hallucination_risk",
            "reliability_score",
            "decision",
            "claims",
        }
        assert set(result.keys()) == expected_keys
