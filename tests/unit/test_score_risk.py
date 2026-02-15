"""Tests for score_risk agent."""

from src.agents.score_risk import score_risk


def _make_state(verdicts_labels, reject=0.3):
    """Build a minimal state dict for testing."""
    return {
        "verdicts": [{"verdict": label} for label in verdicts_labels],
        "config": {"thresholds": {"reject": reject}},
    }


class TestScoreRisk:
    def test_all_supported(self):
        state = _make_state(["supported", "supported", "supported"])
        score_risk(state)
        assert state["score"]["hallucination_risk"] == 0.0
        assert state["score"]["reliability_score"] == 1.0
        assert state["score"]["decision"] == "approve"

    def test_all_unsupported(self):
        state = _make_state(["unsupported", "unsupported", "unsupported"])
        score_risk(state)
        assert state["score"]["hallucination_risk"] == 1.0
        assert state["score"]["reliability_score"] == 0.0
        assert state["score"]["decision"] == "reject"

    def test_all_weakly_supported(self):
        state = _make_state(["weakly_supported"] * 4)
        score_risk(state)
        # weakly_supported are not counted as unsupported
        assert state["score"]["hallucination_risk"] == 0.0
        assert state["score"]["decision"] == "approve"

    def test_mixed_verdicts(self):
        labels = ["unsupported", "weakly_supported", "supported", "supported"]
        state = _make_state(labels)
        score_risk(state)
        # 1 unsupported / 4 total = 0.25
        assert state["score"]["hallucination_risk"] == 0.25
        assert state["score"]["reliability_score"] == 0.75
        assert state["score"]["decision"] == "approve"

    def test_empty_verdicts(self):
        state = _make_state([])
        score_risk(state)
        assert state["score"]["hallucination_risk"] == 0.0
        assert state["score"]["reliability_score"] == 1.0
        assert state["score"]["decision"] == "approve"

    def test_boundary_at_threshold_rejects(self):
        # risk = 1/3 ~= 0.3333 >= 0.3 threshold => reject
        labels = ["unsupported", "supported", "supported"]
        state = _make_state(labels, reject=0.3)
        score_risk(state)
        assert state["score"]["hallucination_risk"] == round(1 / 3, 4)
        assert state["score"]["decision"] == "reject"

    def test_boundary_below_threshold_approves(self):
        # risk = 1/4 = 0.25 < 0.3 threshold => approve
        labels = ["unsupported", "supported", "supported", "supported"]
        state = _make_state(labels, reject=0.3)
        score_risk(state)
        assert state["score"]["hallucination_risk"] == 0.25
        assert state["score"]["decision"] == "approve"

    def test_score_has_all_keys(self):
        state = _make_state(["supported", "unsupported"])
        score_risk(state)
        expected_keys = {
            "hallucination_risk",
            "reliability_score",
            "decision",
            "total_claims",
            "supported",
            "weakly_supported",
            "unsupported",
        }
        assert set(state["score"].keys()) == expected_keys

    def test_counts_are_correct(self):
        labels = ["supported", "supported", "weakly_supported", "unsupported"]
        state = _make_state(labels)
        score_risk(state)
        assert state["score"]["total_claims"] == 4
        assert state["score"]["supported"] == 2
        assert state["score"]["weakly_supported"] == 1
        assert state["score"]["unsupported"] == 1
