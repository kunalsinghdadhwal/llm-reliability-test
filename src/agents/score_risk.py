"""score_risk -- compute hallucination risk score from claim verdicts."""


def score_risk(state: dict) -> None:
    """Read verdicts, compute risk score, write decision to state."""
    verdicts = state["verdicts"]
    threshold = state["config"]["thresholds"]["reject"]

    total = len(verdicts)

    if total == 0:
        hallucination_risk = 0.0
        unsupported_count = 0
        weakly_count = 0
        supported_count = 0
    else:
        unsupported_count = sum(1 for v in verdicts if v["verdict"] == "unsupported")
        weakly_count = sum(1 for v in verdicts if v["verdict"] == "weakly_supported")
        supported_count = total - unsupported_count - weakly_count
        hallucination_risk = unsupported_count / total

    reliability_score = 1 - hallucination_risk
    decision = "approve" if hallucination_risk < threshold else "reject"

    state["score"] = {
        "hallucination_risk": round(hallucination_risk, 4),
        "reliability_score": round(reliability_score, 4),
        "decision": decision,
        "total_claims": total,
        "supported": supported_count,
        "weakly_supported": weakly_count,
        "unsupported": unsupported_count,
    }
