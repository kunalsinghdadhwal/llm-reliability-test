"""Workflow orchestrator -- wires all agents into a sequential pipeline."""

import logging
from uuid import uuid4

from src.agents.extract_claims import extract_claims
from src.agents.generate_prompts import generate_prompts
from src.agents.retrieve_evidence import retrieve_evidence
from src.agents.run_model import run_model
from src.agents.score_risk import score_risk
from src.agents.verify_claims import verify_claims

logger = logging.getLogger(__name__)


def run_workflow(state: dict) -> None:
    """Execute all agents in pipeline order."""
    agents = [
        generate_prompts,
        run_model,
        extract_claims,
        retrieve_evidence,
        verify_claims,
        score_risk,
    ]
    for agent in agents:
        name = agent.__name__
        logger.info("Starting %s", name)
        try:
            agent(state)
        except Exception as exc:
            logger.error("Failed in %s: %s", name, exc)
            raise
        logger.info("Completed %s", name)


def build_response(state: dict) -> dict:
    """Extract a clean response dict from the final state."""
    score = state["score"]
    return {
        "run_id": str(uuid4()),
        "model_under_test": state["config"]["model"]["provider"],
        "total_claims": score["total_claims"],
        "supported": score["supported"],
        "weakly_supported": score["weakly_supported"],
        "unsupported": score["unsupported"],
        "hallucination_risk": score["hallucination_risk"],
        "reliability_score": score["reliability_score"],
        "decision": score["decision"],
        "claims": state.get("verdicts", []),
    }
