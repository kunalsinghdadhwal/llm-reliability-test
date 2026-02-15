"""Integration tests for the full evaluation pipeline via API."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api import app
from tests.integration.conftest import requires_elasticsearch, requires_ollama

MOCK_CLAIMS_RESPONSE = (
    "1. Acme Corp allows returns within 30 days of purchase.\n"
    "2. Shipping is free for orders over $50.\n"
    "3. All electronics carry a 2-year warranty."
)

MOCK_VERIFY_SUPPORTED = "LABEL: supported\nJUSTIFICATION: Evidence confirms this."

MOCK_PROMPTS_RESPONSE = (
    "1. What is the return policy?\n2. What is the shipping cost?\n3. What is the warranty?"
)


@requires_elasticsearch
@requires_ollama
class TestFullPipeline:
    """End-to-end tests that mock only LLM calls (Bedrock), using real ES."""

    def _get_client(self):
        return TestClient(app)

    @patch("src.api.index_doc")
    @patch("src.agents.verify_claims.call_llm", return_value=MOCK_VERIFY_SUPPORTED)
    @patch("src.agents.extract_claims.call_llm", return_value=MOCK_CLAIMS_RESPONSE)
    @patch("src.agents.generate_prompts.call_llm", return_value=MOCK_PROMPTS_RESPONSE)
    def test_evaluate_returns_valid_response(
        self, mock_gen, mock_extract, mock_verify, mock_index, ingest_fixtures
    ):
        client = self._get_client()
        resp = client.post("/evaluate", json={"config_path": "tests/fixtures/test_config.yaml"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["decision"] in ("approve", "reject")
        assert body["total_claims"] > 0
        assert len(body["claims"]) > 0
        assert "run_id" in body
        assert "model_under_test" in body
        assert "hallucination_risk" in body
        assert "reliability_score" in body

    @patch("src.api.index_doc")
    @patch("src.agents.verify_claims.call_llm", return_value=MOCK_VERIFY_SUPPORTED)
    @patch("src.agents.extract_claims.call_llm", return_value=MOCK_CLAIMS_RESPONSE)
    @patch("src.agents.generate_prompts.call_llm", return_value="1. What is the return policy?")
    def test_hallucination_risk_is_valid_float(
        self, mock_gen, mock_extract, mock_verify, mock_index, ingest_fixtures
    ):
        client = self._get_client()
        resp = client.post("/evaluate", json={"config_path": "tests/fixtures/test_config.yaml"})
        body = resp.json()
        assert isinstance(body["hallucination_risk"], float)
        assert 0.0 <= body["hallucination_risk"] <= 1.0
        assert isinstance(body["reliability_score"], float)
        assert 0.0 <= body["reliability_score"] <= 1.0

    def test_health_endpoint(self):
        client = self._get_client()
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
