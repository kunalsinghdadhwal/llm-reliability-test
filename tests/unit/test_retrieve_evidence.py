"""Tests for retrieve_evidence agent."""

from unittest.mock import patch

from src.agents.retrieve_evidence import deduplicate, retrieve_evidence

KEYWORD_RESULTS = [{"content": "doc A"}, {"content": "doc B"}]
VECTOR_RESULTS = [{"content": "doc B"}, {"content": "doc C"}]


def _make_state(claims, index="test_index"):
    return {
        "claims": claims,
        "config": {"elasticsearch": {"index": index}},
    }


class TestDeduplicate:
    def test_removes_duplicates_by_content(self):
        results = [{"content": "A"}, {"content": "B"}, {"content": "A"}]
        deduped = deduplicate(results)
        assert len(deduped) == 2
        assert deduped[0]["content"] == "A"
        assert deduped[1]["content"] == "B"

    def test_preserves_order_first_wins(self):
        results = [{"content": "X", "score": 1}, {"content": "X", "score": 2}]
        deduped = deduplicate(results)
        assert len(deduped) == 1
        assert deduped[0]["score"] == 1

    def test_empty_list(self):
        assert deduplicate([]) == []

    def test_no_content_key_uses_dict_equality(self):
        results = [{"id": 1}, {"id": 2}, {"id": 1}]
        deduped = deduplicate(results)
        assert len(deduped) == 2


class TestRetrieveEvidence:
    @patch("src.agents.retrieve_evidence.vector_search", return_value=VECTOR_RESULTS)
    @patch("src.agents.retrieve_evidence.search_docs", return_value=KEYWORD_RESULTS)
    def test_each_claim_produces_one_entry(self, mock_keyword, mock_vector):
        claims = [
            {"text": "claim 1", "source_prompt": "p1", "source_response": "r1"},
            {"text": "claim 2", "source_prompt": "p2", "source_response": "r2"},
        ]
        state = _make_state(claims)
        retrieve_evidence(state)
        assert len(state["evidence"]) == 2

    @patch("src.agents.retrieve_evidence.vector_search", return_value=VECTOR_RESULTS)
    @patch("src.agents.retrieve_evidence.search_docs", return_value=KEYWORD_RESULTS)
    def test_keyword_and_vector_combined_and_deduped(self, mock_keyword, mock_vector):
        claims = [{"text": "claim 1", "source_prompt": "p", "source_response": "r"}]
        state = _make_state(claims)
        retrieve_evidence(state)
        docs = state["evidence"][0]["documents"]
        contents = [d["content"] for d in docs]
        assert contents == ["doc A", "doc B", "doc C"]

    @patch("src.agents.retrieve_evidence.vector_search", return_value=[])
    @patch("src.agents.retrieve_evidence.search_docs", return_value=[])
    def test_no_results_empty_documents(self, mock_keyword, mock_vector):
        claims = [{"text": "claim 1", "source_prompt": "p", "source_response": "r"}]
        state = _make_state(claims)
        retrieve_evidence(state)
        assert state["evidence"][0]["documents"] == []

    @patch("src.agents.retrieve_evidence.vector_search", return_value=[])
    @patch("src.agents.retrieve_evidence.search_docs", return_value=[])
    def test_config_index_used(self, mock_keyword, mock_vector):
        claims = [{"text": "claim", "source_prompt": "p", "source_response": "r"}]
        state = _make_state(claims, index="my_custom_index")
        retrieve_evidence(state)
        mock_keyword.assert_called_with("claim", index="my_custom_index")
        mock_vector.assert_called_with("claim", index="my_custom_index")

    @patch("src.agents.retrieve_evidence.vector_search", return_value=VECTOR_RESULTS)
    @patch("src.agents.retrieve_evidence.search_docs", return_value=KEYWORD_RESULTS)
    def test_search_called_once_per_claim(self, mock_keyword, mock_vector):
        claims = [
            {"text": "c1", "source_prompt": "p", "source_response": "r"},
            {"text": "c2", "source_prompt": "p", "source_response": "r"},
        ]
        state = _make_state(claims)
        retrieve_evidence(state)
        assert mock_keyword.call_count == 2
        assert mock_vector.call_count == 2

    @patch("src.agents.retrieve_evidence.vector_search", return_value=VECTOR_RESULTS)
    @patch("src.agents.retrieve_evidence.search_docs", return_value=KEYWORD_RESULTS)
    def test_claim_dict_preserved_in_evidence(self, mock_keyword, mock_vector):
        claim = {"text": "test claim", "source_prompt": "sp", "source_response": "sr"}
        state = _make_state([claim])
        retrieve_evidence(state)
        assert state["evidence"][0]["claim"] is claim

    @patch("src.agents.retrieve_evidence.vector_search")
    @patch("src.agents.retrieve_evidence.search_docs")
    def test_empty_claims_no_search_calls(self, mock_keyword, mock_vector):
        state = _make_state([])
        retrieve_evidence(state)
        assert state["evidence"] == []
        mock_keyword.assert_not_called()
        mock_vector.assert_not_called()
