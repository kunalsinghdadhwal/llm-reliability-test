"""Integration tests for document ingestion pipeline."""

from tests.integration.conftest import requires_elasticsearch, requires_ollama


@requires_elasticsearch
@requires_ollama
class TestIngest:
    def test_correct_document_count(self, ingest_fixtures):
        assert ingest_fixtures["documents_processed"] == 3

    def test_keyword_search_finds_return_policy(self, ingest_fixtures, es_client):
        results = es_client.search(
            index="test_trusted_docs",
            body={"query": {"match": {"content": "return policy"}}},
        )
        hits = results["hits"]["hits"]
        assert len(hits) > 0
        assert any("30 days" in hit["_source"]["content"] for hit in hits)

    def test_vector_search_finds_results(self, ingest_fixtures, es_client):
        from tests.integration.conftest import _fake_embed

        query_vector = _fake_embed("refund items")
        results = es_client.search(
            index="test_trusted_docs",
            body={
                "knn": {
                    "field": "embedding",
                    "query_vector": query_vector,
                    "k": 3,
                    "num_candidates": 10,
                }
            },
        )
        hits = results["hits"]["hits"]
        assert len(hits) > 0

    def test_indexed_docs_have_required_fields(self, ingest_fixtures, es_client):
        results = es_client.search(
            index="test_trusted_docs",
            body={"query": {"match_all": {}}, "size": 1},
        )
        hit = results["hits"]["hits"][0]["_source"]
        assert "content" in hit
        assert "embedding" in hit
