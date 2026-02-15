"""Shared fixtures for integration tests."""

import contextlib
import hashlib
import time

import pytest
import requests

from src.ingest.pipeline import run_ingest


def _check_elasticsearch():
    """Return True if Elasticsearch is reachable."""
    try:
        resp = requests.get("http://localhost:9200", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def _check_ollama():
    """Return True if Ollama is reachable."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


requires_elasticsearch = pytest.mark.skipif(
    not _check_elasticsearch(),
    reason="Elasticsearch not available at localhost:9200",
)

requires_ollama = pytest.mark.skipif(
    not _check_ollama(),
    reason="Ollama not available at localhost:11434",
)


def _fake_embed(text):
    """Hash-based fake embedding to avoid requiring AWS Bedrock for Titan."""
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return [int(h[i : i + 2], 16) / 255.0 for i in range(0, 512, 2)]


@pytest.fixture(scope="session", autouse=True)
def _patch_embed():
    """Replace real embed with hash-based fake for all integration tests."""
    from unittest.mock import patch

    with (
        patch("src.ingest.pipeline.embed", side_effect=_fake_embed),
        patch("src.wrappers.elasticsearch_helper.embed", side_effect=_fake_embed),
    ):
        yield


@pytest.fixture(scope="session")
def es_client():
    """Return an Elasticsearch client connected to localhost."""
    from elasticsearch import Elasticsearch

    client = Elasticsearch("http://localhost:9200")
    yield client
    # Cleanup: delete test index
    with contextlib.suppress(Exception):
        client.indices.delete(index="test_trusted_docs", ignore_unavailable=True)


@pytest.fixture(scope="session")
def es_index(es_client):
    """Create the test index with dense_vector mapping before ingest."""
    index_name = "test_trusted_docs"
    if es_client.indices.exists(index=index_name):
        es_client.indices.delete(index=index_name)

    es_client.indices.create(
        index=index_name,
        body={
            "mappings": {
                "properties": {
                    "content": {"type": "text"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 256,
                        "index": True,
                        "similarity": "cosine",
                    },
                }
            }
        },
    )
    yield index_name


@pytest.fixture(scope="session")
def ingest_fixtures(es_index):
    """Run the ingest pipeline once for all integration tests."""
    result = run_ingest("tests/fixtures/test_config.yaml")
    # Give ES a moment to index
    time.sleep(2)
    return result
