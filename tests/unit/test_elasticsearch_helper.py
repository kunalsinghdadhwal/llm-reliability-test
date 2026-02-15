from unittest.mock import MagicMock, patch

import pytest

MOCK_SEARCH_RESPONSE = {
    "hits": {
        "hits": [
            {"_source": {"content": "doc text", "embedding": [0.1, 0.2]}, "_id": "1"},
            {"_source": {"content": "other doc", "embedding": [0.3, 0.4]}, "_id": "2"},
        ]
    }
}

EMPTY_SEARCH_RESPONSE = {"hits": {"hits": []}}


@pytest.fixture(autouse=True)
def _mock_es_and_embed():
    mock_es = MagicMock()
    mock_embed = MagicMock(return_value=[0.1, 0.2, 0.3])

    with (
        patch("src.wrappers.elasticsearch_helper.es", mock_es),
        patch("src.wrappers.elasticsearch_helper.embed", mock_embed),
    ):
        yield mock_es, mock_embed


@pytest.fixture()
def mock_es(_mock_es_and_embed):
    return _mock_es_and_embed[0]


@pytest.fixture()
def mock_embed(_mock_es_and_embed):
    return _mock_es_and_embed[1]


def test_index_doc_calls_es_index(mock_es):
    from src.wrappers.elasticsearch_helper import index_doc

    body = {"content": "hello", "embedding": [0.1]}
    index_doc(index="my_index", doc_id="42", body=body)

    mock_es.index.assert_called_once_with(index="my_index", id="42", document=body)


def test_search_docs_returns_source(mock_es):
    from src.wrappers.elasticsearch_helper import search_docs

    mock_es.search.return_value = MOCK_SEARCH_RESPONSE
    results = search_docs("hello")

    assert len(results) == 2
    assert results[0] == {"content": "doc text", "embedding": [0.1, 0.2]}
    assert results[1] == {"content": "other doc", "embedding": [0.3, 0.4]}


def test_search_docs_uses_match_query(mock_es):
    from src.wrappers.elasticsearch_helper import search_docs

    mock_es.search.return_value = MOCK_SEARCH_RESPONSE
    search_docs("my query")

    mock_es.search.assert_called_once_with(
        index="trusted_docs",
        query={"match": {"content": "my query"}},
    )


def test_vector_search_calls_embed(mock_es, mock_embed):
    from src.wrappers.elasticsearch_helper import vector_search

    mock_es.search.return_value = MOCK_SEARCH_RESPONSE
    vector_search("search text")

    mock_embed.assert_called_once_with("search text")


def test_vector_search_passes_embedding_to_knn(mock_es, mock_embed):
    from src.wrappers.elasticsearch_helper import vector_search

    mock_embed.return_value = [0.5, 0.6, 0.7]
    mock_es.search.return_value = MOCK_SEARCH_RESPONSE
    vector_search("search text", k=3)

    mock_es.search.assert_called_once_with(
        index="trusted_docs",
        knn={
            "field": "embedding",
            "query_vector": [0.5, 0.6, 0.7],
            "k": 3,
            "num_candidates": 100,
        },
    )


def test_vector_search_returns_source(mock_es, mock_embed):
    from src.wrappers.elasticsearch_helper import vector_search

    mock_es.search.return_value = MOCK_SEARCH_RESPONSE
    results = vector_search("search text")

    assert len(results) == 2
    assert results[0] == {"content": "doc text", "embedding": [0.1, 0.2]}
    assert results[1] == {"content": "other doc", "embedding": [0.3, 0.4]}


def test_empty_results_return_empty_list(mock_es):
    from src.wrappers.elasticsearch_helper import search_docs, vector_search

    mock_es.search.return_value = EMPTY_SEARCH_RESPONSE

    assert search_docs("nothing") == []

    mock_es.search.return_value = EMPTY_SEARCH_RESPONSE
    assert vector_search("nothing") == []


def test_es_client_initialized_once_at_module_level():
    from src.wrappers import elasticsearch_helper

    assert hasattr(elasticsearch_helper, "es")
    assert elasticsearch_helper.es is not None
    # Importing again should return the same module (same client instance)
    import importlib

    mod = importlib.import_module("src.wrappers.elasticsearch_helper")
    assert mod.es is elasticsearch_helper.es
