"""Elasticsearch helper -- keyword and vector search against trusted docs."""

import os

from elasticsearch import Elasticsearch

from src.wrappers.bedrock import embed

_hosts = [os.environ.get("ES_HOST", "http://localhost:9200")]
_api_key = os.environ.get("ES_API_KEY")

_kwargs: dict = {"hosts": _hosts}
if _api_key:
    _kwargs["api_key"] = _api_key

es = Elasticsearch(**_kwargs)


def index_doc(index: str, doc_id: str, body: dict) -> None:
    es.index(index=index, id=doc_id, document=body)


def search_docs(query: str, index: str = "trusted_docs") -> list[dict]:
    response = es.search(index=index, query={"match": {"content": query}})
    return [hit["_source"] for hit in response["hits"]["hits"]]


def vector_search(text: str, index: str = "trusted_docs", k: int = 5) -> list[dict]:
    vector = embed(text)
    response = es.search(
        index=index,
        knn={
            "field": "embedding",
            "query_vector": vector,
            "k": k,
            "num_candidates": 100,
        },
    )
    return [hit["_source"] for hit in response["hits"]["hits"]]
