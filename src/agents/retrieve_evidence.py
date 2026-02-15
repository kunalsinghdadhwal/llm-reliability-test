"""retrieve_evidence -- query ES for each claim using keyword + vector hybrid search."""

from src.wrappers.elasticsearch_helper import search_docs, vector_search


def deduplicate(results: list[dict]) -> list[dict]:
    """Remove duplicate documents, preserving order (first occurrence wins).

    Deduplicates by "content" field if present, otherwise by full dict equality.
    """
    seen = []
    unique = []
    for doc in results:
        key = doc.get("content", None)
        if key is not None:
            if key not in seen:
                seen.append(key)
                unique.append(doc)
        else:
            if doc not in unique:
                unique.append(doc)
    return unique


def retrieve_evidence(state: dict) -> None:
    """Retrieve evidence documents for each claim via dual search."""
    claims = state["claims"]
    index = state["config"]["elasticsearch"]["index"]

    evidence = []
    for claim in claims:
        keyword_results = search_docs(claim["text"], index=index)
        vector_results = vector_search(claim["text"], index=index)
        combined = deduplicate(keyword_results + vector_results)
        evidence.append({"claim": claim, "documents": combined})

    state["evidence"] = evidence
