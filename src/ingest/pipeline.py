"""doc_ingest -- offline pipeline to chunk, embed, and index trusted documents."""

import argparse
import hashlib
import re
from pathlib import Path

from src.config.loader import load_config
from src.wrappers.bedrock import embed
from src.wrappers.elasticsearch_helper import index_doc


def clean_text(raw: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    text = re.sub(r"<[^>]+>", "", raw)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into word-level chunks with overlap."""
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap

    return chunks


def run_ingest(config_path: str) -> dict:
    """Read, clean, chunk, embed, and index trusted documents."""
    config = load_config(config_path)
    index = config["elasticsearch"]["index"]
    sources = config["doc_sources"]

    documents_processed = 0
    chunks_indexed = 0

    for source in sources:
        source_type = source.get("type", "")

        if source_type == "s3":
            raise NotImplementedError("S3 source type is not yet supported.")

        if source_type != "local":
            raise ValueError(f"Unknown source type: {source_type}")

        source_path = Path(source["path"])
        if not source_path.exists():
            raise FileNotFoundError(f"Source path not found: {source['path']}")

        files = list(source_path.rglob("*.txt")) + list(source_path.rglob("*.md"))

        for file_path in sorted(files):
            raw = file_path.read_text(encoding="utf-8")
            cleaned = clean_text(raw)
            chunks = chunk_text(cleaned)
            documents_processed += 1

            for chunk in chunks:
                vector = embed(chunk)
                doc_id = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                index_doc(index, doc_id, {"content": chunk, "embedding": vector})
                chunks_indexed += 1

    return {"documents_processed": documents_processed, "chunks_indexed": chunks_indexed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest trusted documents into Elasticsearch.")
    parser.add_argument("--config", default=".llm-reliability.yaml", help="Path to config file.")
    args = parser.parse_args()
    result = run_ingest(args.config)
    print(f"Documents processed: {result['documents_processed']}")
    print(f"Chunks indexed: {result['chunks_indexed']}")
