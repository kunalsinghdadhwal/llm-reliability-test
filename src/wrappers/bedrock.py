"""AWS Bedrock wrapper -- single point of contact with AWS Bedrock.

Supports routing to Ollama when EVALUATOR_PROVIDER=ollama is set.
"""

import hashlib
import json
import os


def _get_bedrock_client():
    """Lazily create Bedrock client (only when actually needed)."""
    import boto3
    return boto3.client(
        "bedrock-runtime",
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )


def call_llm(prompt: str, system: str = "") -> str:
    """Call LLM - routes to Ollama if EVALUATOR_PROVIDER=ollama, else Bedrock."""
    if os.environ.get("EVALUATOR_PROVIDER") == "ollama":
        from ollama import Client as OllamaClient

        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        model_id = os.environ.get("EVALUATOR_MODEL_ID", "qwen2.5:3b")
        client = OllamaClient(host=host)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = client.chat(model=model_id, messages=messages)
        return response.message.content or ""

    # Original Bedrock path
    client = _get_bedrock_client()
    model_id = os.environ.get(
        "BEDROCK_INFERENCE_PROFILE_ID", "anthropic.claude-3-sonnet-20240229-v1:0"
    )
    body: dict = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096,
        "temperature": 0,
    }
    if system:
        body["system"] = system
    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"]


def embed(text: str) -> list[float]:
    """Generate an embedding vector.

    Uses a deterministic hash-based embedding when EVALUATOR_PROVIDER=ollama
    (avoids needing AWS Bedrock for Titan embeddings).
    Otherwise uses Bedrock Titan.
    """
    if os.environ.get("EVALUATOR_PROVIDER") == "ollama":
        # Hash-based fake embedding (256 dims) for Ollama-only mode
        # Chain multiple hashes to get enough bytes for 256 dimensions
        vectors = []
        for seed in range(8):
            h = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()
            vectors.extend(int(h[i : i + 2], 16) / 255.0 for i in range(0, 64, 2))
        return vectors[:256]

    client = _get_bedrock_client()
    model_id = os.environ.get("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps({"inputText": text}),
        accept="application/json",
        contentType="application/json",
    )
    return json.loads(response["body"].read())["embedding"]
