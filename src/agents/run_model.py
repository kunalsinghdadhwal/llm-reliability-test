"""run_model -- run test prompts against the target LLM."""

import os

from src.wrappers.bedrock import call_llm


def call_target_llm(prompt: str, model_config: dict) -> str:
    """Route a prompt to the correct provider."""
    provider = model_config["provider"]

    if provider == "bedrock":
        return call_llm(prompt)
    elif provider == "ollama":
        from ollama import Client as OllamaClient

        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        client = OllamaClient(host=host)
        response = client.chat(
            model=model_config["model_id"],
            messages=[{"role": "user", "content": prompt}],
        )
        return response.message.content or ""
    else:
        raise ValueError(f"Unknown provider: {provider}")


def run_model(state: dict) -> None:
    """Run each prompt against the target LLM and collect responses."""
    prompts = state["prompts"]
    model_config = state["config"]["model"]

    responses = []
    for prompt in prompts:
        result = call_target_llm(prompt, model_config)
        responses.append({"prompt": prompt, "response": result})

    state["responses"] = responses
