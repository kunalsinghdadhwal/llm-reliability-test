"""extract_claims -- decompose LLM responses into atomic factual claims."""

import re

from src.wrappers.bedrock import call_llm

SYSTEM_PROMPT = (
    "You are a claim extraction assistant. Given a text, extract ONLY "
    "atomic factual claims. Each claim should state exactly one fact. "
    "Do NOT include opinions, hedges, or meta-commentary. "
    "Return a numbered list (1. ..., 2. ..., etc.). "
    "If the text contains no factual claims, return exactly: NO CLAIMS"
)


def parse_claims(response: str) -> list[str]:
    """Parse numbered list response into individual claim strings."""
    text = response.strip()
    if text.upper() == "NO CLAIMS":
        return []

    claims = []
    for line in text.splitlines():
        # Strip numbered prefixes like "1. ", "2) "
        cleaned = re.sub(r"^\s*\d+[\.\)]\s*", "", line)
        cleaned = cleaned.strip()
        if cleaned:
            claims.append(cleaned)
    return claims


def extract_claims(state: dict) -> None:
    """Extract atomic factual claims from each LLM response."""
    responses = state["responses"]
    all_claims = []

    for entry in responses:
        prompt_text = entry["prompt"]
        response_text = entry["response"]

        prompt = f"Extract all atomic factual claims from the following text:\n\n{response_text}"

        llm_response = call_llm(prompt, system=SYSTEM_PROMPT)
        claims = parse_claims(llm_response)

        for claim_text in claims:
            all_claims.append(
                {
                    "text": claim_text,
                    "source_prompt": prompt_text,
                    "source_response": response_text,
                }
            )

    state["claims"] = all_claims
