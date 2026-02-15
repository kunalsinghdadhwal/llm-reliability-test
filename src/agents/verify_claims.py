"""verify_claims -- label each claim against retrieved evidence."""

import re

from src.wrappers.bedrock import call_llm

SYSTEM_PROMPT = (
    "You are an evidence verification assistant. Compare the given claim "
    "against the provided evidence documents ONLY. Do not use any outside "
    "knowledge.\n\n"
    "Respond with exactly one of these labels:\n"
    "  supported -- the evidence directly confirms the claim\n"
    "  weakly_supported -- the evidence partially confirms or is ambiguous\n"
    "  unsupported -- no evidence supports the claim, or it is contradicted\n\n"
    "Format your response exactly as:\n"
    "LABEL: <label>\n"
    "JUSTIFICATION: <one sentence explanation>"
)

VALID_LABELS = {"supported", "weakly_supported", "unsupported"}


def parse_verdict(response: str) -> tuple[str, str]:
    """Extract label and justification from Claude's response.

    Returns (label, justification). Defaults to unsupported on parse failure.
    """
    label_match = re.search(r"LABEL:\s*(.+)", response, re.IGNORECASE)
    justification_match = re.search(r"JUSTIFICATION:\s*(.+)", response, re.IGNORECASE)

    if not label_match:
        return ("unsupported", "Could not parse verification response.")

    label = label_match.group(1).strip().lower()
    if label not in VALID_LABELS:
        return ("unsupported", "Could not parse verification response.")

    justification = (
        justification_match.group(1).strip()
        if justification_match
        else "Could not parse verification response."
    )

    return (label, justification)


_CONFIDENCE_MAP = {
    "supported": 1.0,
    "weakly_supported": 0.5,
    "unsupported": 0.0,
}


def verify_claims(state: dict) -> None:
    """Verify each claim against its retrieved evidence documents."""
    evidence_entries = state["evidence"]
    verdicts = []

    for entry in evidence_entries:
        claim = entry["claim"]
        documents = entry["documents"]

        if not documents:
            verdicts.append(
                {
                    "claim": claim["text"],
                    "verdict": "unsupported",
                    "evidence_snippet": "",
                    "confidence": 0.0,
                }
            )
            continue

        evidence_text = "\n\n".join(doc["content"] for doc in documents)
        prompt = f"Claim: {claim['text']}\n\nEvidence:\n{evidence_text}"

        response = call_llm(prompt, system=SYSTEM_PROMPT)
        label, _justification = parse_verdict(response)

        snippet = documents[0]["content"][:200] if documents else ""

        verdicts.append(
            {
                "claim": claim["text"],
                "verdict": label,
                "evidence_snippet": snippet,
                "confidence": _CONFIDENCE_MAP.get(label, 0.0),
            }
        )

    state["verdicts"] = verdicts
