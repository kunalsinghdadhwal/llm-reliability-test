"""generate_prompts -- ask Claude to create domain-specific test prompts."""

import re

from src.wrappers.bedrock import call_llm


def parse_prompts(response: str) -> list[str]:
    """Parse LLM response into individual prompt strings.

    Handles numbered lists, bullet points, and plain newline-separated text.
    """
    lines = response.strip().splitlines()
    prompts = []
    for line in lines:
        # Strip numbered prefixes like "1. ", "2) ", etc.
        cleaned = re.sub(r"^\s*\d+[\.\)]\s*", "", line)
        # Strip bullet prefixes like "- " or "* "
        cleaned = re.sub(r"^\s*[-\*]\s+", "", cleaned)
        cleaned = cleaned.strip()
        if cleaned:
            prompts.append(cleaned)
    return prompts


def generate_prompts(state: dict) -> None:
    """Generate test prompts for the given use case via Claude."""
    config = state["config"]
    use_case = config["use_case"]
    num_prompts = config["evaluation"]["num_prompts"]
    categories = config["evaluation"]["prompt_categories"]

    categories_str = ", ".join(categories)

    system_prompt = (
        "You are a test prompt generator for LLM reliability evaluation. "
        "Generate prompts that are likely to surface hallucinations or "
        "inaccurate information. Return ONLY a numbered list of prompts, "
        "one per line. No extra commentary."
    )

    prompt = (
        f"Generate {num_prompts} test prompts for the following use case:\n"
        f"Use case: {use_case}\n"
        f"Categories to cover: {categories_str}\n\n"
        f"Return exactly {num_prompts} prompts as a numbered list."
    )

    response = call_llm(prompt, system=system_prompt)
    prompts = parse_prompts(response)

    # Truncate to num_prompts if Claude returned more
    state["prompts"] = prompts[:num_prompts]
