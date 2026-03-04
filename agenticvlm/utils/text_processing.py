"""Text processing utilities."""

from __future__ import annotations

import re
from typing import Optional


def clean_thought(thinking: str) -> str:
    """Extract the thinking content from ``<think>...</think>`` tags.

    Args:
        thinking: Raw model output potentially containing think tags.

    Returns:
        Extracted thinking text, or empty string if no tags found.
    """
    match = re.search(r"<think>(.*?)</think>", thinking, re.DOTALL)
    if match:
        return match.group(1)

    think_start = thinking.find("<think>")
    if think_start != -1:
        return thinking[think_start + len("<think>") :]

    return ""


def is_number(s: str) -> bool:
    """Check whether a string represents a valid number."""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def extract_boxed_answer(output_text: str) -> Optional[str]:
    """Extract the answer from GLM-4V box tags or ``<answer>`` tags.

    Looks for ``<|begin_of_box|>...<|end_of_box|>`` first, then falls back
    to ``<answer>...</answer>`` tags.

    Args:
        output_text: Raw model output text.

    Returns:
        Extracted answer string, or ``None`` if no tagged answer found.
    """
    match = re.search(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", output_text)
    if match:
        return match.group(1).strip()

    answer_match = re.search(r"<answer>(.*?)</answer>", output_text, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()

    return None


def mask_thinking(output_text: str, threshold: int = 4) -> str:
    """Mask answer occurrences in the thinker output (Section 3.2).

    Args:
        output_text: Full model output containing both reasoning and answer.
        threshold: τ — minimum occurrence count before masking all instances.

    Returns:
        Output text with answer occurrences selectively replaced by
        ``[RESPONSE]``.
    """
    match = re.search(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", output_text)
    if match is None:
        match = re.search(r"<answer>(.*?)</answer>", output_text, re.DOTALL)

    if not match:
        return output_text

    boxed_text = match.group(1).strip()
    if not boxed_text:
        return output_text

    boxed_text_lower = boxed_text.lower()
    output_text_lower = output_text.lower()

    all_matches: list[tuple[int, int]] = []

    for m in re.finditer(re.escape(boxed_text_lower), output_text_lower):
        start, end = m.span()
        before = output_text_lower[start - 1] if start > 0 else ""
        after = output_text_lower[end] if end < len(output_text_lower) else ""

        embedded = before.isalnum() or after.isalnum()
        if embedded:
            context_start = max(0, start - 20)
            context_end = min(len(output_text_lower), end + 20)
            word_match = re.search(
                r"\b\S*" + re.escape(boxed_text_lower) + r"\S*\b",
                output_text_lower[context_start:context_end],
            )
            if word_match:
                word_start = context_start + word_match.start()
                word_end = context_start + word_match.end()
                all_matches.append((word_start, word_end))
        else:
            all_matches.append((start, end))

    if not all_matches:
        return output_text

    if len(all_matches) > threshold and not is_number(boxed_text):
        spans_to_mask = all_matches
    else:
        num_to_mask = len(all_matches) // 2
        spans_to_mask = all_matches[-num_to_mask:]

    masked_output = output_text
    offset = 0
    for start, end in spans_to_mask:
        masked_output = (
            masked_output[: start + offset] + "[RESPONSE]" + masked_output[end + offset :]
        )
        offset += len("[RESPONSE]") - (end - start)

    return masked_output


def clean_generated_text(text: str) -> str:
    """Remove unwanted special tokens from generated text."""
    unwanted_tokens = [
        "<|im_end|>",
        "<|endoftext|>",
        "</s>",
        "<eos>",
        "<|end|>",
    ]
    for token in unwanted_tokens:
        text = text.replace(token, "")
    return text.strip()
