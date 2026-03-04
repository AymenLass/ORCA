"""Router label definitions and data categories.

The AgenticVLM router classifies each (question, document-image) pair
into one of nine document-type categories.
"""

from __future__ import annotations

# Canonical label list — order matters for index mapping
ROUTER_LABELS: list[str] = [
    "figure/diagram",
    "Yes/No",
    "table/list",
    "layout",
    "Image/Photo",
    "handwritten",
    "free_text",
    "form",
    "others",
]

LABEL_DEFINITIONS: dict[str, str] = {
    "figure/diagram": (
        "Questions about technical figures, diagrams, charts, plots, "
        "and scientific illustrations."
    ),
    "Yes/No": (
        "Binary decision questions that can be answered with Yes or No."
    ),
    "table/list": (
        "Questions requiring extraction from tabular data or structured lists."
    ),
    "layout": (
        "Questions about document structure, positioning, and spatial organization."
    ),
    "Image/Photo": (
        "Questions about photographic content, real-world scenes, and visual elements."
    ),
    "handwritten": (
        "Questions involving handwritten text or low-quality document images "
        "requiring OCR."
    ),
    "free_text": (
        "Questions about unstructured running text, paragraphs, and continuous prose."
    ),
    "form": (
        "Questions about information in forms, applications, and structured "
        "input documents."
    ),
    "others": (
        "Questions that do not fit any specific category above."
    ),
}

LABEL_TO_INDEX: dict[str, int] = {label: idx for idx, label in enumerate(ROUTER_LABELS)}
INDEX_TO_LABEL: dict[int, str] = {idx: label for idx, label in enumerate(ROUTER_LABELS)}
