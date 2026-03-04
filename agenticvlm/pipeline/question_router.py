"""Question router — maps router labels to specialist agent types.

Supports multi-label activation vectors from the Turbo DFS router.
Each label maps to a canonical agent type and its model backend.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Type

from agenticvlm.agents.base import BaseSpecialistAgent
from agenticvlm.agents.internvl_agents import FigureDiagramAgent, ImagePhotoAgent
from agenticvlm.agents.qwen_agents import (
    FormAgent,
    FreeTextAgent,
    HandwrittenAgent,
    LayoutAgent,
    TableListAgent,
    YesNoAgent,
)

logger = logging.getLogger(__name__)

LABEL_TO_AGENT: Dict[str, Type[BaseSpecialistAgent]] = {
    "figure/diagram": FigureDiagramAgent,
    "image/photo": ImagePhotoAgent,
    "form": FormAgent,
    "free_text": FreeTextAgent,
    "handwritten": HandwrittenAgent,
    "Yes/No": YesNoAgent,
    "layout": LayoutAgent,
    "table/list": TableListAgent,
}

INTERNVL_LABELS = {"figure/diagram", "image/photo"}

OCR_LABELS = {"handwritten"}

QWEN_LABELS = {"form", "free_text", "Yes/No", "layout", "table/list"}

AGENT_PRIORITY: Dict[str, int] = {
    "handwritten": 0,
    "layout": 1,
    "form": 2,
    "table/list": 3,
    "free_text": 4,
    "Yes/No": 5,
    "figure/diagram": 6,
    "image/photo": 7,
    "others": 8,
}


def resolve_agent_type(label: str) -> str:
    """Normalize a router label to a canonical form.

    Args:
        label: Raw label from the router model.

    Returns:
        Canonical label string.
    """
    label_lower = label.strip().lower()
    mapping = {
        "figure/diagram": "figure/diagram",
        "figure": "figure/diagram",
        "diagram": "figure/diagram",
        "image/photo": "image/photo",
        "image": "image/photo",
        "photo": "image/photo",
        "form": "form",
        "free_text": "free_text",
        "freetext": "free_text",
        "free text": "free_text",
        "handwritten": "handwritten",
        "yes/no": "Yes/No",
        "yesno": "Yes/No",
        "yes_no": "Yes/No",
        "layout": "layout",
        "table/list": "table/list",
        "table": "table/list",
        "list": "table/list",
        "others": "others",
    }
    return mapping.get(label_lower, "others")


def resolve_multi_labels(labels: List[str]) -> List[str]:
    """Normalize multiple router labels to canonical forms.

    Deduplicates and filters out ``'others'`` when real labels exist.

    Args:
        labels: Raw labels from the multi-label router.

    Returns:
        Deduplicated, canonical label list.
    """
    canonical = []
    seen = set()
    for label in labels:
        c = resolve_agent_type(label)
        if c not in seen:
            seen.add(c)
            canonical.append(c)
    if len(canonical) > 1 and "others" in canonical:
        canonical = [l for l in canonical if l != "others"]
    return canonical if canonical else ["free_text"]


def sort_by_priority(labels: List[str]) -> List[str]:
    """Sort labels by orchestrator priority (lower priority value = first).

    Args:
        labels: Canonical label list.

    Returns:
        Labels sorted by priority.
    """
    return sorted(labels, key=lambda l: AGENT_PRIORITY.get(l, 99))


def get_model_backend(label: str) -> str:
    """Determine which model backend a label requires.

    Returns:
        One of ``'internvl3'``, ``'qwen2_ocr'``, or ``'qwen25vl'``.
    """
    canonical = resolve_agent_type(label)
    if canonical in INTERNVL_LABELS:
        return "internvl3"
    if canonical in OCR_LABELS:
        return "qwen2_ocr"
    return "qwen25vl"
