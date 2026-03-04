"""Tests for the question router utilities."""

import pytest

from agenticvlm.pipeline.question_router import (
    get_model_backend,
    resolve_agent_type,
)


class TestResolveAgentType:
    @pytest.mark.parametrize(
        "label,expected",
        [
            ("figure/diagram", "figure/diagram"),
            ("Figure/Diagram", "figure/diagram"),
            ("figure", "figure/diagram"),
            ("diagram", "figure/diagram"),
            ("image/photo", "image/photo"),
            ("image", "image/photo"),
            ("photo", "image/photo"),
            ("form", "form"),
            ("free_text", "free_text"),
            ("freetext", "free_text"),
            ("handwritten", "handwritten"),
            ("yes/no", "Yes/No"),
            ("Yes/No", "Yes/No"),
            ("layout", "layout"),
            ("table/list", "table/list"),
            ("table", "table/list"),
            ("list", "table/list"),
            ("others", "others"),
            ("unknown_category", "others"),
        ],
    )
    def test_label_normalization(self, label, expected):
        assert resolve_agent_type(label) == expected


class TestGetModelBackend:
    def test_internvl_labels(self):
        assert get_model_backend("figure/diagram") == "internvl3"
        assert get_model_backend("image/photo") == "internvl3"

    def test_ocr_labels(self):
        assert get_model_backend("handwritten") == "qwen2_ocr"

    def test_qwen_labels(self):
        assert get_model_backend("form") == "qwen25vl"
        assert get_model_backend("free_text") == "qwen25vl"
        assert get_model_backend("Yes/No") == "qwen25vl"
        assert get_model_backend("layout") == "qwen25vl"
        assert get_model_backend("table/list") == "qwen25vl"

    def test_others_defaults_to_qwen(self):
        assert get_model_backend("others") == "qwen25vl"
