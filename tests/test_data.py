"""Tests for data module utilities."""

import pytest

from agenticvlm.data.label_definitions import (
    INDEX_TO_LABEL,
    LABEL_DEFINITIONS,
    LABEL_TO_INDEX,
    ROUTER_LABELS,
)


class TestLabelDefinitions:
    def test_nine_labels(self):
        assert len(ROUTER_LABELS) == 9

    def test_all_labels_have_definitions(self):
        for label in ROUTER_LABELS:
            assert label in LABEL_DEFINITIONS
            assert len(LABEL_DEFINITIONS[label]) > 0

    def test_index_mapping_roundtrip(self):
        for label in ROUTER_LABELS:
            idx = LABEL_TO_INDEX[label]
            assert INDEX_TO_LABEL[idx] == label

    def test_expected_labels_present(self):
        expected = {
            "figure/diagram", "Yes/No", "table/list", "layout",
            "Image/Photo", "handwritten", "free_text", "form", "others",
        }
        assert set(ROUTER_LABELS) == expected
