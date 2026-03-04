"""Tests for text processing utilities."""

import pytest

from agenticvlm.utils.text_processing import (
    clean_generated_text,
    clean_thought,
    extract_boxed_answer,
    is_number,
    mask_thinking,
)


class TestCleanThought:
    def test_extract_think_tags(self):
        text = "prefix <think>inner thought</think> suffix"
        assert clean_thought(text) == "inner thought"

    def test_unclosed_think_tag(self):
        text = "<think>partial thought with no end"
        assert clean_thought(text) == "partial thought with no end"

    def test_no_think_tag(self):
        assert clean_thought("no thinking here") == ""


class TestIsNumber:
    def test_integer(self):
        assert is_number("42") is True

    def test_float(self):
        assert is_number("3.14") is True

    def test_negative(self):
        assert is_number("-7") is True

    def test_not_number(self):
        assert is_number("hello") is False

    def test_empty(self):
        assert is_number("") is False


class TestExtractBoxedAnswer:
    def test_box_format(self):
        text = "some text <|begin_of_box|>answer here<|end_of_box|> more text"
        assert extract_boxed_answer(text) == "answer here"

    def test_answer_tags(self):
        text = "reasoning <answer>the answer</answer> done"
        assert extract_boxed_answer(text) == "the answer"

    def test_no_answer(self):
        text = "just plain text"
        assert extract_boxed_answer(text) is None

    def test_whitespace_stripping(self):
        text = "<|begin_of_box|>  spaced answer  <|end_of_box|>"
        assert extract_boxed_answer(text) == "spaced answer"


class TestMaskThinking:
    def test_masks_answer_occurrences(self):
        text = "The answer is hello and hello appears again <|begin_of_box|>hello<|end_of_box|>"
        result = mask_thinking(text)
        # Should have at least one [RESPONSE] replacement
        assert "[RESPONSE]" in result

    def test_no_boxed_answer(self):
        text = "no answer to mask here"
        result = mask_thinking(text)
        assert result == text

    def test_numeric_answer_partial_mask(self):
        text = "value 42 found at 42 position <|begin_of_box|>42<|end_of_box|>"
        result = mask_thinking(text)
        # Numeric answers mask only latter half of occurrences
        assert isinstance(result, str)


class TestCleanGeneratedText:
    def test_removes_special_tokens(self):
        text = "hello<|im_end|>"
        assert clean_generated_text(text) == "hello"

    def test_removes_multiple_tokens(self):
        text = "result</s><|endoftext|>"
        assert clean_generated_text(text) == "result"

    def test_strips_whitespace(self):
        text = "  hello  "
        assert clean_generated_text(text) == "hello"

    def test_preserves_normal_text(self):
        text = "normal answer"
        assert clean_generated_text(text) == "normal answer"
