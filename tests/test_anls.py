"""Tests for the ANLS metric calculator."""

import pytest

from agenticvlm.evaluation.anls import ANLSCalculator


@pytest.fixture
def calc():
    return ANLSCalculator(threshold=0.5)


class TestANLSScore:
    """Unit tests for single-pair ANLS scoring."""

    def test_identical_strings(self, calc):
        assert calc.anls_score("hello", "hello") == 1.0

    def test_identical_case_insensitive(self, calc):
        assert calc.anls_score("Hello", "hello") == 1.0

    def test_both_empty(self, calc):
        assert calc.anls_score("", "") == 1.0

    def test_one_empty(self, calc):
        assert calc.anls_score("", "hello") == 0.0
        assert calc.anls_score("hello", "") == 0.0

    def test_similar_strings(self, calc):
        # "hell" vs "hello" — distance=1, max_len=5, NLD=0.2 < 0.5
        score = calc.anls_score("hell", "hello")
        assert score == pytest.approx(0.8, abs=0.01)

    def test_dissimilar_strings(self, calc):
        # Completely different — NLD >= 0.5
        score = calc.anls_score("abc", "xyz")
        assert score == 0.0

    def test_whitespace_handling(self, calc):
        assert calc.anls_score("  hello  ", "hello") == 1.0

    def test_numeric_answers(self, calc):
        assert calc.anls_score("42", "42") == 1.0
        assert calc.anls_score("42", "43") == pytest.approx(0.5, abs=0.01)


class TestANLSMulti:
    """Tests for multi-ground-truth ANLS."""

    def test_best_match_chosen(self, calc):
        score = calc.anls_score_multi("yes", ["yes", "no", "maybe"])
        assert score == 1.0

    def test_partial_match(self, calc):
        score = calc.anls_score_multi("ye", ["yes", "no"])
        assert score > 0.0

    def test_no_ground_truths(self, calc):
        assert calc.anls_score_multi("hello", []) == 0.0


class TestBatchANLS:
    """Tests for batch scoring."""

    def test_perfect_batch(self, calc):
        preds = ["hello", "world"]
        gts = ["hello", "world"]
        assert calc.batch_anls(preds, gts) == 1.0

    def test_mixed_batch(self, calc):
        preds = ["hello", "xyz"]
        gts = ["hello", "abc"]
        avg = calc.batch_anls(preds, gts)
        assert 0.0 < avg < 1.0

    def test_empty_batch(self, calc):
        assert calc.batch_anls([], []) == 0.0

    def test_length_mismatch_raises(self, calc):
        with pytest.raises(ValueError):
            calc.batch_anls(["a"], ["b", "c"])

    def test_multi_gt_batch(self, calc):
        preds = ["yes"]
        gts = [["yes", "yeah", "yep"]]
        assert calc.batch_anls(preds, gts) == 1.0
