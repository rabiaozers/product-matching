import numpy as np
import pytest

from matchers import HybridMatcher, TfidfMatcher

NAMES = [
    "sony bravia dav-is50 home theater system",
    "delonghi dc50b drip coffee maker black",
    "panasonic nnsd797s stainless steel microwave",
    "logitech v220 cordless optical mouse 910-000153",
]


def test_identical_titles_score_one():
    matcher = TfidfMatcher().fit(NAMES)
    score = matcher.score([NAMES[0]], [NAMES[0]])[0]
    assert score == pytest.approx(1.0)


def test_similar_titles_beat_unrelated_titles():
    matcher = TfidfMatcher().fit(NAMES)
    query = "sony bravia dav-is50 / b home theater"
    similar, unrelated = matcher.score([query, query], [NAMES[0], NAMES[1]])
    assert similar > unrelated


class Constant:
    """A fake matcher that always returns the same score."""

    def __init__(self, value):
        self.value = value

    def fit(self, names):
        return self

    def score(self, left, right):
        return np.full(len(left), self.value)


def test_hybrid_is_a_weighted_average():
    hybrid = HybridMatcher(Constant(1.0), Constant(0.0), weight=0.6)
    assert hybrid.score(["a"], ["b"])[0] == pytest.approx(0.6)
