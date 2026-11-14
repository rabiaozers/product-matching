import numpy as np
import pytest

from matchers import TfidfMatcher
from search import CatalogSearch, hybrid_vectors, tfidf_vectors

CATALOG = [
    "sony bravia dav-is50 home theater system",
    "delonghi dc50b drip coffee maker black",
    "panasonic nnsd797s stainless steel microwave",
    "logitech v220 cordless optical mouse 910-000153",
]


def test_search_finds_the_right_product_first():
    tfidf = TfidfMatcher().fit(CATALOG)
    catalog = CatalogSearch(tfidf_vectors(tfidf.vectorizer), CATALOG)
    row, score = catalog.search("logitech v220 cordless mouse 910000153", k=2)[0]
    assert row == 3
    assert 0.0 < score <= 1.0 + 1e-6


def test_hybrid_vectors_give_the_weighted_score():
    # Dot product of glued vectors = w * first similarity + (1 - w) * second similarity.
    tfidf = TfidfMatcher().fit(CATALOG)
    first = tfidf_vectors(tfidf.vectorizer)
    second = tfidf_vectors(TfidfMatcher().fit(CATALOG + ["extra words"]).vectorizer)
    w = 0.6
    glued = hybrid_vectors(first, second, w)
    a, b = [CATALOG[0]], [CATALOG[1]]

    def dot(vectorize):
        return float((vectorize(a) * vectorize(b)).sum())

    expected = w * dot(first) + (1 - w) * dot(second)
    assert dot(glued) == pytest.approx(expected)
