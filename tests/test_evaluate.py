import numpy as np

from evaluate import best_threshold, confusion, precision_recall_f1


def test_confusion_counts_the_four_outcomes():
    labels = [1, 0, 1, 0, 0]
    predictions = [1, 1, 0, 0, 0]
    assert confusion(labels, predictions) == (1, 1, 1, 2)   # TP, FP, FN, TN


def test_precision_recall_f1():
    labels = [1, 0, 1, 0, 0]
    predictions = [1, 1, 0, 0, 0]
    p, r, f1 = precision_recall_f1(labels, predictions)
    assert p == 0.5 and r == 0.5 and f1 == 0.5


def test_always_different_has_zero_f1():
    labels = [1, 0, 0, 0]
    p, r, f1 = precision_recall_f1(labels, [0, 0, 0, 0])
    assert (p, r, f1) == (0.0, 0.0, 0.0)


def test_best_threshold_separates_clean_scores():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.3, 0.7, 0.9])
    t = best_threshold(labels, scores)
    assert 0.3 <= t < 0.7
    assert ((scores > t).astype(int) == labels).all()
