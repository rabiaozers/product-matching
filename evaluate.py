"""Pick a decision threshold on the validation set, then measure once on the test set."""
import json
import os

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from data import load_catalogs, load_pairs
from matchers import EmbeddingMatcher, HybridMatcher, TfidfMatcher

MODEL_DIR = "models/minilm-abtbuy"
CONFIG_PATH = "models/hybrid_config.json"


def confusion(labels, predictions):
    """Count the four outcomes: true/false positives and negatives."""
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    tp = int(((predictions == 1) & (labels == 1)).sum())   # said "same", really same
    fp = int(((predictions == 1) & (labels == 0)).sum())   # said "same", really different
    fn = int(((predictions == 0) & (labels == 1)).sum())   # said "different", really same
    tn = int(((predictions == 0) & (labels == 0)).sum())   # said "different", really different
    return tp, fp, fn, tn


def precision_recall_f1(labels, predictions):
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    return precision, recall, f1


def best_threshold(labels, scores):
    """Try many thresholds and keep the one with the highest F1."""
    best_t, best_f1 = 0.0, -1.0
    for t in np.arange(0.0, 1.0, 0.01):
        predictions = (scores > t).astype(int)
        _, _, f1 = precision_recall_f1(labels, predictions)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t


if __name__ == "__main__":
    table_a, table_b = load_catalogs()
    valid = load_pairs("valid", table_a, table_b)
    test = load_pairs("test", table_a, table_b)
    all_names = list(table_a["name"]) + list(table_b["name"])

    matchers = [("TF-IDF", TfidfMatcher()), ("Embedding", EmbeddingMatcher())]
    if os.path.isdir("models/minilm-abtbuy"):
        matchers.append(("Fine-tuned", EmbeddingMatcher("models/minilm-abtbuy")))
        weight = 0.5
        if os.path.exists(CONFIG_PATH):   # written by tune_weight.py
            with open(CONFIG_PATH) as f:
                weight = json.load(f)["weight"]
        matchers.append((f"Hybrid (w={weight})", HybridMatcher(TfidfMatcher(), EmbeddingMatcher(MODEL_DIR), weight=weight)))

    for name, matcher in matchers:
        matcher.fit(all_names)
        valid_scores = matcher.score(valid["left_name"], valid["right_name"])
        test_scores = matcher.score(test["left_name"], test["right_name"])

        t = best_threshold(valid["label"], valid_scores)
        predictions = (test_scores > t).astype(int)
        tp, fp, fn, tn = confusion(test["label"], predictions)
        p, r, f1 = precision_recall_f1(test["label"], predictions)
        print(f"{name:17s} threshold={t:.2f}  TP={tp} FP={fp} FN={fn}  "
              f"precision={p:.3f} recall={r:.3f} F1={f1:.3f}")

    lazy = np.zeros(len(test), dtype=int)   # always says "different"
    p, r, f1 = precision_recall_f1(test["label"], lazy)
    print(f"{'lazy baseline':17s} always 'different':  precision={p:.3f} recall={r:.3f} F1={f1:.3f}")
