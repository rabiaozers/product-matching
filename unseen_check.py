"""Leakage check: do the matchers work as well on products never seen during training?"""
from data import load_catalogs, load_pairs
from evaluate import best_threshold, precision_recall_f1
from matchers import EmbeddingMatcher, TfidfMatcher

table_a, table_b = load_catalogs()
train = load_pairs("train", table_a, table_b)
valid = load_pairs("valid", table_a, table_b)
test = load_pairs("test", table_a, table_b)
all_names = list(table_a["name"]) + list(table_b["name"])

seen = test["ltable_id"].isin(train["ltable_id"]) & test["rtable_id"].isin(train["rtable_id"])

print(f"seen pairs:   {seen.sum():5d}  matches: {test[seen]['label'].sum()}")
print(f"unseen pairs: {(~seen).sum():5d}  matches: {test[~seen]['label'].sum()}")

matchers = [("TF-IDF", TfidfMatcher()), ("Fine-tuned", EmbeddingMatcher("models/minilm-abtbuy"))]
for name, matcher in matchers:
    matcher.fit(all_names)
    t = best_threshold(valid["label"], matcher.score(valid["left_name"], valid["right_name"]))
    predictions = (matcher.score(test["left_name"], test["right_name"]) > t).astype(int)
    _, _, f1_seen = precision_recall_f1(test["label"][seen], predictions[seen])
    _, _, f1_unseen = precision_recall_f1(test["label"][~seen], predictions[~seen])
    print(f"{name:11s} F1 seen={f1_seen:.3f}  F1 unseen={f1_unseen:.3f}")
