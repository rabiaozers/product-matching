"""Fine-tune the sentence embedding model on the labelled train pairs.

Same training loop as Project 3: forward -> loss -> zero_grad -> backward -> step.
The loss pushes the cosine similarity of each pair towards its label:
1 for the same product, 0 for different products (like CosineSimilarityLoss).
"""
import random

import pandas as pd

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from data import DATA, load_catalogs, load_pairs

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EASY_NEGATIVES_PER_PAIR = 0   # 0 = original training data (the model the API uses); 1 = the easy-negatives experiment
OUT_DIR = "models/minilm-abtbuy" if EASY_NEGATIVES_PER_PAIR == 0 else "models/minilm-abtbuy-neg"
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5


def batches(pairs, size):
    """Yield the pairs in random order, `size` rows at a time."""
    order = list(range(len(pairs)))
    random.shuffle(order)
    for i in range(0, len(order), size):
        yield pairs.iloc[order[i:i + size]]


def add_easy_negatives(train, table_a, table_b, forbidden, n, seed=0):
    """Add n random (Abt, Buy) pairs labelled 0: products that are almost surely unrelated.

    The original train pairs are all "similar but different" (hard negatives), so the model
    never saw obviously different products. `forbidden` holds every pair that appears in
    train / valid / test, so we never relabel a known pair or peek at valid/test.
    """
    rng = random.Random(seed)
    a_ids = list(table_a["id"])
    b_ids = list(table_b["id"])
    new = set()
    while len(new) < n:
        l, r = rng.choice(a_ids), rng.choice(b_ids)
        if (l, r) not in forbidden and (l, r) not in new:
            new.add((l, r))


    extra = pd.DataFrame(sorted(new), columns=["ltable_id", "rtable_id"])
    extra["label"] = 0
    left = table_a[["id", "name"]].rename(columns={"id": "ltable_id", "name": "left_name"})
    right = table_b[["id", "name"]].rename(columns={"id": "rtable_id", "name": "right_name"})
    extra = extra.merge(left, on="ltable_id").merge(right, on="rtable_id")
    return pd.concat([train, extra], ignore_index=True)


def embed(model, texts):
    """Like model.encode, but keeps the computation graph so gradients can flow."""
    features = model.preprocess(list(texts))
    return model(features)["sentence_embedding"]


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)

    table_a, table_b = load_catalogs()
    train = load_pairs("train", table_a, table_b)
    if EASY_NEGATIVES_PER_PAIR > 0:
        forbidden = set()
        for split in ["train", "valid", "test"]:
            pairs = pd.read_csv(DATA + split + ".csv")
            forbidden |= set(zip(pairs["ltable_id"], pairs["rtable_id"]))
        n_new = EASY_NEGATIVES_PER_PAIR * len(train)
        train = add_easy_negatives(train, table_a, table_b, forbidden, n_new)
        print(f"train pairs: {len(train)} (added {n_new} easy negatives)")

    model = SentenceTransformer(BASE_MODEL)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()

    for epoch in range(EPOCHS):
        total = 0.0
        for batch in batches(train, BATCH_SIZE):
            left = embed(model, batch["left_name"])
            right = embed(model, batch["right_name"])
            similarity = F.cosine_similarity(left, right)
            target = torch.tensor(batch["label"].values, dtype=torch.float32)


            loss = F.mse_loss(similarity, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    

            total += loss.item() * len(batch)
        print(f"epoch {epoch + 1}: mean loss {total / len(train):.4f}")

    model.save(OUT_DIR)
    print("saved to", OUT_DIR)
