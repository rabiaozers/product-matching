"""Find the most similar catalog products for one title with FAISS.

Three ways to turn titles into vectors (all normalised to length 1):
  embedding  - the fine-tuned model (384 numbers)
  tfidf      - character 3-grams (7,578 numbers)
  hybrid     - both glued together:  [ sqrt(w) * tfidf , sqrt(1-w) * embedding ]
               the dot product of two such vectors = w * tfidf_cosine + (1-w) * embedding_cosine,
               which is exactly the hybrid score used by /match.
"""
import json

import faiss
import numpy as np


def embedding_vectors(model):
    """Return a function: list of titles -> fine-tuned embedding vectors."""
    def vectorize(texts):
        return model.encode(list(texts), normalize_embeddings=True, batch_size=64)
    return vectorize


def tfidf_vectors(vectorizer):
    """Return a function: list of titles -> TF-IDF vectors (made dense for FAISS)."""
    def vectorize(texts):
        return vectorizer.transform(list(texts)).toarray()
    return vectorize


def hybrid_vectors(tfidf_vectorize, embedding_vectorize, weight):
    """Return a function: list of titles -> [sqrt(w) * tfidf, sqrt(1-w) * embedding]."""
    def vectorize(texts):
        a = tfidf_vectorize(texts)
        b = embedding_vectorize(texts)
        return np.hstack([np.sqrt(weight) * a, np.sqrt(1 - weight) * b])
    return vectorize


class CatalogSearch:
    """Keeps every catalog product as a vector and finds the nearest ones to a query."""

    def __init__(self, vectorize, names):
        self.vectorize = vectorize
        vectors = vectorize(names).astype("float32")
        # Inner product (IP) of length-1 vectors = cosine similarity.
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query, k=5):
        """Return the k most similar catalog products as (row number, similarity) pairs."""
        query_vector = self.vectorize([query]).astype("float32")
        scores, rows = self.index.search(query_vector, k)
        return [(int(r), float(s)) for r, s in zip(rows[0], scores[0])]


def recall_at_k(catalog, matches, catalog_ids, ks=(1, 5, 10, 20)):
    """For each true match, is the right product among the k nearest? Returns {k: recall}."""
    found = {k: 0 for k in ks}
    for _, pair in matches.iterrows():
        rows = [r for r, _ in catalog.search(pair["left_name"], max(ks))]
        for k in ks:
            if pair["rtable_id"] in set(catalog_ids[rows[:k]]):
                found[k] += 1
    return {k: found[k] / len(matches) for k in ks}


if __name__ == "__main__":
    from data import load_catalogs, load_pairs
    from matchers import EmbeddingMatcher, TfidfMatcher

    with open("models/hybrid_config.json") as f:
        config = json.load(f)

    table_a, table_b = load_catalogs()
    test = load_pairs("test", table_a, table_b)
    matches = test[test["label"] == 1]

    tfidf = TfidfMatcher().fit(list(table_a["name"]) + list(table_b["name"]))
    model = EmbeddingMatcher(config["model_dir"]).model
    methods = {
        "embedding": embedding_vectors(model),
        "tfidf": tfidf_vectors(tfidf.vectorizer),
        f"hybrid (w={config['weight']})": hybrid_vectors(tfidf_vectors(tfidf.vectorizer),
                                                         embedding_vectors(model), config["weight"]),
    }

    query = matches.iloc[0]
    print("query:     ", query["left_name"])
    print("true match:", query["right_name"])

    for name, vectorize in methods.items():
        catalog = CatalogSearch(vectorize, table_b["name"])
        print(f"\n=== {name} ===")
        for row, score in catalog.search(query["left_name"], k=3):
            mark = "  <- true match" if table_b["id"][row] == query["rtable_id"] else ""
            print(f"  {score:.3f}  {table_b['name'][row]}{mark}")
        recalls = recall_at_k(catalog, matches, table_b["id"].values)
        print("  " + "   ".join(f"recall@{k}={v:.3f}" for k, v in recalls.items()))
