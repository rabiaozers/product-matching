"""Streamlit demo of the product matcher: same models, weight and threshold as the FastAPI service."""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))   # use matchers.py and search.py from the repository root

from matchers import EmbeddingMatcher, HybridMatcher, TfidfMatcher  # noqa: E402
from search import CatalogSearch, embedding_vectors, hybrid_vectors, tfidf_vectors  # noqa: E402

with open(HERE / "hybrid_config.json") as f:
    config = json.load(f)
THRESHOLD = config["threshold"]
WEIGHT = config["weight"]


@st.cache_resource(show_spinner="Loading the models (about a minute on the first visit)...")
def load():
    """Build the matcher and the FAISS index once; Streamlit keeps them between visits."""
    table_a = pd.read_csv(HERE / "catalogs" / "tableA.csv")
    table_b = pd.read_csv(HERE / "catalogs" / "tableB.csv")
    matcher = HybridMatcher(TfidfMatcher(), EmbeddingMatcher(config["model"]), weight=WEIGHT)
    matcher.fit(list(table_a["name"]) + list(table_b["name"]))
    catalog = CatalogSearch(
        hybrid_vectors(tfidf_vectors(matcher.first.vectorizer), embedding_vectors(matcher.second.model), WEIGHT),
        table_b["name"],
    )
    return matcher, catalog, table_b


st.set_page_config(page_title="Product Matching", page_icon="🛒")
st.title("Product Matching")
st.write(
    "Do two e-commerce listings describe the same product? A hybrid of character TF-IDF and a "
    "fine-tuned sentence embedding, trained on the Abt-Buy benchmark (test F1 0.745). "
    "[Code and results on GitHub](https://github.com/rabiaozers/product-matching)"
)

matcher, catalog, table_b = load()

EXAMPLES = {
    "Same mouse, different code format": ("logitech black v220 cordless optical mouse 910000153",
                                          "logitech v220 cordless optical mouse 910-000153"),
    "Same camera, different wording": ("nikon black 13.5 megapixel coolpix digital camera coolpixp6000bk",
                                       "nikon coolpix p6000 13 megapixels digital camera w / 4x wide angle"),
    "Known weakness: codes differ by one digit": ("panasonic nnsd797s microwave oven",
                                                  "panasonic nn-sd767s microwave oven"),
}

match_tab, search_tab = st.tabs(["Match two titles", "Search the catalog"])

with match_tab:
    example = st.selectbox("Try an example", ["(type your own)"] + list(EXAMPLES))
    left_default, right_default = EXAMPLES.get(example, ("", ""))
    left = st.text_input("Title from store A", value=left_default)
    right = st.text_input("Title from store B", value=right_default)
    if st.button("Compare", type="primary"):
        if not left.strip() or not right.strip():
            st.warning("Please enter two product titles.")
        else:
            tfidf = float(matcher.first.score([left], [right])[0])
            embedding = float(matcher.second.score([left], [right])[0])
            score = WEIGHT * tfidf + (1 - WEIGHT) * embedding
            if score > THRESHOLD:
                st.success(f"Same product (hybrid score {score:.3f}, threshold {THRESHOLD})")
            else:
                st.info(f"Different products (hybrid score {score:.3f}, threshold {THRESHOLD})")
            st.dataframe(pd.DataFrame({
                "part": ["TF-IDF (characters)", "Fine-tuned embedding (meaning)",
                         f"Hybrid ({WEIGHT} · TF-IDF + {round(1 - WEIGHT, 1)} · embedding)"],
                "score": [round(tfidf, 3), round(embedding, 3), round(score, 3)],
            }), hide_index=True)

with search_tab:
    title = st.text_input("Product title", value="sony bravia theater black micro system davis50b")
    k = st.slider("Number of results", 1, 20, 5)
    if st.button("Search", type="primary"):
        if not title.strip():
            st.warning("Please enter a product title.")
        else:
            rows = []
            for row, score in catalog.search(title, k):
                if row < 0:   # FAISS returns -1 when fewer than k products exist
                    continue
                rows.append({
                    "Buy catalog product": table_b["name"][row],
                    "score": round(score, 3),
                    "same product?": "yes" if score > THRESHOLD else "no",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True)

st.caption("Data: Abt-Buy benchmark (DeepMatcher version). Model: all-MiniLM-L6-v2 fine-tuned on the training pairs.")
