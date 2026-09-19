from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from scripts.harvest_arxiv import TermExtractor


CSV_PATH = "/Users/ekaterina/Documents/GitHub/ont-sim/data/terms.csv"
OUTPUT_PATH = Path(
    "experiments/results/metatext_filter_comparison.csv"
)
BATCH_SIZE = 32

MODELS = [
    "ai-forever/sbert_large_nlu_ru",
    "sentence-transformers/all-mpnet-base-v2",
]

# Conservative metatextual markers selected before similarity evaluation.
# These describe scientific discourse/reporting rather than a domain object.
METATEXT_LEMMAS = {
    "result",
    "introduce",
    "demonstrate",
    "provide",
    "propose",
    "paper",
    "report",
    "show",
    "present",
    "describe",
}


def normalize(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def contains_metatext(lemma):
    return bool(set(lemma.split()) & METATEXT_LEMMAS)


df = pd.read_csv(CSV_PATH, keep_default_na=False)

extractor = TermExtractor()

print(f"Original rows: {len(df)}")
print("Lemmatizing terms...")

df["lemma"] = [
    extractor._lemmatize_text(str(term))
    for term in df["term"]
]

lemma_df = (
    df[df["lemma"].str.strip() != ""]
    [["domain", "lemma"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

filtered_df = lemma_df[
    ~lemma_df["lemma"].apply(contains_metatext)
].copy()

print(f"Unique domain+lemma rows: {len(lemma_df)}")
print(
    "Removed by metatext filter:",
    len(lemma_df) - len(filtered_df),
)
print(f"Remaining rows: {len(filtered_df)}")

print("\n=== REMOVAL BY DOMAIN ===")

for domain, group in lemma_df.groupby("domain"):
    filtered_group = filtered_df[
        filtered_df["domain"] == domain
    ]

    removed = len(group) - len(filtered_group)

    print(
        f"{domain:16s} "
        f"before={len(group):4d}  "
        f"removed={removed:3d}  "
        f"after={len(filtered_group):4d}"
    )

print()

results = []

for model_name in MODELS:
    print("=" * 70)
    print(f"Model: {model_name}")
    print("=" * 70)

    model = SentenceTransformer(model_name)
    centroids = {}

    for domain, group in filtered_df.groupby("domain"):
        terms = group["lemma"].tolist()

        print(
            f"Encoding {domain}: "
            f"{len(terms)} filtered lemmas"
        )

        embeddings = model.encode(
            terms,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        centroid = normalize(embeddings.mean(axis=0))
        centroids[domain] = centroid

    scores = []

    print("\n--- lemma_metatext_filtered ---")

    for d1, d2 in combinations(sorted(centroids), 2):
        score = float(
            np.dot(centroids[d1], centroids[d2])
        )
        scores.append(score)

        results.append(
            {
                "model": model_name,
                "preprocessing": (
                    "lemmatized_deduplicated_"
                    "metatext_filtered"
                ),
                "weighting": "unweighted",
                "domain1": d1,
                "domain2": d2,
                "similarity": score,
            }
        )

        print(
            f"{d1:16s} {d2:16s} "
            f"{score:.9f}"
        )

    print(
        f"range: {min(scores):.9f} .. "
        f"{max(scores):.9f}"
    )
    print(
        f"spread: "
        f"{max(scores) - min(scores):.9f}"
    )
    print()

    del model


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved {len(results_df)} rows to:")
print(OUTPUT_PATH)
