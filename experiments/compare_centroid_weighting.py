from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


CSV_PATH = "/Users/ekaterina/Documents/GitHub/ont-sim/data/terms.csv"
OUTPUT_PATH = Path("experiments/results/centroid_weighting_comparison.csv")
BATCH_SIZE = 32

MODELS = [
    "ai-forever/sbert_large_nlu_ru",
    "sentence-transformers/all-mpnet-base-v2",
]


def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def centroid_unweighted(embeddings):
    return normalize(np.mean(embeddings, axis=0))


def centroid_weighted(embeddings, weights):
    return normalize(np.average(embeddings, axis=0, weights=weights))


df = pd.read_csv(CSV_PATH, keep_default_na=False)
domains = sorted(df["domain"].unique())
results = []

print(f"Corpus: {len(df)} terms")
print(f"Domains: {', '.join(domains)}")

for model_name in MODELS:
    print(f"\n{'=' * 70}")
    print(f"Model: {model_name}")
    print("=" * 70)

    model = SentenceTransformer(model_name)

    centroids = {
        "A_unweighted": {},
        "B_frequency": {},
        "C_tfidf": {},
    }

    for domain, group in df.groupby("domain"):
        terms = group["term"].tolist()

        print(f"Encoding {domain}: {len(terms)} terms")

        embeddings = model.encode(
            terms,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        frequency_weights = group["frequency"].to_numpy(dtype=float)
        tfidf_weights = group["tfidf_score"].to_numpy(dtype=float)

        centroids["A_unweighted"][domain] = centroid_unweighted(embeddings)
        centroids["B_frequency"][domain] = centroid_weighted(
            embeddings, frequency_weights
        )
        centroids["C_tfidf"][domain] = centroid_weighted(
            embeddings, tfidf_weights
        )

    for weighting, method_centroids in centroids.items():
        print(f"\n--- {weighting} ---")
        scores = []

        for d1, d2 in combinations(domains, 2):
            score = float(
                np.dot(method_centroids[d1], method_centroids[d2])
            )
            scores.append(score)

            results.append(
                {
                    "model": model_name,
                    "weighting": weighting,
                    "domain1": d1,
                    "domain2": d2,
                    "similarity": score,
                }
            )

            print(f"{d1:16s} {d2:16s} {score:.9f}")

        print(
            f"range: {min(scores):.9f} .. {max(scores):.9f}"
        )
        print(f"spread: {max(scores) - min(scores):.9f}")

    del model


OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved {len(results_df)} rows to:")
print(OUTPUT_PATH)
