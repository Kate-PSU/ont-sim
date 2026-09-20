"""
Experiment 3: cross-domain IDF weighting.

Compares:
1. lemma + conservative metatext filtering + unweighted centroid
2. lemma + conservative metatext filtering + domain-IDF weighted centroid

The domain-IDF rule was defined before observing similarity results:

    w(t) = log((N + 1) / (df(t) + 1)) + 1

where:
    N     = number of domains
    df(t) = number of domains containing term t

No terms are removed by domain frequency.
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from scripts.harvest_arxiv import TermExtractor


# Historical corpus used in Experiments 1–2.
TERMS_PATH = Path(
    "/Users/ekaterina/Documents/GitHub/ont-sim/data/terms.csv"
)

OUTPUT_PATH = Path(
    "experiments/results/domain_idf_comparison.csv"
)

MODELS = {
    "ru_sbert": "ai-forever/sbert_large_nlu_ru",
    "en_mpnet": "sentence-transformers/all-mpnet-base-v2",
}

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


def prepare_terms() -> pd.DataFrame:
    """Lemmatize, remove conservative metatext, and deduplicate."""

    df = pd.read_csv(TERMS_PATH, keep_default_na=False)

    extractor = TermExtractor()

    df["lemma"] = [
        extractor._lemmatize_text(str(term))
        for term in df["term"]
    ]

    df = df[df["lemma"].str.strip() != ""].copy()

    df = df[
        ~df["lemma"].apply(
            lambda term: bool(
                set(term.split()) & METATEXT_LEMMAS
            )
        )
    ].copy()

    # Clean post-hoc ablation:
    # one canonical lemma per domain.
    df = df.drop_duplicates(["domain", "lemma"]).copy()

    return df


def add_domain_idf(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate cross-domain IDF without deleting any terms."""

    df = df.copy()

    n_domains = df["domain"].nunique()

    domain_frequency = (
        df.groupby("lemma")["domain"]
        .nunique()
        .to_dict()
    )

    df["domain_frequency"] = df["lemma"].map(
        domain_frequency
    )

    df["domain_idf"] = df["domain_frequency"].apply(
        lambda freq: math.log(
            (n_domains + 1) / (freq + 1)
        ) + 1.0
    )

    return df


def centroid(
    embeddings: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Build and L2-normalize a centroid."""

    if weights is None:
        vector = np.mean(embeddings, axis=0)
    else:
        vector = np.average(
            embeddings,
            axis=0,
            weights=weights,
        )

    norm = np.linalg.norm(vector)

    if norm > 0:
        vector = vector / norm

    return vector


def cosine_similarity(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
) -> float:
    """Cosine similarity of normalized centroid vectors."""

    return float(np.dot(vector_a, vector_b))


def run_model(
    model_label: str,
    model_name: str,
    df: pd.DataFrame,
) -> list[dict]:
    """Run unweighted and domain-IDF centroid variants."""

    print(f"\nLoading model: {model_label}")
    print(model_name)

    model = SentenceTransformer(model_name)

    domains = sorted(df["domain"].unique())

    unweighted_centroids = {}
    domain_idf_centroids = {}

    for domain in domains:
        subset = df[df["domain"] == domain]

        terms = subset["lemma"].tolist()

        weights = subset["domain_idf"].to_numpy(
            dtype=float
        )

        print(
            f"  {domain:16s}: "
            f"{len(terms)} terms"
        )

        embeddings = model.encode(
            terms,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        unweighted_centroids[domain] = centroid(
            embeddings
        )

        domain_idf_centroids[domain] = centroid(
            embeddings,
            weights=weights,
        )

    rows = []

    variants = {
        "lemma_metatext_unweighted":
            unweighted_centroids,
        "lemma_metatext_domain_idf":
            domain_idf_centroids,
    }

    for variant, centroids in variants.items():
        print(f"\n{model_label} | {variant}")

        scores = []

        for i, domain_a in enumerate(domains):
            for domain_b in domains[i + 1:]:
                score = cosine_similarity(
                    centroids[domain_a],
                    centroids[domain_b],
                )

                scores.append(score)

                rows.append(
                    {
                        "model": model_label,
                        "model_name": model_name,
                        "variant": variant,
                        "domain_a": domain_a,
                        "domain_b": domain_b,
                        "similarity": score,
                    }
                )

                print(
                    f"  {domain_a:16s} - "
                    f"{domain_b:16s} "
                    f"{score:.9f}"
                )

        spread = max(scores) - min(scores)

        print(
            f"  range: "
            f"{min(scores):.9f} .. "
            f"{max(scores):.9f}"
        )

        print(
            f"  spread: {spread:.9f}"
        )

    return rows


def main() -> None:
    print("=== EXPERIMENT 3: DOMAIN-IDF WEIGHTING ===")

    df = prepare_terms()
    df = add_domain_idf(df)

    print(
        f"\nDomains: {df['domain'].nunique()}"
    )
    print(
        f"Domain+lemma rows: {len(df)}"
    )

    print("\nDomain-IDF weights:")

    weights = (
        df[
            ["domain_frequency", "domain_idf"]
        ]
        .drop_duplicates()
        .sort_values("domain_frequency")
    )

    for _, row in weights.iterrows():
        print(
            f"  DF={int(row['domain_frequency'])}: "
            f"{row['domain_idf']:.6f}"
        )

    all_rows = []

    for model_label, model_name in MODELS.items():
        all_rows.extend(
            run_model(
                model_label,
                model_name,
                df,
            )
        )

    results = pd.DataFrame(all_rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()