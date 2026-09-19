import pandas as pd

from scripts.harvest_arxiv import TermExtractor


CSV_PATH = "/Users/ekaterina/Documents/GitHub/ont-sim/data/terms.csv"

df = pd.read_csv(CSV_PATH, keep_default_na=False)

extractor = TermExtractor()

print(f"Original rows: {len(df)}")
print()

lemmatized = []

for i, term in enumerate(df["term"], start=1):
    lemma = extractor._lemmatize_text(str(term))
    lemmatized.append(lemma)

    if i % 1000 == 0:
        print(f"Lemmatized {i}/{len(df)}")

df["lemma"] = lemmatized

print("\n=== SUMMARY BY DOMAIN ===")

for domain, group in df.groupby("domain"):
    original = len(group)

    valid = group[group["lemma"].str.strip() != ""]
    unique_lemmas = valid["lemma"].nunique()

    collapsed = len(valid) - unique_lemmas
    empty = original - len(valid)

    print(
        f"{domain:16s} "
        f"original={original:4d}  "
        f"unique_lemmas={unique_lemmas:4d}  "
        f"collapsed={collapsed:4d}  "
        f"empty={empty:3d}"
    )

print("\n=== TOTAL ===")

valid_df = df[df["lemma"].str.strip() != ""]

print(f"Original terms:      {len(df)}")
print(f"Valid after lemma:   {len(valid_df)}")
print(
    "Unique domain+lemma:",
    valid_df[["domain", "lemma"]].drop_duplicates().shape[0],
)
print(
    "Collapsed rows:     ",
    len(valid_df)
    - valid_df[["domain", "lemma"]].drop_duplicates().shape[0],
)
print(f"Empty lemmas:        {len(df) - len(valid_df)}")

print("\n=== EXAMPLES OF COLLAPSED FORMS ===")

shown = 0

for (domain, lemma), group in valid_df.groupby(["domain", "lemma"]):
    original_terms = sorted(set(group["term"]))

    if len(original_terms) > 1:
        print(
            f"{domain:16s} {lemma!r} <- "
            + ", ".join(repr(x) for x in original_terms)
        )
        shown += 1

        if shown >= 30:
            break
