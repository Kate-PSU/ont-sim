#!/usr/bin/env python3
"""Фиксация корпуса arXiv для воспроизводимых экспериментов."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from harvest_arxiv import ArxivClient


DEFAULT_CATEGORIES = [
    "cs.LG",
    "cs.CR",
    "q-bio",
    "physics.chem-ph",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Загрузить и сохранить фиксированный корпус arXiv."
    )
    parser.add_argument(
        "--categories",
        default=",".join(DEFAULT_CATEGORIES),
        help="Категории arXiv через запятую.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Количество статей на категорию.",
    )
    parser.add_argument(
        "--output",
        default="data/corpora/arxiv_4domains_20.json",
        help="Путь к JSON-файлу.",
    )
    args = parser.parse_args()

    categories = [
        category.strip()
        for category in args.categories.split(",")
        if category.strip()
    ]

    client = ArxivClient()

    corpus = {
        "source": "arXiv",
        "requested_categories": categories,
        "limit_per_category": args.limit,
        "domains": {},
    }

    for category in categories:
        articles = client.fetch_articles(category, limit=args.limit)

        corpus["domains"][category] = [
            asdict(article)
            for article in articles
        ]

        print(
            f"{category}: сохранено "
            f"{len(articles)}/{args.limit} статей"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            corpus,
            f,
            ensure_ascii=False,
            indent=2,
        )

    total = sum(
        len(articles)
        for articles in corpus["domains"].values()
    )

    print(f"\nВсего сохранено статей: {total}")
    print(f"Корпус сохранён: {output_path}")


if __name__ == "__main__":
    main()