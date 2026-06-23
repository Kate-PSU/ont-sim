#!/usr/bin/env python3
"""
Сбор терминов из Wikipedia API (async version).

Собирает статьи из иерархии категорий для каждого домена,
используя асинхронный пул запросов с auto-scaling.

Особенности:
- Concurrent requests: 4-10 (auto-scale based на rate limits)
- Delay: 2 секунды после каждого успешного запроса
- Exponential backoff при 429 errors

Usage:
    python harvest-wikipedia.py --domain physics --depth 3 --lang en
    python harvest-wikipedia.py --domain mathematics --depth 4 --lang ru --minimal
    python harvest-wikipedia.py --all-domains --depth 3 --lang en
"""

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ============================================================================
# Конфигурация
# ============================================================================

DOMAINS = {
    "en": {
        "physics": "Physics",
        "mathematics": "Mathematics",
        "biology": "Biology",
        "chemistry": "Chemistry",
        "medicine": "Medicine",
        "cs": "Computer_science",
        "economics": "Economics",
        "law": "Law",
        "linguistics": "Linguistics",
        "security": "Computer_security",
    },
    "ru": {
        "physics": "Физика",
        "mathematics": "Математика",
        "biology": "Биология",
        "chemistry": "Химия",
        "medicine": "Медицина",
        "cs": "Информатика",
        "economics": "Экономика",
        "law": "Право",
        "linguistics": "Лингвистика",
        "security": "Информационная_безопасность",
    },
}

WIKIPEDIA_API = {
    "en": "https://en.wikipedia.org/w/api.php",
    "ru": "https://ru.wikipedia.org/w/api.php",
}


# ============================================================================
# Модели данных
# ============================================================================


@dataclass
class TermData:
    """Данные термина из Wikipedia."""

    term: str
    domain: str
    pageid: Optional[int] = None
    categories: list[str] = field(default_factory=list)
    related_terms: list[str] = field(default_factory=list)
    depth: int = 0
    source_category: str = ""
    frequency: int = 1  # Сколько раз встретился термин

    def to_json(self) -> dict:
        """Сериализация в JSON-совместимый dict."""
        return asdict(self)


@dataclass
class VisitedState:
    """Состояние собранных данных для continue mode."""

    terms: dict[str, TermData] = field(default_factory=dict)
    visited_pages: set[str] = field(default_factory=set)
    visited_categories: set[str] = field(default_factory=set)


# ============================================================================
# Async Wikipedia API клиент с concurrent pool
# ============================================================================


class AsyncWikipediaClient:
    """
    Async клиент для Wikipedia API.

    Особенности:
    - Semaphore-based concurrency (min/max)
    - Auto-scaling на основе rate limits
    - 2 секунды delay после успешного запроса
    """

    def __init__(
        self,
        lang: str = "en",
        min_concurrency: int = 4,
        max_concurrency: int = 10,
        success_delay: float = 2.0,
        max_retries: int = 5,
    ):
        self.api_url = WIKIPEDIA_API[lang]
        self.lang = lang
        self.min_concurrency = min_concurrency
        self.max_concurrency = max_concurrency
        self.success_delay = success_delay
        self.max_retries = max_retries

        # Текущее состояние concurrency
        self.current_concurrency = min_concurrency
        self.semaphore: asyncio.Semaphore | None = None

        # Статистика
        self.request_count = 0
        self.success_count = 0
        self.rate_limit_count = 0
        self.failed_requests: list[str] = []

        # Semaphore создаётся в _get_semaphore()
        self._semaphore: asyncio.Semaphore | None = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Lazy initialization семафора."""
        if (
            self._semaphore is None
            or self._semaphore._value != self.current_concurrency
        ):
            self._semaphore = asyncio.Semaphore(self.current_concurrency)
        return self._semaphore

    async def _make_request(self, params: dict, retry_count: int = 0) -> dict:
        """
        Выполнить запрос к Wikipedia API через семафор.

        Args:
            params: Параметры запроса
            retry_count: Номер попытки retry

        Returns:
            Response data
        """
        params["format"] = "json"
        url = f"{self.api_url}?{self._encode_params(params)}"

        semaphore = self._get_semaphore()

        async with semaphore:
            try:
                import aiohttp

                headers = {"User-Agent": "Ont-Sim/1.0 (academic project)"}

                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status == 429:
                            # Rate limited - exponential backoff + reduce concurrency
                            self.rate_limit_count += 1
                            self._reduce_concurrency()

                            wait_time = 60
                            print(
                                f"  [429 Rate Limited] Waiting {wait_time}s, concurrency={self.current_concurrency}..."
                            )
                            await asyncio.sleep(wait_time)

                            if retry_count < self.max_retries - 1:
                                return await self._make_request(params, retry_count + 1)
                            else:
                                self.failed_requests.append(url)
                                return {
                                    "error": f"HTTP 429 after {self.max_retries} retries"
                                }

                        if response.status != 200:
                            print("request failed", response)
                            self.failed_requests.append(url)
                            return {"error": f"HTTP {response.status}"}

                        data = await response.json()
                        self.request_count += 1
                        self.success_count += 1

                        # После успеха ждём + увеличиваем concurrency
                        await asyncio.sleep(self.success_delay)
                        self._increase_concurrency()

                        return data

            except asyncio.TimeoutError:
                print(f"  [Timeout] {url}")
                self.failed_requests.append(url)
                return {"error": "Timeout"}
            except Exception as e:
                print(f"  [Error] {url}: {e}")
                self.failed_requests.append(url)
                return {"error": str(e)}

    @staticmethod
    def _encode_params(params: dict) -> str:
        """Encode params manually (without urllib)."""
        parts = []
        for k, v in params.items():
            parts.append(f"{k}={v}")
        return "&".join(parts)

    def _reduce_concurrency(self) -> None:
        """Установить concurrency = 1 при rate limit."""
        self.current_concurrency = 1
        self._semaphore = None  # Сбросить для пересоздания

    def _increase_concurrency(self) -> None:
        """Увеличить concurrency при успехе (каждые 50 запросов)."""
        if self.current_concurrency < self.max_concurrency:
            if self.success_count % 50 == 0:
                self.current_concurrency = min(
                    self.max_concurrency, self.current_concurrency + 1
                )
                self._semaphore = None  # Сбросить для пересоздания

    async def get_category_members(
        self, category: str, limit: int = 100, cmtype: str = "page"
    ) -> list[dict]:
        """
        Получить членов категории.

        Args:
            category: Название категории
            limit: Максимальное количество
            cmtype: Тип ("page", "subcat", "file")

        Returns:
            Список страниц/подкатегорий
        """
        if not category.startswith("Category:"):
            category = f"Category:{category}"

        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmlimit": min(limit, 500),
            "cmtype": cmtype,
        }

        results = []
        continue_token = None

        while True:
            if continue_token:
                params["cmcontinue"] = continue_token

            data = await self._make_request(params)

            if "error" in data:
                break

            members = data.get("query", {}).get("categorymembers", [])
            results.extend(members)

            continue_info = data.get("continue", {})
            continue_token = continue_info.get("cmcontinue")

            if not continue_token or len(results) >= limit:
                break

        return results[:limit]

    async def get_page_data(self, titles: list[str]) -> dict[int, dict]:
        """
        Получить данные страниц (категории, ссылки).

        Returns:
            Dict pageid -> page data
        """
        if not titles:
            return {}

        results = {}

        # Разбиваем на batches по 50 титулов
        for i in range(0, len(titles), 50):
            batch = titles[i : i + 50]
            titles_str = "|".join(batch)

            params = {
                "action": "query",
                "titles": titles_str,
                "prop": "categories|links",
                "cllimit": 500,
                "pllimit": 500,
                "plnamespace": 0,  # Только статьи
            }

            data = await self._make_request(params)

            if "error" in data:
                continue

            pages = data.get("query", {}).get("pages", {})
            results.update(pages)

        return results

    async def retry_failed_requests(self) -> int:
        """Retry неудачных запросов."""
        if not self.failed_requests:
            return 0

        print(f"\nRetrying {len(self.failed_requests)} failed requests...")
        successful = 0
        remaining = []

        for url in self.failed_requests:
            try:
                import aiohttp

                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url, timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status == 200:
                            await response.json()
                            successful += 1
                            await asyncio.sleep(2)
            except Exception:
                remaining.append(url)

        self.failed_requests = remaining
        if remaining:
            print(f"  {successful} succeeded, {len(remaining)} still failed")
        else:
            print(f"  All {successful} requests succeeded!")

        return successful


# ============================================================================
# Утилиты
# ============================================================================


def load_existing_csv(domain: str, lang: str) -> VisitedState:
    """Загрузить существующие термины из CSV файла.

    Args:
        domain: Ключ домена
        lang: Язык (en/ru)

    Returns:
        VisitedState с терминами
    """
    csv_path = Path(f"data/wiki_{lang}/{domain}.csv")
    state = VisitedState()

    if not csv_path.exists():
        return state

    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline()  # Пропускаем заголовок
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 2)  # term,domain,frequency
            if len(parts) >= 3:
                term, dom, freq = parts[0], parts[1], parts[2]
                try:
                    frequency = int(freq)
                except ValueError:
                    frequency = 1

                state.terms[term] = TermData(
                    term=term,
                    domain=dom,
                    frequency=frequency,
                )
                state.visited_pages.add(term)

    print(f"  [Continue] Loaded {len(state.terms)} existing terms from {csv_path}")
    return state


# ============================================================================
# Async Harvester
# ============================================================================


class AsyncWikipediaHarvester:
    """Async сборщик терминов из Wikipedia."""

    def __init__(
        self,
        lang: str = "en",
        depth: int = 1,
        limit_per_level: int = 50,
        min_concurrency: int = 4,
        max_concurrency: int = 10,
        success_delay: float = 2.0,
        max_retries: int = 5,
        continue_mode: bool = True,
    ):
        self.lang = lang
        self.depth = depth
        self.limit_per_level = limit_per_level
        self.domains_config = DOMAINS[lang]
        self.continue_mode = continue_mode

        # Создаём async клиента
        self.client = AsyncWikipediaClient(
            lang=lang,
            min_concurrency=min_concurrency,
            max_concurrency=max_concurrency,
            success_delay=success_delay,
            max_retries=max_retries,
        )

    async def harvest_domain(
        self, domain: str, save_full: bool = True, save_minimal: bool = True
    ) -> list[TermData]:
        """
        Собрать термины для одного домена.

        Args:
            domain: Ключ домена
            save_full: Сохранять полные JSON данные
            save_minimal: Сохранять минимальный CSV

        Returns:
            Список собранных терминов
        """
        if domain not in self.domains_config:
            raise ValueError(
                f"Unknown domain: {domain}. Available: {list(self.domains_config.keys())}"
            )

        seed_category = self.domains_config[domain]
        print(f"Harvesting {domain} from '{seed_category}' (depth={self.depth})")

        # Загружаем состояние для continue mode
        state = VisitedState()
        if self.continue_mode:
            state = load_existing_csv(domain, self.lang)

        # BFS по категориям
        queue: list[tuple[str, int]] = [(seed_category, 0)]

        while queue:
            category_name, current_depth = queue.pop(0)

            if current_depth > self.depth:
                continue

            if category_name in state.visited_categories:
                continue
            state.visited_categories.add(category_name)

            print(
                f"  [{current_depth}] Category: {category_name}...", end=" ", flush=True
            )

            # Получаем страницы и подкатегории (параллельно)
            members_task = self.client.get_category_members(
                category_name, limit=self.limit_per_level, cmtype="page"
            )
            subcats_task = self.client.get_category_members(
                category_name, limit=self.limit_per_level, cmtype="subcat"
            )

            members, subcategories = await asyncio.gather(members_task, subcats_task)

            # Добавляем новые подкатегории в очередь
            if current_depth < self.depth:
                for subcat in subcategories:
                    subcat_name = subcat["title"].replace("Category:", "")
                    # Пропускаем если категория уже собрана
                    if subcat_name in state.visited_categories:
                        continue
                    # Пропускаем если категория уже есть как термин (terminology collision)
                    if subcat_name in state.terms:
                        continue
                    queue.append((subcat_name, current_depth + 1))

            # Фильтруем - берём только новые страницы
            page_titles = [
                m["title"] for m in members if m["title"] not in state.visited_pages
            ]

            if page_titles:
                # Получаем данные только для новых страниц
                page_data = await self.client.get_page_data(page_titles)

                new_count = 0
                for page in page_data.values():
                    if "pageid" not in page:
                        continue

                    page_title = page.get("title", "")
                    state.visited_pages.add(page_title)

                    # Если термин уже есть - увеличиваем frequency
                    if page_title in state.terms:
                        state.terms[page_title].frequency += 1
                        continue

                    # Извлекаем связанные термины (только из main namespace)
                    related = []
                    for link in page.get("links", []):
                        link_title = link.get("title", "")
                        if link_title and ":" not in link_title:
                            related.append(link_title)

                    # Категории страницы
                    categories = [
                        cat["title"].replace("Category:", "")
                        for cat in page.get("categories", [])
                        if "Category:" in cat.get("title", "")
                    ]

                    term = TermData(
                        term=page_title,
                        domain=domain,
                        pageid=page.get("pageid"),
                        categories=categories,
                        related_terms=related[:20],
                        depth=current_depth,
                        source_category=category_name,
                        frequency=1,
                    )
                    state.terms[page_title] = term
                    new_count += 1

                print(
                    f"found {len(page_titles)} pages ({new_count} new), total: {len(state.terms)}"
                )
            else:
                print(f"found 0 new pages, total: {len(state.terms)}")

        terms = list(state.terms.values())

        # Сортируем по frequency descending
        terms.sort(key=lambda t: t.frequency, reverse=True)

        # Сохранение результатов
        output_dir = Path(f"data/wiki_{self.lang}")
        output_dir.mkdir(parents=True, exist_ok=True)

        if save_full:
            full_path = output_dir / f"{domain}_full.json"
            with open(full_path, "w", encoding="utf-8") as f:
                json.dump([t.to_json() for t in terms], f, ensure_ascii=False, indent=2)
            print(f"  Saved full JSON: {full_path}")

        if save_minimal:
            minimal_path = output_dir / f"{domain}.csv"
            with open(minimal_path, "w", encoding="utf-8") as f:
                f.write("term,domain,frequency\n")
                for t in terms:
                    f.write(f"{t.term},{t.domain},{t.frequency}\n")
            print(f"  Saved minimal CSV: {minimal_path}")

        print(f"  Total: {len(terms)} unique terms")
        print(
            f"  Stats: {self.client.request_count} requests, "
            f"{self.client.success_count} success, "
            f"{self.client.rate_limit_count} rate-limited"
        )

        return terms

    async def harvest_all_domains(
        self, save_full: bool = True, save_minimal: bool = True
    ) -> dict[str, list[TermData]]:
        """Собрать термины для всех доменов."""
        all_results = {}

        for domain in self.domains_config:
            print(f"\n{'='*60}")
            results = await self.harvest_domain(domain, save_full, save_minimal)
            all_results[domain] = results

        # Retry failed requests
        if self.client.failed_requests:
            await self.client.retry_failed_requests()

        # Объединённые файлы
        output_dir = Path(f"data/wiki_{self.lang}")
        output_dir.mkdir(parents=True, exist_ok=True)

        all_terms = []
        for terms in all_results.values():
            all_terms.extend(terms)

        if save_full:
            full_path = output_dir / "all_domains_full.json"
            with open(full_path, "w", encoding="utf-8") as f:
                json.dump(
                    [t.to_json() for t in all_terms], f, ensure_ascii=False, indent=2
                )
            print(f"\nSaved all domains full JSON: {full_path}")

        if save_minimal:
            minimal_path = output_dir / "all_domains.csv"
            with open(minimal_path, "w", encoding="utf-8") as f:
                f.write("term,domain,frequency\n")
                seen = set()
                for t in all_terms:
                    key = (t.term, t.domain)
                    if key not in seen:
                        seen.add(key)
                        f.write(f"{t.term},{t.domain},{t.frequency}\n")
            print(f"Saved all domains minimal CSV: {minimal_path}")

        return all_results


# ============================================================================
# CLI
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Harvest terms from Wikipedia API (async)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Собрать физику, английская Wikipedia, глубина 3
  python harvest_wikipedia.py --domain physics --depth 3 --lang en
  
  # Собрать математику, русская Wikipedia, глубина 4
  python harvest_wikipedia.py --domain mathematics --depth 4 --lang ru
  
  # Все домены, английская Wikipedia
  python harvest_wikipedia.py --all-domains --depth 3 --lang en
  
Concurrency settings:
  --min-concurrency 4    # Starting concurrent requests
  --max-concurrency 10  # Maximum concurrent requests
  --success-delay 2.0    # Delay after each success (seconds)
        """,
    )

    parser.add_argument(
        "--domain", "-d", choices=list(DOMAINS["en"].keys()), help="Domain to harvest"
    )

    parser.add_argument(
        "--all-domains", action="store_true", help="Harvest all domains (default)"
    )

    parser.add_argument(
        "--continue",
        dest="continue_mode",
        action="store_true",
        default=True,
        help="Continue from existing data (merge with existing CSV)",
    )

    parser.add_argument(
        "--no-continue",
        dest="continue_mode",
        action="store_false",
        help="Overwrite existing data",
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=1,
        choices=range(1, 6),
        help="Maximum category depth (default: 1)",
    )

    parser.add_argument(
        "--lang",
        choices=["all", "en", "ru"],
        default="all",
        help="Wikipedia language (all/en/ru)",
    )

    parser.add_argument(
        "--limit-per-level",
        type=int,
        default=50,
        help="Max pages per category level (default: 50)",
    )

    # Concurrency settings
    parser.add_argument(
        "--min-concurrency",
        type=int,
        default=4,
        help="Minimum concurrent requests (default: 4)",
    )

    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=10,
        help="Maximum concurrent requests (default: 10)",
    )

    parser.add_argument(
        "--success-delay",
        type=float,
        default=2.0,
        help="Delay after successful request in seconds (default: 2.0)",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retry attempts on rate limit (default: 5)",
    )

    parser.add_argument(
        "--minimal", action="store_true", help="Save only minimal CSV (term,domain)"
    )

    parser.add_argument(
        "--no-minimal",
        dest="save_minimal",
        action="store_false",
        default=True,
        help="Don't save minimal CSV",
    )

    parser.add_argument(
        "--full", action="store_true", help="Save full JSON data (default)"
    )

    parser.add_argument(
        "--no-full",
        dest="save_full",
        action="store_false",
        default=True,
        help="Don't save full JSON",
    )

    return parser.parse_args()


async def async_main():
    args = parse_args()

    if not args.domain and not args.all_domains:
        print("Error: specify --domain or --all-domains")
        return 1

    save_full = getattr(args, "save_full", True)
    save_minimal = getattr(args, "save_minimal", True)

    if args.minimal:
        save_full = False
        save_minimal = True

    # Определяем языки: "all" = оба, иначе один
    langs = ["en", "ru"] if args.lang == "all" else [args.lang]

    for lang in langs:
        harvester = AsyncWikipediaHarvester(
            lang=lang,
            depth=args.depth,
            limit_per_level=args.limit_per_level,
            min_concurrency=args.min_concurrency,
            max_concurrency=args.max_concurrency,
            success_delay=args.success_delay,
            max_retries=args.max_retries,
            continue_mode=args.continue_mode,
        )

        print(f"\n{'#'*60}")
        print(f"# Harvesting Wikipedia ({lang})")
        print(
            f"# Concurrency: {args.min_concurrency}-{args.max_concurrency}, delay: {args.success_delay}s"
        )
        print(f"{'#'*60}")

        if args.all_domains:
            await harvester.harvest_all_domains(save_full, save_minimal)
        else:
            await harvester.harvest_domain(args.domain, save_full, save_minimal)

    return 0


def main():
    return asyncio.run(async_main())


if __name__ == "__main__":
    exit(main())
