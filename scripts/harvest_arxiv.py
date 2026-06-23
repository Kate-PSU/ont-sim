#!/usr/bin/env python3
"""
Скрипт сбора терминов из arXiv.

Загружает статьи из arXiv API, извлекает термины методом TF-IDF
и экспортирует результаты в CSV формат для последующего анализа.

Параметры командной строки:
  --output      Путь к выходному CSV файлу (по умолчанию: data/arxiv_terms.csv)
  --limit       Максимальное количество статей на категорию (по умолчанию: 800)
  --categories  Список категорий через запятую (по умолчанию: cs.LG,cs.CR,q-bio,physics.chem-ph)
"""

import argparse
import csv
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ============================================================================
# КОНСТАНТЫ
# ============================================================================

# URL arXiv API
ARXIV_API_URL = "http://export.arxiv.org/api/query"

# Категории по умолчанию
DEFAULT_CATEGORIES = ["cs.LG", "cs.CR", "q-bio", "physics.chem-ph"]

# Лимит статей на категорию по умолчанию
DEFAULT_LIMIT = 800

# Размер пачки для запросов к arXiv API (не более 2000 за раз)
ARXIV_BATCH_SIZE = 200

# Путь к выходному файлу по умолчанию
DEFAULT_OUTPUT_PATH = "data/arxiv_terms.csv"

# Параметры TF-IDF
TFIDF_NGRAM_RANGE = (1, 3)  # униграммы, биграммы, триграммы
TFIDF_MIN_DF = 3  # минимальная документная частота
TFIDF_MAX_DF = 0.8  # максимальная документная частота
TFIDF_MAX_FEATURES = 10000  # максимальное количество признаков

# Задержка между запросами к arXiv API (секунды)
ARXIV_REQUEST_DELAY = 3.0

# ============================================================================
# АКАДЕМИЧЕСКИЕ СТОП-СЛОВА
# ============================================================================

# Список академических и общих стоп-слов для фильтрации
# Включает: предлоги, союзы, местоимения, общие глаголы, модальные слова
ACADEMIC_STOPWORDS: set[str] = {
    # Артикли
    "a",
    "an",
    "the",
    # Предлоги
    "of",
    "in",
    "to",
    "for",
    "with",
    "on",
    "at",
    "by",
    "from",
    "as",
    "is",
    "was",
    "are",
    "were",
    "been",
    "be",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "shall",
    "can",
    "need",
    # Союзы
    "and",
    "or",
    "but",
    "if",
    "then",
    "else",
    "when",
    "where",
    "while",
    "although",
    "because",
    "since",
    "until",
    "unless",
    "though",
    "however",
    "therefore",
    # Местоимения
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "my",
    "your",
    "his",
    "its",
    "our",
    "their",
    "mine",
    "yours",
    "hers",
    "ours",
    "this",
    "that",
    "these",
    "those",
    "who",
    "whom",
    "which",
    "what",
    "whose",
    # Наречия
    "also",
    "more",
    "most",
    "very",
    "just",
    "only",
    "even",
    "still",
    "already",
    "often",
    "always",
    "never",
    "sometimes",
    "usually",
    "really",
    "well",
    "so",
    # Общие слова в научных текстах
    "paper",
    "work",
    "study",
    "research",
    "article",
    "paper",
    "approach",
    "method",
    "result",
    "results",
    "show",
    "shows",
    "shown",
    "demonstrate",
    "demonstrates",
    "propose",
    "proposes",
    "proposed",
    "suggest",
    "suggests",
    "suggested",
    "present",
    "presents",
    "presented",
    "consider",
    "considers",
    "considered",
    "provide",
    "provides",
    "provided",
    "use",
    "uses",
    "used",
    "using",
    "based",
    "different",
    "various",
    "several",
    "many",
    "such",
    "first",
    "second",
    "third",
    "new",
    "one",
    "two",
    "three",
    "four",
    "five",
    "way",
    "ways",
    "make",
    "made",
    "given",
    "give",
    "gives",
    "within",
    "between",
    "after",
    "before",
    "during",
    "each",
    "both",
    "all",
    "any",
    "some",
    "few",
    "most",
    "other",
    "another",
    "than",
    "into",
    "through",
    "over",
    "under",
    "about",
    "against",
    "among",
    # Математические и статистические термины высокой частотности
    "set",
    "sets",
    "function",
    "functions",
    "value",
    "values",
    "case",
    "cases",
    "point",
    "points",
    "number",
    "numbers",
    "model",
    "models",
    "problem",
    "problems",
    "data",
    "algorithm",
    "algorithms",
    "time",
    "example",
    "examples",
    "order",
    "equation",
    "equations",
    "theorem",
    "theorems",
    "proof",
    "proofs",
    "definition",
    "definitions",
    "system",
    "systems",
    "theory",
    "theories",
    "analysis",
    "property",
    "properties",
    "process",
    "processes",
    "state",
    "states",
    "group",
    "groups",
    # Глаголы высокой частотности
    "showed",
    "obtained",
    "given",
    "shown",
    "known",
    "found",
    "considered",
    "applied",
    "developed",
    "introduced",
    "described",
    "defined",
    "written",
    "observed",
    "reported",
    "performed",
    "estimated",
    "investigated",
    "derived",
    # Дополнительные стоп-слова
    "ie",
    "eg",
    "etc",
    "vs",
    "via",
    "per",
    "not",
    "no",
    "yes",
    "see",
    "fig",
    "figure",
    "table",
    "section",
    "chapter",
    "page",
    "pages",
    "reference",
    "references",
    "author",
    "authors",
    "author's",
    "copyright",
    "available",
    "abstract",
}

# Дополнительные паттерны для фильтрации
STOPWORD_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\d+$"),  # Только цифры
    re.compile(r"^[a-z]$"),  # Одиночные буквы
    re.compile(r"^[^a-zA-Z]+$"),  # Не содержит букв
]

# ============================================================================
# НАСТРОЙКА ЛОГИРОВАНИЯ
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)


# ============================================================================
# ТИПЫ ДАННЫХ
# ============================================================================


@dataclass
class ArxivArticle:
    """Структура данных для статьи из arXiv.

    Attributes:
        arxiv_id: Идентификатор статьи в arXiv
        title: Название статьи
        abstract: Аннотация статьи
        categories: Список категорий статьи
        authors: Список авторов
        published: Дата публикации
    """

    arxiv_id: str
    title: str
    abstract: str
    categories: list[str] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)
    published: str = ""


@dataclass
class ExtractedTerm:
    """Структура данных для извлечённого термина.

    Attributes:
        term: Сам термин (или n-грамма)
        domain: Домен/категория arXiv
        frequency: Частота встречаемости в корпусе
        tfidf_score: Средний TF-IDF score (опционально)
    """

    term: str
    domain: str
    frequency: int
    tfidf_score: Optional[float] = None


# ============================================================================
# КЛАССЫ
# ============================================================================


class ArxivClient:
    """Клиент для загрузки данных с arXiv API.

    Использует feedparser для парсинга Atom-фидов arXiv.
    Поддерживает загрузку статей по категориям с пагинацией.

    Attributes:
        base_url: Базовый URL arXiv API
        request_delay: Задержка между запросами (секунды)
    """

    def __init__(
        self, base_url: str = ARXIV_API_URL, request_delay: float = ARXIV_REQUEST_DELAY
    ) -> None:
        """Инициализация клиента arXiv.

        Args:
            base_url: URL arXiv API
            request_delay: Минимальная задержка между запросами
        """
        self.base_url = base_url
        self.request_delay = request_delay
        self._feedparser_available = self._check_feedparser()

    def _check_feedparser(self) -> bool:
        """Проверка доступности feedparser.

        Returns:
            True если feedparser установлен, иначе False
        """
        try:
            import feedparser

            return True
        except ImportError:
            logger.warning(
                "feedparser не установлен. Установите: pip install feedparser"
            )
            return False

    def fetch_articles(
        self, category: str, limit: int = DEFAULT_LIMIT
    ) -> list[ArxivArticle]:
        """Загрузка статей из arXiv по категории.

        Args:
            category: Категория arXiv (например, 'cs.LG')
            limit: Максимальное количество статей

        Returns:
            Список объектов ArxivArticle

        Raises:
            ImportError: Если feedparser не установлен
            RuntimeError: При ошибке загрузки данных
        """
        if not self._feedparser_available:
            raise ImportError(
                "feedparser обязателен для работы. "
                "Установите: pip install feedparser"
            )

        import feedparser

        articles: list[ArxivArticle] = []
        start = 0

        logger.info(f"Загрузка статей категории {category} (лимит: {limit})")

        while start < limit:
            # Размер пачки (не более 2000)
            batch_size = min(ARXIV_BATCH_SIZE, limit - start)

            # Формирование URL запроса
            query = f"cat:{category}"
            url = (
                f"{self.base_url}"
                f"?search_query={query}"
                f"&start={start}"
                f"&max_results={batch_size}"
                f"&sortBy=submittedDate"
                f"&sortOrder=descending"
            )

            logger.debug(f"Запрос: {url}")

            # Загрузка данных
            feed = feedparser.parse(url)

            if feed.bozo:
                logger.warning(
                    f"Ошибка парсинга на позиции {start}: {feed.bozo_exception}"
                )

            # Обработка записей
            entries_count = 0
            for entry in feed.entries:
                article = self._parse_entry(entry)
                if article:
                    articles.append(article)
                    entries_count += 1

                if len(articles) >= limit:
                    break

            logger.info(
                f"  Загружено {len(articles)}/{limit} статей "
                f"(итерация: +{entries_count})"
            )

            # Проверка на окончание данных
            if len(feed.entries) < batch_size:
                logger.info(f"  Достигнут конец категории {category}")
                break

            start += batch_size

            # Задержка между запросами
            if start < limit:
                time.sleep(self.request_delay)

        logger.info(f"  Итого для {category}: {len(articles)} статей")
        return articles

    def _parse_entry(self, entry) -> Optional[ArxivArticle]:
        """Парсинг одной записи из фида.

        Args:
            entry: Запись из feedparser

        Returns:
            Объект ArxivArticle или None при ошибке
        """
        try:
            # Извлечение ID (из ссылки или поля id)
            arxiv_id = ""
            if hasattr(entry, "id"):
                # ID имеет вид http://arxiv.org/abs/xxxx.xxxxx
                arxiv_id = entry.id.split("/")[-1] if "/" in entry.id else entry.id
            elif hasattr(entry, "link"):
                arxiv_id = entry.link.split("/")[-1] if "/" in entry.link else ""

            # Извлечение названия
            title = ""
            if hasattr(entry, "title"):
                title = entry.title.replace("\n", " ").strip()

            # Извлечение аннотации
            abstract = ""
            if hasattr(entry, "summary"):
                abstract = entry.summary.replace("\n", " ").strip()
            elif hasattr(entry, "summary_detail"):
                abstract = entry.summary_detail.value.replace("\n", " ").strip()

            # Извлечение категорий
            categories: list[str] = []
            if hasattr(entry, "tags"):
                categories = [tag.term for tag in entry.tags]

            # Извлечение авторов
            authors: list[str] = []
            if hasattr(entry, "authors"):
                authors = [author.name for author in entry.authors]

            # Дата публикации
            published = ""
            if hasattr(entry, "published"):
                published = entry.published

            return ArxivArticle(
                arxiv_id=arxiv_id,
                title=title,
                abstract=abstract,
                categories=categories,
                authors=authors,
                published=published,
            )

        except Exception as e:
            logger.warning(f"Ошибка парсинга записи: {e}")
            return None


class TermExtractor:
    """Извлекатель терминов с использованием TF-IDF.

    Класс выполняет следующие функции:
    1. Токенизация и предобработка текста
    2. Фильтрация стоп-слов
    3. Вычисление TF-IDF матрицы
    4. Извлечение наиболее значимых терминов
    5. Валидация терминов через spaCy (опционально)

    Attributes:
        ngram_range: Диапазон n-грамм (min, max)
        min_df: Минимальная документная частота
        max_df: Максимальная документная частота (доля)
        max_features: Максимальное количество признаков
    """

    def __init__(
        self,
        ngram_range: tuple[int, int] = TFIDF_NGRAM_RANGE,
        min_df: int = TFIDF_MIN_DF,
        max_df: float = TFIDF_MAX_DF,
        max_features: int = TFIDF_MAX_FEATURES,
    ) -> None:
        """Инициализация извлекателя терминов.

        Args:
            ngram_range: Диапазон n-грамм (например, (1,3) для 1-3 грамм)
            min_df: Минимальная документная частота
            max_df: Максимальная доля документов (0.0-1.0)
            max_features: Максимальное количество признаков TF-IDF
        """
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features

        # Проверка доступности spaCy
        self._spacy_available = self._check_spacy()
        self._nlp: Optional[object] = None

        # Инициализация TF-IDF векторизатора
        self._vectorizer = None

        # Кэш обработанных документов
        self._processed_docs: list[str] = []

    def _check_spacy(self) -> bool:
        """Проверка доступности spaCy.

        Returns:
            True если spaCy установлен, иначе False
        """
        try:
            import spacy

            # Попытка загрузки модели en_core_web_sm
            try:
                self._nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy загружена успешно (en_core_web_sm)")
                return True
            except OSError:
                logger.warning(
                    "Модель spaCy en_core_web_sm не найдена. "
                    "Валидация терминов будет пропущена. "
                    "Установите модель: python -m spacy download en_core_web_sm"
                )
                return False
        except ImportError:
            logger.warning(
                "spaCy не установлен. Валидация терминов будет пропущена. "
                "Установите: pip install spacy"
            )
            return False

    def _preprocess_text(self, text: str) -> str:
        """Предобработка текста перед TF-IDF анализом.

        Args:
            text: Исходный текст

        Returns:
            Предобработанный текст
        """
        # Удаление URL
        text = re.sub(r"http[s]?://\S+", "", text)

        # Удаление email
        text = re.sub(r"\S+@\S+", "", text)

        # Удаление специальных символов, оставляя только буквы и пробелы
        text = re.sub(r"[^a-zA-Z\s]", " ", text)

        # Нормализация пробелов
        text = re.sub(r"\s+", " ", text)

        # Приведение к нижнему регистру
        text = text.lower()

        return text.strip()

    def _filter_stopwords(self, tokens: list[str]) -> list[str]:
        """Фильтрация стоп-слов из списка токенов.

        Args:
            tokens: Список токенов

        Returns:
            Отфильтрованный список токенов
        """
        return [
            token
            for token in tokens
            if (
                token not in ACADEMIC_STOPWORDS
                and len(token) > 2
                and not any(pattern.match(token) for pattern in STOPWORD_PATTERNS)
            )
        ]

    def _validate_term_spacy(self, term: str) -> bool:
        """Валидация термина через spaCy.

        Проверяет, что термин содержит хотя бы одну
        именную часть речи (существительное, прилагательное и т.д.)

        Args:
            term: Термин для валидации

        Returns:
            True если термин валиден, False в противном случае
        """
        if not self._spacy_available or self._nlp is None:
            return True

        try:
            doc = self._nlp(term)

            # Проверяем, есть ли в термине именная часть речи
            for token in doc:
                if token.pos_ in ("NOUN", "PROPN", "ADJ"):
                    return True

            return False

        except Exception as e:
            logger.debug(f"Ошибка валидации через spaCy: {e}")
            return True  # При ошибке пропускаем термин

    def extract_terms(
        self, documents: list[str], domain: str, min_terms: int = 100
    ) -> list[ExtractedTerm]:
        """Извлечение терминов из коллекции документов.

        Args:
            documents: Список текстов документов
            domain: Домен/категория для извлечённых терминов
            min_terms: Минимальное количество терминов для возврата

        Returns:
            Список объектов ExtractedTerm
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Предобработка документов
        processed_docs = [self._preprocess_text(doc) for doc in documents]
        self._processed_docs = processed_docs

        logger.info(f"Обработано {len(documents)} документов для домена {domain}")

        # Создание TF-IDF векторизатора
        self._vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            max_features=self.max_features,
            stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",  # Минимум 2 буквы
        )

        try:
            tfidf_matrix = self._vectorizer.fit_transform(processed_docs)
        except ValueError as e:
            logger.error(f"Ошибка построения TF-IDF матрицы: {e}")
            return []

        # Получение имён признаков и статистики
        feature_names = self._vectorizer.get_feature_names_out()

        # Вычисление частотности терминов (сколько документов содержит термин)
        document_counts = (tfidf_matrix > 0).sum(axis=0).A1

        # Вычисление среднего TF-IDF для каждого термина
        mean_tfidf = tfidf_matrix.mean(axis=0).A1

        # Фильтрация и создание списка терминов
        terms: list[ExtractedTerm] = []

        for i, term in enumerate(feature_names):
            # Проверка стоп-слов
            tokens = term.lower().split()
            if not self._filter_stopwords(tokens):
                continue

            # Валидация через spaCy
            if not self._validate_term_spacy(term):
                continue

            # Добавление термина
            extracted = ExtractedTerm(
                term=term,
                domain=domain,
                frequency=int(document_counts[i]),
                tfidf_score=float(mean_tfidf[i]),
            )
            terms.append(extracted)

        # Сортировка по частоте (по убыванию)
        terms.sort(key=lambda x: x.frequency, reverse=True)

        logger.info(
            f"Извлечено {len(terms)} терминов для домена {domain} "
            f"(из {len(feature_names)} кандидатов)"
        )

        return terms


class CsvExporter:
    """Экспортёр результатов в CSV формат.

    Поддерживает экспорт с различными разделителями
    и кодировками.

    Attributes:
        output_path: Путь к выходному файлу
        delimiter: Разделитель полей
    """

    def __init__(
        self, output_path: str = DEFAULT_OUTPUT_PATH, delimiter: str = ","
    ) -> None:
        """Инициализация экспортёра.

        Args:
            output_path: Путь к выходному CSV файлу
            delimiter: Разделитель полей CSV
        """
        self.output_path = Path(output_path)
        self.delimiter = delimiter

        # Создание директории для вывода
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def export(self, terms: list[ExtractedTerm], language: str = "en") -> None:
        """Экспорт терминов в CSV файл.

        Args:
            terms: Список извлечённых терминов
            language: Язык терминов ('en' или 'ru')

        Raises:
            IOError: При ошибке записи файла
        """
        fieldnames = ["term", "domain", "frequency"]

        # Добавление дополнительных полей для TF-IDF
        has_tfidf = any(t.tfidf_score is not None for t in terms)
        if has_tfidf:
            fieldnames.append("tfidf_score")

        logger.info(f"Экспорт {len(terms)} терминов в {self.output_path}")

        try:
            with open(self.output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=fieldnames,
                    delimiter=self.delimiter,
                    quoting=csv.QUOTE_MINIMAL,
                )

                writer.writeheader()

                for term in terms:
                    row: dict[str, object] = {
                        "term": term.term,
                        "domain": term.domain,
                        "frequency": term.frequency,
                    }

                    if has_tfidf:
                        row["tfidf_score"] = term.tfidf_score or 0.0

                    writer.writerow(row)

            logger.info(f"Экспорт завершён: {self.output_path}")

        except IOError as e:
            logger.error(f"Ошибка записи CSV: {e}")
            raise


# ============================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================================


def parse_arguments() -> argparse.Namespace:
    """Парсинг аргументов командной строки.

    Returns:
        Объект с распарсенными аргументами
    """
    parser = argparse.ArgumentParser(
        description="Скрипт сбора терминов из аннотаций arXiv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python scripts/harvest_arxiv_terms.py
  python scripts/harvest_arxiv_terms.py --output my_terms.csv --limit 500
  python scripts/harvest_arxiv_terms.py --categories cs.LG,cs.CV
        """,
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Путь к выходному CSV файлу (по умолчанию: {DEFAULT_OUTPUT_PATH})",
    )

    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Максимальное количество статей на категорию (по умолчанию: {DEFAULT_LIMIT})",
    )

    parser.add_argument(
        "--categories",
        "-c",
        type=str,
        default=",".join(DEFAULT_CATEGORIES),
        help=f'Категории arXiv через запятую (по умолчанию: {",".join(DEFAULT_CATEGORIES)})',
    )

    parser.add_argument(
        "--language",
        "-lang",
        type=str,
        choices=["en", "ru"],
        default="en",
        help="Язык терминов для экспорта (по умолчанию: en)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Подробный вывод (DEBUG level)"
    )

    return parser.parse_args()


def main() -> int:
    """Основная функция скрипта.

    Returns:
        Код завершения (0 - успех, 1 - ошибка)
    """
    # Парсинг аргументов
    args = parse_arguments()

    # Настройка уровня логирования
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("Скрипт сбора терминов из arXiv")
    logger.info("=" * 60)

    # Парсинг категорий
    categories = [cat.strip() for cat in args.categories.split(",")]
    logger.info(f"Категории: {categories}")
    logger.info(f"Лимит на категорию: {args.limit}")
    logger.info(f"Выходной файл: {args.output}")

    # Проверка наличия feedparser
    try:
        import feedparser
    except ImportError:
        logger.error(
            "feedparser не установлен. Установите зависимости:\n"
            "  source venv/bin/activate\n"
            "  pip install feedparser"
        )
        return 1

    # Инициализация клиента и извлекателя
    client = ArxivClient()
    extractor = TermExtractor()
    exporter = CsvExporter(output_path=args.output)

    # Сбор всех терминов
    all_terms: list[ExtractedTerm] = []
    total_articles = 0

    for category in categories:
        logger.info("-" * 40)

        try:
            # Загрузка статей
            articles = client.fetch_articles(category=category, limit=args.limit)

            if not articles:
                logger.warning(f"Нет статей для категории {category}")
                continue

            total_articles += len(articles)

            # Извлечение текстов (аннотации + названия)
            documents = []
            for article in articles:
                # Комбинируем название и аннотацию
                text = f"{article.title} {article.abstract}"
                documents.append(text)

            # Извлечение терминов
            terms = extractor.extract_terms(documents=documents, domain=category)

            all_terms.extend(terms)

            logger.info(
                f"Итого для {category}: {len(terms)} терминов из {len(articles)} статей"
            )

        except Exception as e:
            logger.error(f"Ошибка обработки категории {category}: {e}")
            continue

    # Экспорт результатов
    if all_terms:
        logger.info("-" * 40)
        logger.info(f"Всего извлечено терминов: {len(all_terms)}")
        logger.info(f"Всего обработано статей: {total_articles}")

        exporter.export(terms=all_terms, language=args.language)
    else:
        logger.warning("Термины не найдены. CSV файл не создан.")
        return 1

    logger.info("=" * 60)
    logger.info("Сбор терминов завершён успешно")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
