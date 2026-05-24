"""
Модуль формализованных Research Questions для бенчмарка.

Каждый Research Question связывает исследовательский вопрос из диссертации
с контрастными парами методов и критериями успеха.

Источник: vkr/notes/2026-04-12_benchmark-redesign.md
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ResearchQuestionID(Enum):
    """ID исследовательских вопросов."""
    RQ1 = "rq1"  # Контекстные vs Статические эмбеддинги
    RQ2 = "rq2"  # Лексикографические vs Векторные методы
    RQ3 = "rq3"  # Тематические модели vs Косинусное сходство


@dataclass(frozen=True)
class ContrastPair:
    """Контрастная пара методов.
    
    Сравнивает два метода на предмет гипотезы RQ.
    """
    method_a: str  # Первый метод (baseline или контроль)
    method_b: str  # Второй метод (экспериментальный)
    rationale: str  # Почему именно это сравнение
    
    @property
    def name(self) -> str:
        """Название пары в формате 'method_a vs method_b'."""
        return f"{self.method_a} vs {self.method_b}"


@dataclass(frozen=True)
class ResearchQuestion:
    """Формализованный Research Question.
    
    Attributes:
        id: Уникальный идентификатор (RQ1, RQ2, RQ3)
        title: Краткое название вопроса
        description: Полное описание исследования
        hypothesis: Гипотеза, которую проверяем
        contrast_pairs: Список контрастных пар для проверки
        success_criteria: Минимальные значения метрик для подтверждения гипотезы
        datasets: Датасеты для валидации
    """
    id: ResearchQuestionID
    title: str
    description: str
    hypothesis: str
    contrast_pairs: tuple[ContrastPair, ...]
    success_criteria: dict[str, float]  # method -> min_spearman
    datasets: tuple[str, ...]


# =============================================================================
# RQ1: Контекстные vs Статические эмбеддинги
# =============================================================================
RQ1_PAIRS = (
    ContrastPair(
        method_a="sbert",
        method_b="doc2vec",
        rationale="SBERT учитывает контекст, Doc2Vec — статические векторы",
    ),
)

RQ1 = ResearchQuestion(
    id=ResearchQuestionID.RQ1,
    title="Контекстные эмбеддинги (SBERT) vs Статические (Doc2Vec)",
    description=(
        "Исследование влияет ли учет контекста слова на качество оценки "
        "семантической близости в русском языке. SBERT генерирует разные "
        "векторы для одного слова в разных контекстах, в отличие от Doc2Vec."
    ),
    hypothesis=(
        "Контекстные эмбеддинги (SBERT) лучше моделируют полисемию "
        "русских слов и обеспечивают более высокую корреляцию с экспертными "
        "оценками близости, чем статические эмбеддинги (Doc2Vec)."
    ),
    contrast_pairs=RQ1_PAIRS,
    success_criteria={
        "sbert": 0.50,   # SBERT должен быть значимо лучше Doc2Vec
        "doc2vec": 0.00, # Doc2Vec как минимум не отрицательный
    },
    datasets=("hj-rg", "simlex999_ru", "simlex999_en"),
)


# =============================================================================
# RQ2: Лексикографические vs Векторные методы
# =============================================================================
RQ2_PAIRS = (
    ContrastPair(
        method_a="wordnet_lin",
        method_b="sbert",
        rationale="WordNet использует таксономию, SBERT — векторное пространство",
    ),
    ContrastPair(
        method_a="wordnet_lin",
        method_b="tfidf",
        rationale="WordNet vs TF-IDF на русском корпусе",
    ),
)

RQ2 = ResearchQuestion(
    id=ResearchQuestionID.RQ2,
    title="Лексикографические (WordNet) vs Векторные методы",
    description=(
        "Сравнение онтологического подхода (WordNet) с векторным (SBERT, TF-IDF). "
        "WordNet использует таксономию synset'ов, векторные методы — статистику "
        "совместной встречаемости терминов."
    ),
    hypothesis=(
        "Векторные методы (SBERT, TF-IDF) превосходят лексикографические (WordNet) "
        "на русских данных из-за неполноты RuWordNet и отсутствия морфологического "
        "покрытия для многих научных терминов."
    ),
    contrast_pairs=RQ2_PAIRS,
    success_criteria={
        "sbert": 0.60,   # SBERT значимо лучше WordNet
        "wordnet_lin": 0.20,  # WordNet даёт результат выше случайного
    },
    datasets=("hj-rg", "simlex999_ru"),
)


# =============================================================================
# RQ3: Тематические модели vs Косинусное сходство
# =============================================================================
RQ3_PAIRS = (
    ContrastPair(
        method_a="sbert",
        method_b="rag_centroid",
        rationale="SBERT vs RAG-centroid: прямое сравнение vs retrieval-based",
    ),
    ContrastPair(
        method_a="tfidf",
        method_b="rag_centroid",
        rationale="TF-IDF vs RAG-centroid: корпусные статистики vs retrieval",
    ),
)

RQ3 = ResearchQuestion(
    id=ResearchQuestionID.RQ3,
    title="Retrieval-agnostic vs Косинусное сходство",
    description=(
        "Исследование: может ли retrieval-based подход (RAG-centroid) "
        "использовать информацию из доменных корпусов для улучшения "
        "оценки близости терминов."
    ),
    hypothesis=(
        "Retrieval-agnostic метод (RAG-centroid) не уступает классическому "
        "косинусному сходству эмбеддингов при наличии релевантного корпуса."
    ),
    contrast_pairs=RQ3_PAIRS,
    success_criteria={
        "rag_centroid": 0.50,  # RAG-centroid конкурентоспособен
        "sbert": 0.50,          # SBERT как baseline
    },
    datasets=("hj-rg",),
)


# =============================================================================
# Реестр всех Research Questions
# =============================================================================

ALL_RESEARCH_QUESTIONS: tuple[ResearchQuestion, ...] = (
    RQ1,
    RQ2,
    RQ3,
)

RQ_BY_ID: dict[ResearchQuestionID, ResearchQuestion] = {
    rq.id: rq for rq in ALL_RESEARCH_QUESTIONS
}


def get_research_question(rq_id: str | ResearchQuestionID) -> Optional[ResearchQuestion]:
    """Получить RQ по ID.
    
    Args:
        rq_id: Строка 'rq1', 'rq2', 'rq3' или enum ResearchQuestionID.
    
    Returns:
        ResearchQuestion или None если не найден.
    """
    if isinstance(rq_id, str):
        try:
            rq_id = ResearchQuestionID(rq_id.lower())
        except ValueError:
            return None
    return RQ_BY_ID.get(rq_id)


def get_contrast_pairs_for_method(method: str) -> list[tuple[ResearchQuestion, ContrastPair]]:
    """Получить все контрастные пары, где участвует метод.
    
    Args:
        method: Название метода (sbert, doc2vec, wordnet_lin, etc.)
    
    Returns:
        Список пар (RQ, ContrastPair), где метод участвует.
    """
    result = []
    for rq in ALL_RESEARCH_QUESTIONS:
        for pair in rq.contrast_pairs:
            if pair.method_a == method or pair.method_b == method:
                result.append((rq, pair))
    return result


def get_methods_for_rq(rq_id: str | ResearchQuestionID) -> tuple[str, ...]:
    """Получить все методы, задействованные в RQ.
    
    Args:
        rq_id: ID Research Question.
    
    Returns:
        Кортеж уникальных имён методов.
    """
    rq = get_research_question(rq_id)
    if not rq:
        return ()
    
    methods = set()
    for pair in rq.contrast_pairs:
        methods.add(pair.method_a)
        methods.add(pair.method_b)
    return tuple(sorted(methods))


def format_rq_summary(rq: ResearchQuestion) -> str:
    """Форматировать краткое описание RQ для документации.
    
    Args:
        rq: Research Question.
    
    Returns:
        Markdown-строку с описанием.
    """
    lines = [
        f"### {rq.id.value.upper()}: {rq.title}",
        "",
        f"**Гипотеза:** {rq.hypothesis}",
        "",
        "**Контрастные пары:**",
    ]
    for pair in rq.contrast_pairs:
        lines.append(f"- {pair.name}: {pair.rationale}")
    
    lines.append("")
    lines.append(f"**Датасеты:** {', '.join(rq.datasets)}")
    lines.append("")
    lines.append("**Критерии успеха:**")
    for method, min_score in rq.success_criteria.items():
        lines.append(f"- {method}: spearman ≥ {min_score}")
    
    return "\n".join(lines)
