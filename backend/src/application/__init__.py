# backend/src/application/__init__.py
# Слой приложения (Use Cases / Services)
#
# Версия: 1.0
# Обновлено: 2026-04-06

"""
Слой приложения содержит:
- Use Cases — бизнес-операции
- Services — высокоуровневые сервисы
"""

from .centroid_service import CentroidService
from .similarity_service import SimilarityService
from .benchmark_service import BenchmarkService, BenchmarkPair, MethodResult, BenchmarkComparison
from .research_questions import (
    ResearchQuestion,
    ResearchQuestionID,
    ContrastPair,
    ALL_RESEARCH_QUESTIONS,
    RQ1, RQ2, RQ3,
    get_research_question,
    get_contrast_pairs_for_method,
    get_methods_for_rq,
    format_rq_summary,
)

__all__ = [
    "CentroidService",
    "SimilarityService",
    "BenchmarkService",
    # Research Questions
    "ResearchQuestion",
    "ResearchQuestionID",
    "ContrastPair",
    "ALL_RESEARCH_QUESTIONS",
    "RQ1",
    "RQ2",
    "RQ3",
    "get_research_question",
    "get_contrast_pairs_for_method",
    "get_methods_for_rq",
    "format_rq_summary",
]
