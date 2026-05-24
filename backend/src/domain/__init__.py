# backend/src/domain/__init__.py
# Доменный слой (Clean Architecture)
#
# Версия: 1.0
# Обновлено: 2026-04-06

"""
Доменный слой содержит:
- Сущности (Entity) — Термин, Домен, Близость
- Value Objects — Вектор, Оценка близости
- Domain Services — Логика расчёта близости
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Term:
    """Термин из предметной области.
    
    Атрибуты:
        name: Название термина.
        domain: Домен, к которому относится термин.
        frequency: Частота встречаемости в корпусе.
    """
    name: str
    domain: str
    frequency: int = 1


@dataclass
class Domain:
    """Предметная область.
    
    Атрибуты:
        name: Название домена.
        terms: Список терминов в домене.
    """
    name: str
    terms: list[Term]


@dataclass
class Similarity:
    """Оценка близости между доменами.
    
    Атрибуты:
        domain1: Первый домен.
        domain2: Второй домен.
        score: Значение близости [0, 1].
        metric: Используемая метрика.
    """
    domain1: str
    domain2: str
    score: float
    metric: str = "cosine"


@dataclass
class GraphData:
    """Данные графа для визуализации.
    
    Атрибуты:
        nodes: Список узлов (доменов).
        edges: Список рёбер (близостей).
    """
    nodes: list[dict]
    edges: list[dict]
