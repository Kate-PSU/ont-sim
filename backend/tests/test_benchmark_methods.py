# backend/tests/test_benchmark_methods.py
# Тесты для методов бенчмарка (util functions)
#
# Версия: 1.0
# Обновлено: 2026-04-12

"""
Тесты для util функций из run_benchmark_grid.py:
- cosine_similarity()
- euclidean_distance()
- normalize_to_01()
- normalize_distance_to_similarity()
- load_domain_terms()
- extend_corpus_if_needed()
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Добавляем scripts в путь для импорта
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

# Импортируем функции из run_benchmark_grid
from run_benchmark_grid import (
    cosine_similarity,
    euclidean_distance,
    normalize_to_01,
    normalize_distance_to_similarity,
    load_domain_terms,
    extend_corpus_if_needed,
    MIN_CORPUS_SIZE,
)


class TestCosineSimilarity:
    """Тесты функции cosine_similarity()."""

    def test_identical_vectors(self):
        """Тест: идентичные векторы дают similarity = 1."""
        v = np.array([1.0, 2.0, 3.0])
        result = cosine_similarity(v, v)
        assert abs(result - 1.0) < 1e-6

    def test_opposite_vectors(self):
        """Тест: противоположные векторы дают similarity = -1."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([-1.0, 0.0, 0.0])
        result = cosine_similarity(v1, v2)
        assert abs(result + 1.0) < 1e-6

    def test_perpendicular_vectors(self):
        """Тест: перпендикулярные векторы дают similarity = 0."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        result = cosine_similarity(v1, v2)
        assert abs(result) < 1e-6

    def test_zero_vector(self):
        """Тест: нулевой вектор возвращает 0."""
        v1 = np.array([0.0, 0.0])
        v2 = np.array([1.0, 2.0])
        result = cosine_similarity(v1, v2)
        assert result == 0.0

    def test_partial_similarity(self):
        """Тест: частично похожие векторы."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.5, 0.5])
        result = cosine_similarity(v1, v2)
        # cos(45°) ≈ 0.707
        assert abs(result - 0.7071) < 0.01

    def test_2d_rotation(self):
        """Тест: вращение на 60 градусов."""
        angle = np.pi / 3  # 60°
        v1 = np.array([1.0, 0.0])
        v2 = np.array([np.cos(angle), np.sin(angle)])
        result = cosine_similarity(v1, v2)
        assert abs(result - 0.5) < 1e-6


class TestEuclideanDistance:
    """Тесты функции euclidean_distance()."""

    def test_same_point(self):
        """Тест: одинаковые точки дают расстояние 0."""
        v = np.array([1.0, 2.0, 3.0])
        result = euclidean_distance(v, v)
        assert result == 0.0

    def test_unit_distance(self):
        """Тест: единичное расстояние."""
        v1 = np.array([0.0, 0.0])
        v2 = np.array([1.0, 0.0])
        result = euclidean_distance(v1, v2)
        assert abs(result - 1.0) < 1e-6

    def test_2d_distance(self):
        """Тест: евклидово расстояние в 2D."""
        v1 = np.array([0.0, 0.0])
        v2 = np.array([3.0, 4.0])
        result = euclidean_distance(v1, v2)
        # 3-4-5 треугольник
        assert abs(result - 5.0) < 1e-6

    def test_3d_distance(self):
        """Тест: евклидово расстояние в 3D."""
        v1 = np.array([0.0, 0.0, 0.0])
        v2 = np.array([1.0, 2.0, 2.0])
        result = euclidean_distance(v1, v2)
        assert abs(result - 3.0) < 1e-6


class TestNormalizeTo01:
    """Тесты функции normalize_to_01()."""

    def test_min_value(self):
        """Тест: минимальное значение -> 0."""
        result = normalize_to_01(-1.0, min_val=-1.0, max_val=1.0)
        assert result == 0.0

    def test_max_value(self):
        """Тест: максимальное значение -> 1."""
        result = normalize_to_01(1.0, min_val=-1.0, max_val=1.0)
        assert result == 1.0

    def test_mid_value(self):
        """Тест: среднее значение -> 0.5."""
        result = normalize_to_01(0.0, min_val=-1.0, max_val=1.0)
        assert result == 0.5

    def test_default_bounds(self):
        """Тест: границы по умолчанию."""
        result = normalize_to_01(0.5)
        assert result == 0.75  # (0.5 - (-1)) / (1 - (-1)) = 1.5/2 = 0.75

    def test_out_of_range_high(self):
        """Тест: значение выше максимума -> 1."""
        result = normalize_to_01(5.0, min_val=-1.0, max_val=1.0)
        assert result == 1.0

    def test_out_of_range_low(self):
        """Тест: значение ниже минимума -> 0."""
        result = normalize_to_01(-5.0, min_val=-1.0, max_val=1.0)
        assert result == 0.0

    def test_cosine_output_normalization(self):
        """Тест: нормализация выхода cosine similarity [-1, 1] -> [0, 1]."""
        # cosine = 0.7071 -> normalized ≈ 0.8536
        result = normalize_to_01(0.7071, min_val=-1.0, max_val=1.0)
        assert abs(result - 0.8535) < 0.01


class TestNormalizeDistanceToSimilarity:
    """Тесты функции normalize_distance_to_similarity()."""

    def test_zero_distance(self):
        """Тест: нулевое расстояние -> 1 (идеальное сходство)."""
        result = normalize_distance_to_similarity(0.0, max_dist=10.0)
        assert result == 1.0

    def test_max_distance(self):
        """Тест: максимальное расстояние -> 0."""
        result = normalize_distance_to_similarity(10.0, max_dist=10.0)
        assert result == 0.0

    def test_half_distance(self):
        """Тест: половина расстояния -> 0.5."""
        result = normalize_distance_to_similarity(5.0, max_dist=10.0)
        assert result == 0.5

    def test_default_max(self):
        """Тест: граница по умолчанию max_dist=10."""
        result = normalize_distance_to_similarity(5.0)
        assert result == 0.5

    def test_out_of_range(self):
        """Тест: расстояние больше максимума -> 0."""
        result = normalize_distance_to_similarity(20.0, max_dist=10.0)
        assert result == 0.0


class TestLoadDomainTerms:
    """Тесты функции load_domain_terms()."""

    def test_loads_terms(self):
        """Тест: загрузка терминов из terms.csv."""
        terms = load_domain_terms()
        assert len(terms) > 0
        assert isinstance(terms, list)
        assert all(isinstance(t, str) for t in terms)

    def test_caches_terms(self):
        """Тест: повторный вызов возвращает тот же результат."""
        terms1 = load_domain_terms()
        terms2 = load_domain_terms()
        assert terms1 == terms2
        assert len(terms1) > 100  # Ожидаем ~298 терминов

    def test_terms_not_empty(self):
        """Тест: загруженные термины не пустые."""
        terms = load_domain_terms()
        # Фильтруем пустые и whitespace-only
        non_empty = [t for t in terms if t and t.strip()]
        assert len(non_empty) > 0

    def test_at_least_100_terms(self):
        """Тест: минимум 100 терминов в корпусе."""
        terms = load_domain_terms()
        assert len(terms) >= 100, f"Expected >=100 terms, got {len(terms)}"


class TestExtendCorpusIfNeeded:
    """Тесты функции extend_corpus_if_needed()."""

    def test_no_extension_needed(self):
        """Тест: достаточно терминов — расширение не нужно."""
        # Создаём 101 термин
        terms = [f"term_{i}" for i in range(101)]
        extended, was_extended = extend_corpus_if_needed(terms)
        
        assert extended == terms
        assert was_extended is False

    def test_extension_needed(self):
        """Тест: мало терминов — расширяем из terms.csv."""
        # Создаём 50 терминов (меньше MIN_CORPUS_SIZE=100)
        terms = [f"test_term_{i}" for i in range(50)]
        extended, was_extended = extend_corpus_if_needed(terms)
        
        assert len(extended) > len(terms)
        assert was_extended is True
        # Должны добавиться термины из terms.csv
        assert len(extended) >= 100

    def test_force_extension(self):
        """Тест: принудительное расширение (force=True)."""
        terms = [f"term_{i}" for i in range(150)]
        extended, was_extended = extend_corpus_if_needed(terms, force=True)
        
        # Даже если достаточно терминов, force добавляет из terms.csv
        assert len(extended) > len(terms)
        assert was_extended is True

    def test_original_terms_preserved(self):
        """Тест: оригинальные термины сохраняются."""
        original = ["уникальный_термин_123", "другой_уникальный_456"]
        extended, _ = extend_corpus_if_needed(original)
        
        # Оригинальные термины должны быть в начале
        assert extended[:len(original)] == original

    def test_no_duplicates(self):
        """Тест: нет дубликатов в расширенном корпусе."""
        terms = load_domain_terms()[:50]  # Берём из terms.csv
        extended, _ = extend_corpus_if_needed(terms)
        
        # Проверяем уникальность
        assert len(extended) == len(set(extended))

    def test_keeps_order(self):
        """Тест: порядок оригинальных терминов сохранён."""
        terms = ["z термин", "a термин", "m термин"]
        extended, _ = extend_corpus_if_needed(terms)
        
        # Оригинальные термины сохраняют порядок
        assert extended[:3] == terms


class TestBenchmarkConstants:
    """Тесты констант бенчмарка."""

    def test_min_corpus_size(self):
        """Тест: MIN_CORPUS_SIZE = 100."""
        assert MIN_CORPUS_SIZE == 100

    def test_domain_corpus_path_exists(self):
        """Тест: путь к terms.csv существует."""
        from scripts.run_benchmark_grid import DOMAIN_CORPUS_PATH_EN
        # Проверяем что константа существует
        assert DOMAIN_CORPUS_PATH_EN is not None
        # Проверяем что это путь к CSV файлу
        assert str(DOMAIN_CORPUS_PATH_EN).endswith('.csv')
