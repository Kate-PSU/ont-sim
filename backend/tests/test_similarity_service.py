# backend/tests/test_similarity_service.py
# Тесты для SimilarityService
#
# Версия: 1.0
# Обновлено: 2026-04-06

"""
Модульные тесты для сервиса расчёта близости доменов.
"""

import numpy as np
import pytest

from src.application.similarity_service import SimilarityService
from src.domain import Domain, GraphData


class TestSimilarityService:
    """Тесты для SimilarityService."""

    def test_cosine_similarity_identical_vectors(self, sample_vectors):
        """Тест: идентичные векторы имеют близость 1.0."""
        service = SimilarityService(metric="cosine")
        result = service.calculate_similarity(
            sample_vectors["vec_a"],
            sample_vectors["vec_b"],
        )
        assert result == pytest.approx(1.0, abs=1e-5)

    def test_cosine_similarity_orthogonal_vectors(self, sample_vectors):
        """Тест: ортогональные векторы имеют близость 0.0."""
        service = SimilarityService(metric="cosine")
        result = service.calculate_similarity(
            sample_vectors["vec_a"],
            sample_vectors["vec_c"],
        )
        assert result == pytest.approx(0.0, abs=1e-5)

    def test_cosine_similarity_opposite_vectors(self, sample_vectors):
        """Тест: противоположные векторы имеют близость -1.0."""
        service = SimilarityService(metric="cosine")
        result = service.calculate_similarity(
            sample_vectors["vec_a"],
            sample_vectors["vec_d"],
        )
        assert result == pytest.approx(-1.0, abs=1e-5)

    def test_cosine_similarity_zero_vector(self, sample_vectors):
        """Тест: нулевой вектор возвращает 0.0."""
        service = SimilarityService(metric="cosine")
        zero_vec = np.zeros(3)
        result = service.calculate_similarity(
            sample_vectors["vec_a"],
            zero_vec,
        )
        assert result == 0.0

    def test_cosine_similarity_both_zero_vectors(self):
        """Тест: оба нулевых вектора возвращают 0.0."""
        service = SimilarityService(metric="cosine")
        zero_vec = np.zeros(3)
        result = service.calculate_similarity(zero_vec, zero_vec)
        assert result == 0.0

    def test_euclidean_similarity_identical_vectors(self, sample_vectors):
        """Тест: идентичные векторы имеют близость 1.0."""
        service = SimilarityService(metric="euclidean")
        result = service.calculate_similarity(
            sample_vectors["vec_a"],
            sample_vectors["vec_b"],
        )
        assert result == pytest.approx(1.0, abs=1e-5)

    def test_euclidean_similarity_distant_vectors(self, sample_vectors):
        """Тест: далёкие векторы имеют близость близкую к 0.0."""
        service = SimilarityService(metric="euclidean")
        result = service.calculate_similarity(
            sample_vectors["vec_a"],
            sample_vectors["vec_d"],
        )
        assert 0.3 <= result <= 0.5  # расстояние = 2, близость = 1/3

    def test_euclidean_similarity_zero_vector(self, sample_vectors):
        """Тест: сравнение не-нулевого вектора с нулевым возвращает 0.0."""
        service = SimilarityService(metric="euclidean")
        zero_vec = np.zeros(3)
        result = service.calculate_similarity(
            sample_vectors["vec_a"],
            zero_vec,
        )
        # При сравнении с нулевым вектором возвращаем 0.0
        assert result == 0.0
    
    def test_euclidean_similarity_both_zero_vectors(self):
        """Тест: оба нулевых вектора возвращают 1.0."""
        service = SimilarityService(metric="euclidean")
        zero_vec = np.zeros(3)
        result = service.calculate_similarity(zero_vec, zero_vec)
        assert result == 1.0

    def test_invalid_metric_raises_error(self, sample_vectors):
        """Тест: некорректная метрика вызывает ValueError."""
        service = SimilarityService(metric="invalid")
        with pytest.raises(ValueError, match="Неизвестная метрика"):
            service.calculate_similarity(
                sample_vectors["vec_a"],
                sample_vectors["vec_b"],
            )

    def test_build_graph_no_edges(self, sample_centroids):
        """Тест: граф не содержит рёбер при высоком пороге."""
        service = SimilarityService(metric="cosine")
        domains = [
            Domain(name="mathematics", terms=[]),
            Domain(name="physics", terms=[]),
            Domain(name="literature", terms=[]),
        ]
        result = service.build_graph(domains, sample_centroids, threshold=0.99)
        
        assert len(result.nodes) == 3
        assert len(result.edges) == 0

    def test_build_graph_with_edges(self, sample_centroids):
        """Тест: граф содержит рёбра при низком пороге."""
        service = SimilarityService(metric="cosine")
        domains = [
            Domain(name="mathematics", terms=[]),
            Domain(name="physics", terms=[]),
            Domain(name="literature", terms=[]),
        ]
        result = service.build_graph(domains, sample_centroids, threshold=0.0)
        
        assert len(result.nodes) == 3
        # math-physics должны быть похожи, math-literature различны
        assert len(result.edges) >= 1

    def test_build_graph_creates_correct_node_format(self):
        """Тест: узлы создаются в правильном формате."""
        service = SimilarityService(metric="cosine")
        domains = [
            Domain(name="domain1", terms=[]),
        ]
        centroids = {"domain1": np.array([1.0, 0.0])}
        result = service.build_graph(domains, centroids, threshold=0.5)
        
        assert len(result.nodes) == 1
        assert result.nodes[0]["id"] == "domain1"
        assert result.nodes[0]["label"] == "domain1"


class TestSimilarityMatrix:
    """Тесты расчёта матрицы близости."""
    
    def test_similarity_matrix_single_domain(self):
        """Тест: матрица для одного домена."""
        service = SimilarityService(metric="cosine")
        centroids = {"ML": np.array([1.0, 0.0])}
        
        domains, matrix = service.calculate_similarity_matrix(centroids)
        
        assert len(domains) == 1
        assert matrix.shape == (1, 1)
        assert matrix[0][0] == 1.0
    
    def test_similarity_matrix_multiple_domains(self):
        """Тест: матрица для нескольких доменов."""
        service = SimilarityService(metric="cosine")
        centroids = {
            "ML": np.array([1.0, 0.0]),
            "NLP": np.array([1.0, 0.0]),
            "BIO": np.array([-1.0, 0.0]),
        }
        
        domains, matrix = service.calculate_similarity_matrix(centroids)
        
        assert len(domains) == 3
        assert matrix.shape == (3, 3)
        # Диагональ = 1
        assert matrix[0][0] == 1.0
        assert matrix[1][1] == 1.0
        assert matrix[2][2] == 1.0
        # ML-NLP похожи (1.0)
        # ML-BIO различны (-1.0)
    
    def test_similarity_matrix_symmetric(self):
        """Тест: матрица симметрична."""
        service = SimilarityService(metric="cosine")
        centroids = {
            "ML": np.array([1.0, 0.0]),
            "NLP": np.array([0.5, 0.5]),
        }
        
        _, matrix = service.calculate_similarity_matrix(centroids)
        
        assert matrix[0][1] == matrix[1][0]
    
    def test_similarity_matrix_empty(self):
        """Тест: пустой словарь."""
        service = SimilarityService(metric="cosine")
        
        domains, matrix = service.calculate_similarity_matrix({})
        
        assert domains == []
        assert matrix.size == 0


class TestSimilarityPairs:
    """Тесты получения пар близости."""
    
    def test_get_pairs_all_above_threshold(self):
        """Тест: пары с положительной близостью."""
        service = SimilarityService(metric="cosine")
        centroids = {
            "ML": np.array([1.0, 0.0]),
            "NLP": np.array([1.0, 0.0]),
            "BIO": np.array([-1.0, 0.0]),
        }
        
        # threshold=0.0: только ML-NLP (1.0) проходит, ML-BIO (-1.0) и NLP-BIO (-1.0) нет
        pairs = service.get_similarity_pairs(centroids, threshold=0.0)
        
        # Только 1 пара проходит threshold=0.0
        assert len(pairs) == 1
        assert pairs[0]["domain1"] == "ML"
        assert pairs[0]["domain2"] == "NLP"
    
    def test_get_pairs_filtered_by_threshold(self):
        """Тест: фильтрация по порогу."""
        service = SimilarityService(metric="cosine")
        centroids = {
            "ML": np.array([1.0, 0.0]),
            "NLP": np.array([1.0, 0.0]),
            "BIO": np.array([-1.0, 0.0]),
        }
        
        pairs = service.get_similarity_pairs(centroids, threshold=0.5)
        
        # ML-NLP (~1.0) проходят, ML-BIO (-1.0) нет
        assert len(pairs) >= 1
        for p in pairs:
            assert p["score"] >= 0.5
    
    def test_get_pairs_sorted_by_score(self):
        """Тест: пары отсортированы по score."""
        service = SimilarityService(metric="cosine")
        centroids = {
            "A": np.array([1.0, 0.0]),
            "B": np.array([0.0, 1.0]),
            "C": np.array([-1.0, 0.0]),
        }
        
        pairs = service.get_similarity_pairs(centroids, threshold=-1.0)
        
        for i in range(len(pairs) - 1):
            assert pairs[i]["score"] >= pairs[i + 1]["score"]
    
    def test_get_pairs_top_n(self):
        """Тест: ограничение top-N."""
        service = SimilarityService(metric="cosine")
        centroids = {
            "A": np.array([1.0, 0.0]),
            "B": np.array([1.0, 0.0]),
            "C": np.array([0.5, 0.5]),
            "D": np.array([-1.0, 0.0]),
        }
        
        pairs = service.get_similarity_pairs(centroids, threshold=-1.0, top_n=2)
        
        assert len(pairs) == 2
