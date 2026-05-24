# backend/tests/test_centroid_service.py
# Тесты для сервиса расчёта центроидов
#
# Версия: 1.0
# Обновлено: 2026-04-06

"""
Тесты для CentroidService.
TDD: сначала тесты, потом код
"""

import numpy as np
import pytest

from src.application.centroid_service import (
    CentroidService,
    CentroidResult,
    calculate_centroid_batch,
    cosine_similarity_between_centroids,
)
from src.domain import Domain, Term


class TestCentroidServiceInit:
    """Тесты инициализации CentroidService."""
    
    def test_init_without_weights(self):
        """Тест: инициализация без весов."""
        service = CentroidService()
        assert service.weights is None
    
    def test_init_with_weights(self):
        """Тест: инициализация с весами."""
        weights = np.array([0.5, 0.5])
        service = CentroidService(weights=weights)
        assert service.weights is not None
        np.testing.assert_array_equal(service.weights, weights)


class TestCalculateCentroid:
    """Тесты расчёта центроида."""
    
    def test_centroid_single_vector(self):
        """Тест: центроид одного вектора."""
        service = CentroidService()
        vector = np.array([1.0, 2.0, 3.0])
        
        centroid = service.calculate_centroid(vector)
        
        np.testing.assert_array_equal(centroid, vector)
    
    def test_centroid_multiple_vectors_simple(self):
        """Тест: центроид без весов."""
        service = CentroidService()
        embeddings = np.array([
            [1.0, 0.0],
            [3.0, 0.0],
        ])
        
        centroid = service.calculate_centroid(embeddings)
        
        np.testing.assert_array_almost_equal(centroid, [2.0, 0.0])
    
    def test_centroid_multiple_vectors_weighted(self):
        """Тест: взвешенный центроид."""
        service = CentroidService()
        embeddings = np.array([
            [1.0, 0.0],
            [3.0, 0.0],
        ])
        weights = np.array([0.75, 0.25])
        
        centroid = service.calculate_centroid(embeddings, weights)
        
        # 0.75 * [1,0] + 0.25 * [3,0] = [0.75 + 0.75, 0] = [1.5, 0]
        np.testing.assert_array_almost_equal(centroid, [1.5, 0.0])
    
    def test_centroid_with_service_weights(self):
        """Тест: использование весов из сервиса."""
        service = CentroidService(weights=np.array([0.8, 0.2]))
        embeddings = np.array([
            [1.0, 0.0],
            [5.0, 0.0],
        ])
        
        centroid = service.calculate_centroid(embeddings)
        
        # 0.8 * [1,0] + 0.2 * [5,0] = [0.8 + 1.0, 0] = [1.8, 0]
        np.testing.assert_array_almost_equal(centroid, [1.8, 0.0])
    
    def test_centroid_empty_raises(self):
        """Тест: пустой массив вызывает ошибку."""
        service = CentroidService()
        embeddings = np.array([])
        
        with pytest.raises(ValueError):
            service.calculate_centroid(embeddings)
    
    def test_centroid_3d_array(self):
        """Тест: центроид для 3D массива."""
        service = CentroidService()
        embeddings = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ])
        
        centroid = service.calculate_centroid(embeddings)
        
        np.testing.assert_array_almost_equal(
            centroid, 
            [4.0, 5.0, 6.0]
        )


class TestCalculateDomainCentroid:
    """Тесты расчёта центроида домена."""
    
    def test_domain_centroid(self):
        """Тест: центроид домена."""
        service = CentroidService()
        domain = Domain(
            name="ML",
            terms=[Term("термин1", "ML"), Term("термин2", "ML")]
        )
        embeddings = np.array([
            [1.0, 0.0],
            [3.0, 0.0],
        ])
        
        result = service.calculate_domain_centroid(domain, embeddings)
        
        assert isinstance(result, CentroidResult)
        assert result.domain == "ML"
        assert result.terms_count == 2
        assert not result.used_weights
        np.testing.assert_array_almost_equal(result.centroid, [2.0, 0.0])
    
    def test_domain_centroid_weighted(self):
        """Тест: взвешенный центроид домена."""
        service = CentroidService()
        domain = Domain(
            name="ML",
            terms=[Term("термин1", "ML"), Term("термин2", "ML")]
        )
        embeddings = np.array([
            [2.0, 0.0],
            [4.0, 0.0],
        ])
        weights = np.array([0.5, 0.5])
        
        result = service.calculate_domain_centroid(domain, embeddings, weights)
        
        assert result.used_weights
        np.testing.assert_array_almost_equal(result.centroid, [3.0, 0.0])


class TestCalculateAllCentroids:
    """Тесты расчёта центроидов для всех доменов."""
    
    def test_all_centroids(self):
        """Тест: центроиды для всех доменов."""
        service = CentroidService()
        domains = [
            Domain("ML", [Term("т1", "ML"), Term("т2", "ML")]),
            Domain("NLP", [Term("т3", "NLP")]),
        ]
        domain_embeddings = {
            "ML": np.array([[1.0, 0.0], [3.0, 0.0]]),
            "NLP": np.array([[5.0, 0.0]]),
        }
        
        centroids = service.calculate_all_centroids(domains, domain_embeddings)
        
        assert len(centroids) == 2
        np.testing.assert_array_almost_equal(centroids["ML"], [2.0, 0.0])
        np.testing.assert_array_almost_equal(centroids["NLP"], [5.0, 0.0])
    
    def test_all_centroids_with_weights(self):
        """Тест: центроиды с весами."""
        service = CentroidService()
        domains = [
            Domain("ML", [Term("т1", "ML"), Term("т2", "ML")]),
        ]
        domain_embeddings = {
            "ML": np.array([[1.0, 0.0], [3.0, 0.0]]),
        }
        domain_weights = {
            "ML": np.array([0.25, 0.75]),
        }
        
        centroids = service.calculate_all_centroids(
            domains, domain_embeddings, domain_weights
        )
        
        # 0.25 * [1,0] + 0.75 * [3,0] = [0.25 + 2.25, 0] = [2.5, 0]
        np.testing.assert_array_almost_equal(centroids["ML"], [2.5, 0.0])


class TestCalculateCentroidBatch:
    """Тесты пакетного расчёта центроидов."""
    
    def test_batch_single(self):
        """Тест: пакет с одним массивом."""
        centroids = calculate_centroid_batch([
            np.array([[1.0, 0.0], [3.0, 0.0]])
        ])
        
        assert len(centroids) == 1
        np.testing.assert_array_almost_equal(centroids[0], [2.0, 0.0])
    
    def test_batch_multiple(self):
        """Тест: пакет с несколькими массивами."""
        centroids = calculate_centroid_batch([
            np.array([[1.0, 0.0], [3.0, 0.0]]),
            np.array([[5.0, 0.0]]),
        ])
        
        assert len(centroids) == 2
        np.testing.assert_array_almost_equal(centroids[0], [2.0, 0.0])
        np.testing.assert_array_almost_equal(centroids[1], [5.0, 0.0])
    
    def test_batch_with_weights(self):
        """Тест: пакет с весами."""
        centroids = calculate_centroid_batch(
            [np.array([[1.0, 0.0], [3.0, 0.0]])],
            [np.array([0.3, 0.7])]
        )
        
        np.testing.assert_array_almost_equal(centroids[0], [2.4, 0.0])


class TestCosineSimilarityBetweenCentroids:
    """Тесты расчёта близости между центроидами."""
    
    def test_similarity_between_centroids(self):
        """Тест: попарная близость."""
        centroids = {
            "ML": np.array([1.0, 0.0]),
            "NLP": np.array([1.0, 0.0]),
            "BIO": np.array([-1.0, 0.0]),
        }
        
        similarities = cosine_similarity_between_centroids(centroids)
        
        assert len(similarities) == 3  # 3 пары
        # ML и NLP должны быть близки (1.0)
        assert similarities[("ML", "NLP")] == pytest.approx(1.0)
        # ML и BIO должны быть далеки (-1.0)
        assert similarities[("ML", "BIO")] == pytest.approx(-1.0)
    
    def test_empty_centroids(self):
        """Тест: пустой словарь."""
        similarities = cosine_similarity_between_centroids({})
        assert similarities == {}
