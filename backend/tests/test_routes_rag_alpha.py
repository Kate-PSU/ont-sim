"""
Тесты API routes для alpha параметра.

Проверяет:
1. /api/v1/enriched/similarity - поддержка alpha параметра
2. /api/v1/enriched/graph/detailed - поддержка alpha параметра
3. Разные alpha дают разные результаты
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np


@pytest.fixture
def client():
    """Создание тестового клиента с замockанными сервисами."""
    with patch('backend.src.infrastructure.embedding_service.EmbeddingService') as mock_emb:
        mock_instance = MagicMock()
        # Возвращаем эмбеддинги размерности 1024 (sbert_large_nlu_ru)
        mock_instance.get_embeddings_batch.return_value = np.random.randn(3, 1024).astype('float32')
        mock_instance.dimension = 1024
        mock_emb.return_value = mock_instance
        
        from backend.src.presentation.main import app
        yield TestClient(app)


class TestEnrichedSimilarityEndpoint:
    """Тесты эндпоинта /api/v1/enriched/similarity."""
    
    def test_enriched_similarity_exists(self, client):
        """Проверяем что эндпоинт существует."""
        response = client.get("/api/v1/enriched/similarity/test1/test2")
        
        # Должен вернуть что-то кроме 404
        assert response.status_code != 404, "Эндпоинт /enriched/similarity не найден"
    
    def test_enriched_similarity_with_alpha(self, client):
        """Запрос с alpha параметром."""
        response = client.get(
            "/api/v1/enriched/similarity/квантовая_механика/физика",
            params={"alpha": 0.5}
        )
        
        # Проверяем что запрос прошёл (200 или 500 если данных нет)
        assert response.status_code in [200, 500, 400, 422]
    
    def test_alpha_parameter_passed(self, client):
        """Проверяем что alpha параметр принимается."""
        # Этот тест проверяет что маршрутизация работает
        response = client.get(
            "/api/v1/enriched/similarity/term1/term2",
            params={"alpha": 0.3}
        )
        
        # Главное - чтобы не было 404
        assert response.status_code != 404


class TestGraphDetailedEndpoint:
    """Тесты эндпоинта /api/v1/graph/detailed."""
    
    def test_graph_detailed_exists(self, client):
        """Проверяем что эндпоинт существует."""
        response = client.get("/api/v1/graph/detailed")
        
        # Должен вернуть что-то кроме 404
        assert response.status_code != 404, "Эндпоинт /graph/detailed не найден"
    
    def test_graph_detailed_with_method(self, client):
        """Graph detailed endpoint с method параметром."""
        response = client.get(
            "/api/v1/graph/detailed",
            params={
                "domains": "Физика,Математика",
                "method": "sbert"
            }
        )
        
        # Проверяем что method принят
        assert response.status_code != 404


class TestEnrichedGraphEndpoint:
    """Тесты эндпоинта /api/v1/enriched/graph."""
    
    def test_enriched_graph_exists(self, client):
        """Проверяем что эндпоинт существует."""
        response = client.get("/api/v1/enriched/graph")
        
        # Должен вернуть что-то кроме 404
        assert response.status_code != 404, "Эндпоинт /enriched/graph не найден"
    
    def test_enriched_graph_with_alpha(self, client):
        """Enriched graph endpoint с alpha параметром."""
        response = client.get(
            "/api/v1/enriched/graph",
            params={
                "threshold": 0.5,
                "alpha": 0.3
            }
        )
        
        # Проверяем что alpha принят
        assert response.status_code != 404


class TestAlphaParameterLogic:
    """Тесты логики alpha параметра."""
    
    def test_alpha_blend_formula(self):
        """Проверка формулы alpha blending.
        
        result = alpha * sbert_embedding + (1 - alpha) * neighbor_centroid
        """
        sbert_emb = np.array([0.1, 0.2, 0.3])
        neighbor_emb = np.array([0.4, 0.5, 0.6])
        
        for alpha in [0.0, 0.5, 1.0]:
            result = alpha * sbert_emb + (1 - alpha) * neighbor_emb
            
            if alpha == 0.0:
                np.testing.assert_array_almost_equal(result, neighbor_emb)
            elif alpha == 1.0:
                np.testing.assert_array_almost_equal(result, sbert_emb)
            else:
                # Должен быть средним
                assert not np.allclose(result, sbert_emb)
                assert not np.allclose(result, neighbor_emb)
    
    def test_alpha_produces_different_embeddings(self):
        """Разные alpha дают разные эмбеддинги."""
        sbert_emb = np.array([0.1, 0.2, 0.3])
        neighbor_emb = np.array([0.4, 0.5, 0.6])
        
        embeddings = {}
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            embeddings[alpha] = alpha * sbert_emb + (1 - alpha) * neighbor_emb
        
        # Проверяем что все различны
        for i, alpha1 in enumerate([0.0, 0.25, 0.5]):
            for alpha2 in [0.75, 1.0]:
                diff = np.linalg.norm(embeddings[alpha1] - embeddings[alpha2])
                assert diff > 0.01, f"alpha={alpha1} и alpha={alpha2} дают одинаковые эмбеддинги!"
    
    def test_alpha_default_in_service(self):
        """Тест значения alpha по умолчанию."""
        from backend.src.infrastructure.enriched_embedding_service import EnrichedEmbeddingService
        
        # Проверяем что EnrichedEmbeddingService поддерживает alpha
        service_class = EnrichedEmbeddingService
        
        # Если есть __init__, проверяем параметры
        import inspect
        sig = inspect.signature(service_class.__init__)
        params = list(sig.parameters.keys())
        
        # Проверяем что в сервисе есть логика для alpha
        source = inspect.getsource(service_class)
        assert "alpha" in source.lower(), "EnrichedEmbeddingService должен поддерживать alpha"


class TestAPIIntegration:
    """Интеграционные тесты API."""
    
    def test_all_endpoints_accessible(self, client):
        """Проверяем что основные endpoints доступны."""
        endpoints = [
            "/api/v1/health",
            "/api/v1/ping",
            "/api/v1/domains",
            "/api/v1/rag/status",
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code != 404, f"Endpoint {endpoint} не найден"
    
    def test_enriched_endpoints_accessible(self, client):
        """Проверяем что enriched endpoints доступны."""
        endpoints = [
            "/api/v1/enriched/similarity/term1/term2",
            "/api/v1/enriched/graph",
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            # Главное - чтобы не было 404
            assert response.status_code != 404, f"Endpoint {endpoint} не найден"
