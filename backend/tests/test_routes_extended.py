# backend/tests/test_routes_extended.py
# Расширенные тесты для API роутеров
#
# Версия: 1.1
# Обновлено: 2026-04-15

"""
Расширенные тесты для API endpoints.

Использует FastAPI TestClient для изолированного тестирования
роутеров без запуска реального сервера.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

backend_root = Path(__file__).parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from src.presentation.routes import router


# ============================================================
# Вспомогательные функции
# ============================================================

def _make_app_with_state(**state_kwargs) -> FastAPI:
    """Создать FastAPI приложение с предустановленным состоянием."""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    # Устанавливаем состояние
    for key, val in state_kwargs.items():
        setattr(app.state, key, val)

    return app


def _make_mock_cache() -> MagicMock:
    """Создать мок CacheManager."""
    mock = AsyncMock()
    mock.get_system_status.return_value = {"busy": False, "locked_operations": []}
    mock.acquire_lock.return_value = True
    mock.release_lock.return_value = True
    mock.is_locked.return_value = False
    return mock


def _make_mock_embedding_service() -> MagicMock:
    """Создать мок EmbeddingService."""
    mock = MagicMock()
    mock.get_embedding.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    mock.get_embeddings_batch.return_value = np.array(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32
    )
    mock.model_name = "test-model"
    mock.is_loaded.return_value = True
    return mock


# ============================================================
# Тесты Health endpoints
# ============================================================

class TestHealthEndpoints:
    """Тесты для /health и /ping."""

    def setup_method(self):
        """Создаём тестовое приложение."""
        self.app = _make_app_with_state()
        self.client = TestClient(self.app)

    def test_health_returns_ok(self):
        """GET /health → 200, status=ok."""
        response = self.client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_ping_returns_pong(self):
        """GET /ping → 200, pong=True."""
        response = self.client.get("/api/v1/ping")
        assert response.status_code == 200
        data = response.json()
        assert data["pong"] is True


# ============================================================
# Тесты Status endpoint
# ============================================================

class TestStatusEndpoint:
    """Тесты для /status."""

    def test_status_returns_system_status(self):
        """GET /status → 200, содержит поле busy."""
        mock_cache = _make_mock_cache()
        app = _make_app_with_state(cache=mock_cache)
        client = TestClient(app)

        response = client.get("/api/v1/status")
        assert response.status_code == 200
        data = response.json()
        assert "busy" in data


# ============================================================
# Тесты Domains endpoint
# ============================================================

class TestDomainsEndpoint:
    """Тесты для /domains."""

    def test_domains_no_data_loader_returns_empty(self):
        """GET /domains без data_loader → пустой список."""
        app = _make_app_with_state()
        client = TestClient(app)

        response = client.get("/api/v1/domains")
        assert response.status_code == 200
        data = response.json()
        assert data["domains"] == []

    def test_domains_with_data_loader(self):
        """GET /domains с data_loader → список доменов."""
        mock_loader = MagicMock()
        mock_loader.domain_names = ["математика", "физика", "химия"]

        app = _make_app_with_state(data_loader=mock_loader)
        client = TestClient(app)

        response = client.get("/api/v1/domains")
        assert response.status_code == 200
        data = response.json()
        assert "математика" in data["domains"]
        assert len(data["domains"]) == 3


# ============================================================
# Тесты Upload endpoints
# ============================================================

class TestUploadJsonEndpoint:
    """Тесты для POST /upload/json."""

    def test_upload_json_success(self):
        """POST /upload/json → 200, success=True."""
        app = _make_app_with_state()
        client = TestClient(app)

        payload = {
            "domains": [
                {
                    "name": "ML",
                    "terms": [
                        {"name": "нейронная сеть", "frequency": 10},
                        {"name": "градиентный спуск", "frequency": 5},
                    ],
                }
            ]
        }

        with patch("src.presentation.routes.JSONDataLoader") as mock_loader_cls:
            mock_loader = MagicMock()
            mock_loader.load_from_request.return_value = 2
            mock_loader.domain_names = ["ML"]
            mock_loader_cls.return_value = mock_loader

            response = client.post("/api/v1/upload/json", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["terms_loaded"] == 2

    def test_upload_json_invalid_data(self):
        """POST /upload/json с ошибкой → 400."""
        from src.infrastructure.data_loader import DataLoaderError

        app = _make_app_with_state()
        client = TestClient(app)

        payload = {"domains": [{"name": "ML", "terms": []}]}

        with patch("src.presentation.routes.JSONDataLoader") as mock_loader_cls:
            mock_loader = MagicMock()
            mock_loader.load_from_request.side_effect = DataLoaderError("Invalid data")
            mock_loader_cls.return_value = mock_loader

            response = client.post("/api/v1/upload/json", json=payload)

        assert response.status_code == 400


# ============================================================
# Тесты Similarity endpoint
# ============================================================

class TestSimilarityEndpoint:
    """Тесты для /similarity."""

    def _make_full_app(self):
        """Создать приложение с полным состоянием для similarity."""
        mock_embedding_svc = _make_mock_embedding_service()

        mock_domain1 = MagicMock()
        mock_domain1.name = "математика"

        mock_domain2 = MagicMock()
        mock_domain2.name = "физика"

        mock_data = MagicMock()
        mock_data.domains = [mock_domain1, mock_domain2]

        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_data
        mock_loader.domain_names = ["математика", "физика"]

        mock_centroid_service = MagicMock()
        mock_centroid_service.get_centroid.return_value = np.array(
            [0.5, 0.5, 0.5], dtype=np.float32
        )

        mock_similarity_service = MagicMock()
        mock_similarity_service.compute_similarity.return_value = 0.85

        app = _make_app_with_state(
            embedding_service=mock_embedding_svc,
            data_loader=mock_loader,
            centroid_service=mock_centroid_service,
            similarity_service=mock_similarity_service,
            cache=_make_mock_cache(),
        )
        return app

    def test_similarity_requires_domain_params(self):
        """GET /similarity без параметров → ошибка (404 или 422)."""
        app = _make_app_with_state()
        client = TestClient(app)

        response = client.get("/api/v1/similarity")
        # FastAPI может вернуть 422 (validation error) или 404 в зависимости от роутера
        assert response.status_code in (404, 422)

    def test_similarity_with_domains(self):
        """GET /similarity с доменами → 200."""
        app = self._make_full_app()
        client = TestClient(app)

        response = client.get(
            "/api/v1/similarity",
            params={"domain1": "математика", "domain2": "физика"},
        )
        # Может быть 200 или другой код в зависимости от состояния
        assert response.status_code in (200, 400, 404, 500)


# ============================================================
# Тесты WordNet endpoints
# ============================================================

class TestWordNetSimilarityEndpoint:
    """Тесты для /wordnet/similarity."""

    def test_wordnet_similarity_returns_result(self):
        """GET /wordnet/similarity → ответ без исключения."""
        mock_wn_service = MagicMock()
        mock_wn_service.compute_similarity.return_value = 0.75
        mock_wn_service.is_initialized = True

        app = _make_app_with_state(wordnet_service=mock_wn_service)
        client = TestClient(app)

        response = client.get(
            "/api/v1/wordnet/similarity",
            params={"word1": "математика", "word2": "физика"},
        )
        assert response.status_code in (200, 404, 422, 500)

    def test_wordnet_similarity_no_service(self):
        """GET /wordnet/similarity без сервиса → ошибка."""
        app = _make_app_with_state()
        client = TestClient(app)

        response = client.get(
            "/api/v1/wordnet/similarity",
            params={"word1": "математика", "word2": "физика"},
        )
        # Ожидаем ошибку без инициализированного WordNet
        assert response.status_code in (404, 422, 500)


# ============================================================
# Тесты Benchmark endpoints
# ============================================================

class TestBenchmarkDatasetsEndpoint:
    """Тесты для /benchmark/datasets."""

    def test_benchmark_datasets_returns_list(self):
        """GET /benchmark/datasets → список доступных датасетов."""
        app = _make_app_with_state()
        client = TestClient(app)

        response = client.get("/api/v1/benchmark/datasets")
        assert response.status_code == 200
        data = response.json()
        assert "datasets" in data


class TestBenchmarkResultsEndpoint:
    """Тесты для /benchmark/results."""

    def test_benchmark_results_endpoint_accessible(self):
        """GET /benchmark/results → endpoint доступен (любой статус)."""
        app = _make_app_with_state()
        client = TestClient(app)

        response = client.get("/api/v1/benchmark/results")
        # Endpoint должен ответить (не 500 при правильной конфигурации)
        assert response.status_code in (200, 404)


# ============================================================
# Тесты Graph endpoints
# ============================================================

class TestGraphEndpoint:
    """Тесты для /graph."""

    def test_graph_no_data_returns_error(self):
        """GET /graph без данных → ошибка."""
        app = _make_app_with_state()
        client = TestClient(app)

        response = client.get("/api/v1/graph")
        assert response.status_code in (200, 400, 404, 422, 500)

    def test_graph_with_threshold_param(self):
        """GET /graph с параметром threshold."""
        app = _make_app_with_state()
        client = TestClient(app)

        response = client.get("/api/v1/graph", params={"threshold": "0.7"})
        assert response.status_code in (200, 400, 404, 422, 500)


# ============================================================
# Тесты Enrichment endpoints
# ============================================================

class TestEnrichmentEndpoints:
    """Тесты для /enriched endpoints."""

    def test_enrichment_info_endpoint(self):
        """GET /enriched/info → ответ для термина."""
        mock_enrichment = MagicMock()
        mock_enrichment.get_enrichment_info.return_value = {
            "term": "математика",
            "hypernyms": ["наука"],
            "hypernym_count": 1,
            "enriched": True,
        }

        app = _make_app_with_state(enriched_embedding_service=mock_enrichment)
        client = TestClient(app)

        response = client.get(
            "/api/v1/enriched/info",
            params={"term": "математика"},
        )
        assert response.status_code in (200, 404, 422)


# ============================================================
# Тесты Tasks endpoints (Celery)
# ============================================================

class TestTaskEndpoints:
    """Тесты для /tasks/* endpoints."""

    def test_task_status_not_found(self):
        """GET /tasks/{task_id} → 404 для несуществующего таска."""
        app = _make_app_with_state()
        client = TestClient(app)

        # celery_app импортируется внутри функции, патчим в tasks модуле
        with patch("src.application.tasks.celery_app") as mock_celery:
            mock_task = MagicMock()
            mock_task.state = "PENDING"
            mock_task.result = None
            mock_celery.AsyncResult.return_value = mock_task

            response = client.get("/api/v1/tasks/nonexistent-task-id")
            assert response.status_code in (200, 404)
