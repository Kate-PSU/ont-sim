# backend/tests/test_similarity_api.py
# Тесты для API эндпоинта /domains/{d1}/similarity/{d2}
#
# Версия: 1.0
# Обновлено: 2026-04-06

"""
Интеграционные тесты для эндпоинта расчёта близости доменов.
"""

import warnings
import pytest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.presentation.routes import router
from src.infrastructure.data_loader import CSVDataLoader
from src.infrastructure.cache_manager import CacheManager
from src.domain import Domain, Term


class MockCache:
    """Mock для Redis кэша."""
    
    def __init__(self):
        self._embeddings = {}
        self._centroids = {}
        self._similarity = {}
    
    async def get_embedding(self, term: str):
        return self._embeddings.get(term)
    
    async def set_embedding(self, term: str, embedding: np.ndarray):
        self._embeddings[term] = embedding
    
    async def get_centroid(self, domain: str):
        return self._centroids.get(domain)
    
    async def set_centroid(self, domain: str, centroid: np.ndarray):
        self._centroids[domain] = centroid
    
    async def get_similarity(self, domain1: str, domain2: str):
        key = f"{domain1}:{domain2}"
        return self._similarity.get(key)
    
    async def set_similarity(self, domain1: str, domain2: str, score: float):
        key1 = f"{domain1}:{domain2}"
        key2 = f"{domain2}:{domain1}"
        self._similarity[key1] = score
        self._similarity[key2] = score


@pytest.fixture
def mock_data_loader():
    """Фикстура с тестовыми данными."""
    loader = CSVDataLoader()
    csv_content = """term,domain,frequency
машинное обучение,ML,10
нейронная сеть,ML,8
глубокое обучение,ML,12
биология,BIO,15
генетика,BIO,10
"""
    loader.load_string(csv_content)
    return loader


@pytest.fixture
def mock_cache():
    """Фикстура mock кэша."""
    return MockCache()


@pytest.fixture
def client(mock_data_loader, mock_cache):
    """Фикстура тестового клиента."""
    @asynccontextmanager
    async def mock_lifespan(app: FastAPI):
        app.state.cache = mock_cache
        app.state.data_loader = mock_data_loader
        yield
        await mock_cache.disconnect() if hasattr(mock_cache, 'disconnect') else None
    
    # Создаём тестовое приложение с mock lifespan
    test_app = FastAPI(lifespan=mock_lifespan)
    test_app.include_router(router, prefix="/api/v1")
    
    # Отключаем httpx DeprecationWarning для TestClient
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="httpx")
        with TestClient(test_app) as client:
            yield client


class TestSimilarityEndpoint:
    """Тесты эндпоинта /domains/{d1}/similarity/{d2}."""
    
    def test_get_domains(self, client):
        """Тест получения списка доменов."""
        response = client.get("/api/v1/domains")
        assert response.status_code == 200
        data = response.json()
        assert "domains" in data
        assert "ML" in data["domains"]
        assert "BIO" in data["domains"]
    
    def test_similarity_same_domain_returns_400(self, client):
        """Тест: одинаковые домены возвращают 400."""
        response = client.get("/api/v1/domains/ML/similarity/ML")
        assert response.status_code == 400
        assert "разными" in response.json()["detail"].lower() or "1.0" in response.json()["detail"]
    
    def test_similarity_domain_not_found(self, client):
        """Тест: несуществующий домен возвращает 404."""
        response = client.get("/api/v1/domains/ML/similarity/UNKNOWN")
        assert response.status_code == 404


