# backend/tests/test_embedding_service.py
# Тесты для сервиса эмбеддингов
#
# Версия: 1.0
# Обновлено: 2026-04-06

"""
Тесты для EmbeddingService.
Использует легковесную модель для тестов.
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from src.infrastructure import EmbeddingService


# Тестовая модель (微型 для скорости)
TEST_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


class TestEmbeddingService:
    """Тесты EmbeddingService."""
    
    def test_init_default_model(self):
        """Тест инициализации с моделью по умолчанию."""
        service = EmbeddingService()
        assert service.model_name == "ai-forever/sbert_large_nlu_ru"
        assert service._model is None  # Lazy loading
    
    def test_init_custom_model(self):
        """Тест инициализации с кастомной моделью."""
        service = EmbeddingService(model_name=TEST_MODEL)
        assert service.model_name == TEST_MODEL
        assert service._model is None
    
    def test_init_with_cache(self):
        """Тест инициализации с кэшем."""
        mock_cache = MagicMock()
        service = EmbeddingService(cache=mock_cache)
        assert service.cache is mock_cache
    
    @patch("src.infrastructure.embedding_service.SentenceTransformer")
    def test_model_lazy_loading(self, mock_st_class):
        """Тест lazy-загрузки модели."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_st_class.return_value = mock_model
        
        service = EmbeddingService(model_name=TEST_MODEL)
        assert service._model is None
        
        # Первое обращение загружает модель
        _ = service.model
        
        mock_st_class.assert_called_once_with(TEST_MODEL)
        assert service._model is not None
    
    @patch("src.infrastructure.embedding_service.SentenceTransformer")
    def test_get_embedding(self, mock_st_class):
        """Тест получения эмбеддинга термина."""
        expected_embedding = np.array([0.1, 0.2, 0.3, 0.4])
        mock_model = MagicMock()
        mock_model.encode.return_value = expected_embedding
        mock_st_class.return_value = mock_model
        
        service = EmbeddingService(model_name=TEST_MODEL)
        result = service.get_embedding("машинное обучение")
        
        mock_model.encode.assert_called_once_with(
            "машинное обучение",
            convert_to_numpy=True
        )
        np.testing.assert_array_equal(result, expected_embedding)
    
    @patch("src.infrastructure.embedding_service.SentenceTransformer")
    def test_get_embedding_dimension(self, mock_st_class):
        """Тест получения размерности эмбеддинга."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_class.return_value = mock_model
        
        service = EmbeddingService(model_name=TEST_MODEL)
        dim = service.get_embedding_dimension()
        
        assert dim == 384
        mock_model.get_sentence_embedding_dimension.assert_called_once()
    
    @patch("src.infrastructure.embedding_service.SentenceTransformer")
    def test_get_embeddings_batch(self, mock_st_class):
        """Тест пакетного получения эмбеддингов."""
        expected_embeddings = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ])
        mock_model = MagicMock()
        mock_model.encode.return_value = expected_embeddings
        mock_st_class.return_value = mock_model
        
        service = EmbeddingService(model_name=TEST_MODEL)
        terms = ["нейронная сеть", "глубокое обучение"]
        result = service.get_embeddings_batch(terms)
        
        mock_model.encode.assert_called_once_with(terms, convert_to_numpy=True)
        np.testing.assert_array_equal(result, expected_embeddings)


class TestEmbeddingServiceWithCache:
    """Тесты EmbeddingService с кэшированием."""
    
    @pytest.mark.asyncio
    async def test_get_embedding_cached_cache_hit(self):
        """Тест кэш-попадания."""
        cached_embedding = np.array([0.1, 0.2, 0.3])
        mock_cache = AsyncMock()
        mock_cache.get_embedding.return_value = cached_embedding
        
        service = EmbeddingService(model_name=TEST_MODEL, cache=mock_cache)
        
        # Кэш работает, модель не должна вызываться
        result = await service.get_embedding_cached("нейронная сеть")
        
        np.testing.assert_array_equal(result, cached_embedding)
        mock_cache.get_embedding.assert_called_once_with("нейронная сеть")
        # set_embedding не должен вызываться при кэш-попадании
    
    @pytest.mark.asyncio
    async def test_get_embedding_cached_cache_miss(self):
        """Тест кэш-промаха."""
        computed_embedding = np.array([0.4, 0.5, 0.6])
        mock_cache = AsyncMock()
        mock_cache.get_embedding.return_value = None
        mock_cache.set_embedding.return_value = None
        
        service = EmbeddingService(model_name=TEST_MODEL, cache=mock_cache)
        
        with patch.object(service, "get_embedding") as mock_get:
            mock_get.return_value = computed_embedding
            result = await service.get_embedding_cached("машинное обучение")
        
        np.testing.assert_array_equal(result, computed_embedding)
        mock_cache.get_embedding.assert_called_once()
        mock_cache.set_embedding.assert_called_once_with(
            "машинное обучение",
            computed_embedding
        )
    
    @pytest.mark.asyncio
    async def test_get_embedding_cached_no_cache(self):
        """Тест без кэша."""
        computed_embedding = np.array([0.7, 0.8, 0.9])
        
        service = EmbeddingService(model_name=TEST_MODEL, cache=None)
        
        with patch.object(service, "get_embedding") as mock_get:
            mock_get.return_value = computed_embedding
            result = await service.get_embedding_cached("термин")
        
        np.testing.assert_array_equal(result, computed_embedding)
        mock_get.assert_called_once_with("термин")


# Примечание: Интеграционные тесты с реальной моделью требуют GPU/CUDA.
# Для их запуска используйте: CUDA_VISIBLE_DEVICES="" python -m pytest ...
# или запустите тесты в Docker с GPU.
