# backend/tests/test_retrieval_service.py
# Тесты для RetrievalService
#
# Версия: 1.0
# Обновлено: 2026-04-10

"""
Тесты для RetrievalService с FAISS.
"""

import pytest
import numpy as np


class TestRetrievalService:
    """Тесты для RetrievalService."""
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Mock EmbeddingService."""
        class MockEmbeddingService:
            def __init__(self):
                self.call_count = 0
                self.calls = []
            
            def get_embedding(self, term: str) -> np.ndarray:
                self.call_count += 1
                self.calls.append(term)
                # Генерируем детерминированный эмбеддинг на основе хеша термина
                np.random.seed(hash(term) % (2**31))
                return np.random.rand(384).astype('float32')
            
            def get_embedding_dimension(self) -> int:
                return 384
        
        return MockEmbeddingService()
    
    @pytest.fixture
    def retrieval_service(self, mock_embedding_service):
        """Создание RetrievalService с моком."""
        from src.infrastructure.retrieval_service import RetrievalService
        return RetrievalService(mock_embedding_service)
    
    def test_init(self, retrieval_service, mock_embedding_service):
        """Тест инициализации."""
        assert retrieval_service.embedding_service is mock_embedding_service
        assert retrieval_service._index is None
        assert retrieval_service.embeddings_matrix is None
        assert retrieval_service.terms == []
    
    def test_build_index(self, retrieval_service, mock_embedding_service):
        """Тест построения индекса."""
        terms = ["компьютер", "ноутбук", "клавиатура"]
        retrieval_service.build_index(terms)
        
        assert retrieval_service.terms == terms
        assert retrieval_service.embeddings_matrix is not None
        assert retrieval_service._index is not None
        assert retrieval_service.embeddings_matrix.shape[0] == len(terms)
        assert mock_embedding_service.call_count == len(terms)
    
    def test_build_index_normalizes_embeddings(self, retrieval_service, mock_embedding_service):
        """Тест нормализации эмбеддингов."""
        terms = ["word1", "word2"]
        retrieval_service.build_index(terms)
        
        # Проверяем что эмбеддинги нормализованы (L2 = 1)
        norms = np.linalg.norm(retrieval_service.embeddings_matrix, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)
    
    def test_retrieve_neighbors_requires_index(self, retrieval_service):
        """Тест что retrieve_neighbors требует построенного индекса."""
        with pytest.raises(ValueError, match="Index not built"):
            retrieval_service.retrieve_neighbors("test")
    
    def test_retrieve_neighbors_excludes_query(self, retrieval_service, mock_embedding_service):
        """Тест что сам запрос исключается из результатов."""
        terms = ["компьютер", "ноутбук", "программа", "данные", "сеть"]
        
        retrieval_service.build_index(terms)
        neighbors = retrieval_service.retrieve_neighbors("компьютер", k=3)
        
        neighbor_terms = [n[0] for n in neighbors]
        assert "компьютер" not in neighbor_terms
    
    def test_retrieve_neighbors_returns_correct_format(self, retrieval_service, mock_embedding_service):
        """Тест формата возвращаемых данных."""
        terms = ["кот", "собака", "животное", "питомец", "домашний"]
        retrieval_service.build_index(terms)
        
        neighbors = retrieval_service.retrieve_neighbors("кот", k=2)
        
        assert isinstance(neighbors, list)
        assert len(neighbors) <= 2
        for term, score in neighbors:
            assert isinstance(term, str)
            assert isinstance(score, float)
            assert -1.1 <= score <= 1.1  # cosine similarity after normalization
    
    def test_retrieve_neighbors_k_limit(self, retrieval_service, mock_embedding_service):
        """Тест ограничения количества результатов."""
        terms = ["word" + str(i) for i in range(20)]
        retrieval_service.build_index(terms)
        
        neighbors = retrieval_service.retrieve_neighbors("word1", k=5)
        assert len(neighbors) <= 5
    
    def test_get_retrieved_context(self, retrieval_service, mock_embedding_service):
        """Тест получения контекста."""
        terms = ["машина", "автомобиль", "транспорт", "дорога", "двигатель"]
        retrieval_service.build_index(terms)
        
        context = retrieval_service.get_retrieved_context("машина", k=3)
        
        assert isinstance(context, list)
        assert len(context) <= 3
        assert all(isinstance(t, str) for t in context)
        assert "машина" not in context
    
    def test_empty_terms(self, retrieval_service, mock_embedding_service):
        """Тест с пустым списком терминов."""
        retrieval_service.build_index([])
        
        assert retrieval_service.terms == []
        assert retrieval_service._index is None
        assert retrieval_service.embeddings_matrix is None
        
        # Проверяем что retrieve_neighbors падает с пустым списком
        with pytest.raises(ValueError, match="Index not built"):
            retrieval_service.retrieve_neighbors("test", k=2)
