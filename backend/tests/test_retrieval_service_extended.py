# backend/tests/test_retrieval_service_extended.py
# Расширенные тесты для RetrievalService
#
# Версия: 1.0
# Обновлено: 2026-04-15

"""
Расширенные тесты для RetrievalService — RAG retrieval с FAISS.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.infrastructure.retrieval_service import (
    RetrievalService,
    get_cached_retrieval,
    set_cached_retrieval,
)


class TestRetrievalServiceInit:
    """Тесты инициализации RetrievalService."""

    def test_init_with_embedding_service(self):
        """Тест: инициализация с embedding service."""
        mock_emb = MagicMock()
        service = RetrievalService(mock_emb)
        
        assert service.embedding_service is mock_emb
        assert service._index is None
        assert service.embeddings_matrix is None
        assert service.terms == []

    def test_index_property(self):
        """Тест: property index возвращает _index."""
        mock_emb = MagicMock()
        service = RetrievalService(mock_emb)
        
        # Index должен быть None по умолчанию
        assert service.index is None


class TestRetrievalServiceBuildIndex:
    """Тесты построения индекса."""

    def test_build_index_empty_terms(self):
        """Тест: построение индекса с пустым списком терминов."""
        mock_emb = MagicMock()
        service = RetrievalService(mock_emb)
        
        service.build_index([])
        
        assert service._index is None
        assert service.embeddings_matrix is None

    def test_build_index_optimization_skips_same_terms(self):
        """Тест: оптимизация — не перестраиваем если термины те же."""
        mock_emb = MagicMock()
        mock_emb.get_embedding.return_value = np.array([1.0, 2.0, 3.0])
        
        service = RetrievalService(mock_emb)
        terms = ["term1", "term2"]
        
        # Первый вызов — строим индекс
        service.build_index(terms)
        assert service._index is not None
        
        # Второй вызов с теми же терминами — пропускаем
        service.build_index(terms)
        # Индекс не должен измениться
        assert service._index is not None

    def test_build_index_normalizes_embeddings(self):
        """Тест: эмбеддинги нормализуются для cosine similarity."""
        mock_emb = MagicMock()
        # Возвращаем ненормализованный вектор
        mock_emb.get_embedding.return_value = np.array([1.0, 2.0, 3.0])
        
        service = RetrievalService(mock_emb)
        service.build_index(["term1"])
        
        # Проверяем что эмбеддинги нормализованы
        assert service.embeddings_matrix is not None
        # L2 норма должна быть 1.0 для cosine similarity
        norms = np.linalg.norm(service.embeddings_matrix, axis=1)
        assert np.allclose(norms, 1.0)


class TestRetrievalServiceRetrieve:
    """Тесты поиска соседей."""

    def test_retrieve_neighbors_without_index(self):
        """Тест: ошибка если индекс не построен."""
        mock_emb = MagicMock()
        service = RetrievalService(mock_emb)
        
        with pytest.raises(ValueError, match="Index not built"):
            service.retrieve_neighbors("term")

    def test_retrieve_neighbors_excludes_query(self):
        """Тест: retrieved не включает сам запрос."""
        mock_emb = MagicMock()
        # Все эмбеддинги одинаковые
        mock_emb.get_embedding.return_value = np.array([1.0, 0.0, 0.0])
        
        service = RetrievalService(mock_emb)
        service.build_index(["term1", "term2", "term3"])
        
        neighbors = service.retrieve_neighbors("term1", k=2)
        
        # Ищем term1 в результатах
        terms = [n[0] for n in neighbors]
        assert "term1" not in terms

    def test_retrieve_neighbors_k_plus_one(self):
        """Тест: поиск k+1 чтобы исключить сам запрос."""
        mock_emb = MagicMock()
        mock_emb.get_embedding.return_value = np.array([1.0, 0.0, 0.0])
        
        service = RetrievalService(mock_emb)
        # Все термины одинаковые — ближайший сам запрос
        service.build_index(["term1", "term2"])
        
        # Запрашиваем 1 соседа, получаем 2 (+1 для исключения)
        neighbors = service.retrieve_neighbors("term1", k=1)
        
        assert len(neighbors) <= 1


class TestRetrievalServiceContext:
    """Тесты получения контекста."""

    def test_get_retrieved_context(self):
        """Тест: получение контекста retrieved терминов."""
        mock_emb = MagicMock()
        mock_emb.get_embedding.return_value = np.array([1.0, 0.0, 0.0])
        
        service = RetrievalService(mock_emb)
        service.build_index(["term1", "term2", "term3", "term4", "term5"])
        
        context = service.get_retrieved_context("term1", k=3)
        
        assert isinstance(context, list)
        assert len(context) <= 3
        assert "term1" not in context


class TestRetrievalServiceCache:
    """Тесты кэширования."""

    def test_get_cached_retrieval_not_exists(self):
        """Тест: получение несуществующего кэша."""
        result = get_cached_retrieval("nonexistent_key")
        assert result is None

    def test_set_and_get_cached_retrieval(self):
        """Тест: сохранение и получение из кэша."""
        mock_emb = MagicMock()
        service = RetrievalService(mock_emb)
        
        set_cached_retrieval("test_key", service)
        result = get_cached_retrieval("test_key")
        
        assert result is service


class TestRetrievalServiceSaveLoad:
    """Тесты сохранения и загрузки индекса."""

    def test_save_index_without_build(self):
        """Тест: ошибка при сохранении без построения индекса."""
        mock_emb = MagicMock()
        service = RetrievalService(mock_emb)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Index not built"):
                service.save_index(Path(tmpdir) / "test.index")

    def test_get_index_path(self):
        """Тест: получение пути для индекса."""
        cache_key = "test_dataset_hash"
        path = RetrievalService.get_index_path(cache_key)
        
        assert str(cache_key) in str(path)
        assert str(path).endswith(".index")

    def test_get_index_path_with_custom_base(self):
        """Тест: получение пути с кастомной базовой директорией."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            cache_key = "custom_key"
            path = RetrievalService.get_index_path(cache_key, base_dir=base)
            
            assert path.parent == base
            assert cache_key in str(path)

    def test_load_index_not_exists(self):
        """Тест: загрузка несуществующего индекса."""
        mock_emb = MagicMock()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = RetrievalService.load_index(
                Path(tmpdir) / "nonexistent.index",
                mock_emb
            )
            
            assert result is None

    def test_load_index_missing_metadata(self):
        """Тест: загрузка без метаданных."""
        mock_emb = MagicMock()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Создаём только .index без .meta.pkl
            index_path = Path(tmpdir) / "test.index"
            index_path.touch()
            
            result = RetrievalService.load_index(index_path, mock_emb)
            assert result is None


class TestRetrievalServiceEdgeCases:
    """Edge cases для RetrievalService."""

    def test_empty_terms_list(self):
        """Тест: пустой список терминов."""
        mock_emb = MagicMock()
        service = RetrievalService(mock_emb)
        
        service.build_index([])
        
        assert service.terms == []
        assert service._index is None

    def test_retrieve_with_empty_index(self):
        """Тест: retrieval после очистки индекса."""
        mock_emb = MagicMock()
        service = RetrievalService(mock_emb)
        
        # Строим, потом очищаем
        service.build_index(["term"])
        service._index = None
        
        with pytest.raises(ValueError, match="Index not built"):
            service.retrieve_neighbors("term")
