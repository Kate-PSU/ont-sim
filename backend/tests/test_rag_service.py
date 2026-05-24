# backend/tests/test_rag_service.py
# Тесты для RAG сервиса
#
# Версия: 1.0
# Обновлено: 2026-04-15

"""
Тесты для RAGService.

Использует мокинг EmbeddingService и RetrievalService
для изоляции тестируемой логики.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

backend_root = Path(__file__).parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))


def _make_embedding_service(dim: int = 8):
    """Создать мок EmbeddingService, возвращающий случайные векторы."""
    mock_svc = MagicMock()
    mock_svc.get_embedding.side_effect = lambda term: np.random.rand(dim).astype(np.float32)
    return mock_svc


class TestRAGServiceInit:
    """Тесты инициализации RAGService."""

    def test_default_k_neighbors(self):
        """Проверяет дефолтное количество соседей."""
        from src.infrastructure.rag_service import RAGService

        svc = RAGService(embedding_service=_make_embedding_service())
        assert svc.k_neighbors == 5

    def test_custom_k_neighbors(self):
        """Проверяет установку кастомного k_neighbors."""
        from src.infrastructure.rag_service import RAGService

        svc = RAGService(embedding_service=_make_embedding_service(), k_neighbors=3)
        assert svc.k_neighbors == 3

    def test_no_retrieval_initially(self):
        """Retrieval сервис не создаётся при инициализации."""
        from src.infrastructure.rag_service import RAGService

        svc = RAGService(embedding_service=_make_embedding_service())
        assert svc._retrieval is None
        assert svc._corpus_terms == []


class TestRAGServiceSetCorpus:
    """Тесты метода set_corpus."""

    def test_set_corpus_stores_terms(self):
        """set_corpus сохраняет список терминов."""
        from src.infrastructure.rag_service import RAGService

        emb_svc = _make_embedding_service()
        svc = RAGService(embedding_service=emb_svc)

        with patch("src.infrastructure.rag_service.RetrievalService") as mock_rs_cls:
            mock_rs = MagicMock()
            mock_rs_cls.return_value = mock_rs

            terms = ["математика", "физика", "химия"]
            svc.set_corpus(terms)

            assert svc._corpus_terms == terms

    def test_set_corpus_builds_index(self):
        """set_corpus вызывает build_index."""
        from src.infrastructure.rag_service import RAGService

        emb_svc = _make_embedding_service()
        svc = RAGService(embedding_service=emb_svc, k_neighbors=3)

        with patch("src.infrastructure.rag_service.RetrievalService") as mock_rs_cls:
            mock_rs = MagicMock()
            mock_rs_cls.return_value = mock_rs

            terms = ["математика", "физика"]
            svc.set_corpus(terms)

            mock_rs.build_index.assert_called_once_with(terms, k=3)

    def test_set_corpus_creates_retrieval_service(self):
        """set_corpus создаёт RetrievalService."""
        from src.infrastructure.rag_service import RAGService

        emb_svc = _make_embedding_service()
        svc = RAGService(embedding_service=emb_svc)

        with patch("src.infrastructure.rag_service.RetrievalService") as mock_rs_cls:
            mock_rs = MagicMock()
            mock_rs_cls.return_value = mock_rs

            svc.set_corpus(["term1"])
            assert svc._retrieval is not None


class TestRAGServiceGetTermCentroid:
    """Тесты метода get_term_centroid."""

    def test_no_retrieval_returns_base_embedding(self):
        """Без retrieval возвращает базовый эмбеддинг."""
        from src.infrastructure.rag_service import RAGService

        base_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        emb_svc = MagicMock()
        emb_svc.get_embedding.return_value = base_emb

        svc = RAGService(embedding_service=emb_svc)
        # _retrieval == None

        result = svc.get_term_centroid("математика")
        np.testing.assert_array_equal(result, base_emb)

    def test_empty_corpus_returns_base_embedding(self):
        """Пустой корпус возвращает базовый эмбеддинг."""
        from src.infrastructure.rag_service import RAGService

        base_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        emb_svc = MagicMock()
        emb_svc.get_embedding.return_value = base_emb

        svc = RAGService(embedding_service=emb_svc)
        svc._retrieval = MagicMock()  # retrieval есть, но корпус пустой
        svc._corpus_terms = []

        result = svc.get_term_centroid("математика")
        np.testing.assert_array_equal(result, base_emb)

    def test_with_retrieval_aggregates_embeddings(self):
        """С retrieval агрегирует эмбеддинг с соседями."""
        from src.infrastructure.rag_service import RAGService

        base_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        neighbor_emb = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        emb_svc = MagicMock()
        emb_svc.get_embedding.side_effect = lambda t: (
            base_emb if t == "математика" else neighbor_emb
        )

        svc = RAGService(embedding_service=emb_svc, k_neighbors=1)
        mock_retrieval = MagicMock()
        mock_retrieval.retrieve_neighbors.return_value = [("физика", 0.9)]
        svc._retrieval = mock_retrieval
        svc._corpus_terms = ["физика"]

        result = svc.get_term_centroid("математика")

        # Результат должен быть нормализованным средним
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)
        # Норма должна быть ~1
        assert abs(np.linalg.norm(result) - 1.0) < 1e-5

    def test_retrieval_exception_returns_base_embedding(self):
        """При ошибке retrieval возвращает базовый эмбеддинг."""
        from src.infrastructure.rag_service import RAGService

        base_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        emb_svc = MagicMock()
        emb_svc.get_embedding.return_value = base_emb

        svc = RAGService(embedding_service=emb_svc)
        mock_retrieval = MagicMock()
        mock_retrieval.retrieve_neighbors.side_effect = RuntimeError("FAISS error")
        svc._retrieval = mock_retrieval
        svc._corpus_terms = ["физика"]

        result = svc.get_term_centroid("математика")
        np.testing.assert_array_equal(result, base_emb)

    def test_empty_neighbors_returns_base_embedding(self):
        """Пустой список соседей возвращает базовый эмбеддинг."""
        from src.infrastructure.rag_service import RAGService

        base_emb = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
        emb_svc = MagicMock()
        emb_svc.get_embedding.return_value = base_emb

        svc = RAGService(embedding_service=emb_svc)
        mock_retrieval = MagicMock()
        mock_retrieval.retrieve_neighbors.return_value = []
        svc._retrieval = mock_retrieval
        svc._corpus_terms = ["физика"]

        result = svc.get_term_centroid("математика")
        np.testing.assert_array_equal(result, base_emb)


class TestRAGServiceGetCentroid:
    """Тесты метода get_centroid."""

    def test_empty_terms_returns_zero_vector(self):
        """Пустой список терминов возвращает нулевой вектор."""
        from src.infrastructure.rag_service import RAGService

        emb_svc = _make_embedding_service()
        svc = RAGService(embedding_service=emb_svc)

        result = svc.get_centroid([])
        assert isinstance(result, np.ndarray)
        assert result.shape == (1024,)
        assert np.all(result == 0)

    def test_returns_normalized_vector(self):
        """get_centroid возвращает нормализованный вектор."""
        from src.infrastructure.rag_service import RAGService

        emb_svc = MagicMock()
        emb_svc.get_embedding.return_value = np.array([3.0, 4.0, 0.0, 0.0], dtype=np.float32)

        svc = RAGService(embedding_service=emb_svc)

        with patch("src.infrastructure.rag_service.RetrievalService") as mock_rs_cls:
            mock_rs = MagicMock()
            mock_rs.retrieve_neighbors.return_value = []
            mock_rs_cls.return_value = mock_rs

            result = svc.get_centroid(["математика"])

        assert abs(np.linalg.norm(result) - 1.0) < 1e-5

    def test_uses_existing_corpus(self):
        """get_centroid использует уже установленный корпус."""
        from src.infrastructure.rag_service import RAGService

        emb_svc = MagicMock()
        emb_svc.get_embedding.return_value = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        svc = RAGService(embedding_service=emb_svc)
        svc._corpus_terms = ["уже_установлен"]

        with patch.object(svc, "set_corpus") as mock_set_corpus:
            with patch.object(svc, "get_term_centroid") as mock_gtc:
                mock_gtc.return_value = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                svc.get_centroid(["математика"])
                # set_corpus не должен вызываться
                mock_set_corpus.assert_not_called()


class TestRAGServiceSaveLoad:
    """Тесты save_state и load_state."""

    def test_save_state_no_retrieval(self):
        """save_state без retrieval не падает."""
        from src.infrastructure.rag_service import RAGService

        emb_svc = _make_embedding_service()
        svc = RAGService(embedding_service=emb_svc)
        # Не должно бросать исключение
        svc.save_state(Path("/tmp/test_rag"))

    def test_save_state_with_retrieval_calls_save_index(self):
        """save_state с retrieval вызывает save_index."""
        from src.infrastructure.rag_service import RAGService

        emb_svc = _make_embedding_service()
        svc = RAGService(embedding_service=emb_svc)
        svc._corpus_terms = ["математика", "физика"]

        mock_retrieval = MagicMock()
        svc._retrieval = mock_retrieval

        with patch("src.infrastructure.rag_service.RetrievalService.get_index_path") as mock_path:
            mock_path.return_value = Path("/tmp/test.index")
            svc.save_state(Path("/tmp/test_rag"))

        mock_retrieval.save_index.assert_called_once()

    def test_load_state_nonexistent_path(self):
        """load_state с несуществующим путём возвращает сервис без retrieval."""
        from src.infrastructure.rag_service import RAGService

        emb_svc = _make_embedding_service()
        path = Path("/nonexistent/path/to/state")

        svc = RAGService.load_state(path, embedding_service=emb_svc)

        assert isinstance(svc, RAGService)
        assert svc._retrieval is None

    def test_load_state_creates_service_with_embedding_service(self):
        """load_state создаёт RAGService с переданным embedding_service."""
        from src.infrastructure.rag_service import RAGService

        emb_svc = _make_embedding_service()
        path = Path("/nonexistent/path")

        svc = RAGService.load_state(path, embedding_service=emb_svc)
        assert svc.embedding_service is emb_svc
