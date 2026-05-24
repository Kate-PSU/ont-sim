# backend/tests/test_retrieval_service_save_load.py
# Тесты сохранения и загрузки FAISS индекса
#
# Версия: 1.0
# Обновлено: 2026-04-15

"""
Тесты для save_index и load_index в RetrievalService.

Используют временные директории для реального тестирования
персистентности FAISS индекса.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

backend_root = Path(__file__).parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from src.infrastructure.retrieval_service import RetrievalService


def _make_embedding_service(dim: int = 8):
    """Создать мок EmbeddingService с детерминированными эмбеддингами."""
    mock = MagicMock()
    rng = np.random.default_rng(seed=42)

    term_vectors = {}

    def get_embedding(term: str) -> np.ndarray:
        if term not in term_vectors:
            term_vectors[term] = rng.random(dim).astype(np.float32)
        return term_vectors[term]

    mock.get_embedding.side_effect = get_embedding
    return mock


class TestSaveAndLoadIndex:
    """Тесты полного цикла save → load для FAISS индекса."""

    def test_save_and_load_roundtrip(self):
        """Тест: сохранение и загрузка индекса сохраняет термины."""
        emb_svc = _make_embedding_service(dim=16)

        terms = ["математика", "физика", "химия", "биология"]

        service = RetrievalService(emb_svc)
        service.build_index(terms)

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "test.index"
            service.save_index(index_path)

            # Проверяем, что файлы созданы
            assert index_path.exists()
            meta_path = index_path.with_suffix(".meta.pkl")
            assert meta_path.exists()

            # Загружаем индекс
            loaded_service = RetrievalService.load_index(index_path, emb_svc)

            assert loaded_service is not None
            assert loaded_service.terms == terms

    def test_save_and_load_retrieve_works(self):
        """Тест: после загрузки retrieval работает корректно."""
        emb_svc = _make_embedding_service(dim=16)

        terms = ["математика", "физика", "химия", "биология", "история"]

        service = RetrievalService(emb_svc)
        service.build_index(terms)

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "test.index"
            service.save_index(index_path)

            loaded_service = RetrievalService.load_index(index_path, emb_svc)
            assert loaded_service is not None

            # Должен уметь искать соседей
            neighbors = loaded_service.retrieve_neighbors("математика", k=2)
            assert isinstance(neighbors, list)
            assert len(neighbors) <= 2
            # Сам запрос не должен быть в результатах
            neighbor_terms = [n[0] for n in neighbors]
            assert "математика" not in neighbor_terms

    def test_save_creates_metadata_file(self):
        """Тест: save_index создаёт .meta.pkl файл."""
        emb_svc = _make_embedding_service(dim=8)
        service = RetrievalService(emb_svc)
        service.build_index(["term1", "term2"])

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "my_index.index"
            service.save_index(index_path)

            meta_path = Path(tmpdir) / "my_index.meta.pkl"
            assert meta_path.exists()

    def test_save_creates_parent_directories(self):
        """Тест: save_index создаёт родительские директории."""
        emb_svc = _make_embedding_service(dim=8)
        service = RetrievalService(emb_svc)
        service.build_index(["term1"])

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "nested" / "deep" / "test.index"
            service.save_index(nested_path)
            assert nested_path.exists()

    def test_load_index_missing_meta_returns_none(self):
        """Тест: load_index без .meta.pkl возвращает None."""
        emb_svc = _make_embedding_service(dim=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Создаём только .index файл, без .meta.pkl
            index_path = Path(tmpdir) / "test.index"
            index_path.write_bytes(b"fake_index_data")

            result = RetrievalService.load_index(index_path, emb_svc)
            assert result is None

    def test_load_index_nonexistent_returns_none(self):
        """Тест: load_index с несуществующим путём возвращает None."""
        emb_svc = _make_embedding_service(dim=8)
        result = RetrievalService.load_index(
            Path("/nonexistent/path/test.index"), emb_svc
        )
        assert result is None

    def test_load_index_corrupted_returns_none(self):
        """Тест: load_index с повреждённым файлом возвращает None."""
        import pickle

        emb_svc = _make_embedding_service(dim=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "corrupted.index"
            meta_path = index_path.with_suffix(".meta.pkl")

            # Создаём валидный meta но поврежденный index
            meta = {"terms": ["term1"], "embeddings_shape": (1, 8)}
            with open(meta_path, "wb") as f:
                pickle.dump(meta, f)

            # Пишем мусор в index файл
            index_path.write_bytes(b"corrupted_data_not_faiss")

            result = RetrievalService.load_index(index_path, emb_svc)
            assert result is None

    def test_save_index_without_build_raises(self):
        """Тест: save_index без построенного индекса бросает ValueError."""
        emb_svc = _make_embedding_service(dim=8)
        service = RetrievalService(emb_svc)

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Index not built"):
                service.save_index(Path(tmpdir) / "test.index")

    def test_terms_preserved_after_roundtrip(self):
        """Тест: список терминов сохраняется корректно."""
        emb_svc = _make_embedding_service(dim=16)
        original_terms = ["алгоритм", "структура данных", "граф", "дерево", "хеш-таблица"]

        service = RetrievalService(emb_svc)
        service.build_index(original_terms)

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "terms_test.index"
            service.save_index(index_path)

            loaded = RetrievalService.load_index(index_path, emb_svc)
            assert loaded is not None
            assert loaded.terms == original_terms

    def test_embeddings_matrix_restored_after_load(self):
        """Тест: embeddings_matrix восстанавливается при загрузке."""
        emb_svc = _make_embedding_service(dim=16)
        terms = ["математика", "физика", "химия"]

        service = RetrievalService(emb_svc)
        service.build_index(terms)

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "emb_test.index"
            service.save_index(index_path)

            loaded = RetrievalService.load_index(index_path, emb_svc)
            assert loaded is not None
            assert loaded.embeddings_matrix is not None
            assert loaded.embeddings_matrix.shape[0] == len(terms)


class TestGetIndexPath:
    """Тесты метода get_index_path."""

    def test_default_base_dir(self):
        """Тест: дефолтная директория содержит rag_indices."""
        path = RetrievalService.get_index_path("my_key")
        assert "rag_indices" in str(path)
        assert str(path).endswith("my_key.index")

    def test_custom_base_dir(self):
        """Тест: кастомная базовая директория."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            path = RetrievalService.get_index_path("test_key", base_dir=base)
            assert path.parent == base
            assert path.name == "test_key.index"

    def test_cache_key_in_filename(self):
        """Тест: cache_key входит в имя файла."""
        cache_key = "simlex999_en_abc123"
        path = RetrievalService.get_index_path(cache_key)
        assert cache_key in path.name
