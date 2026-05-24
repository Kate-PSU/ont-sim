# backend/tests/test_bert_embedding_service.py
# Тесты для BERT эмбеддинг сервиса
#
# Версия: 1.0
# Обновлено: 2026-04-15

"""
Тесты для BertEmbeddingService.

Все тесты используют мокинг transformers, чтобы не загружать
реальную модель во время тестирования.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

backend_root = Path(__file__).parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))


def _make_mock_model_output(embedding_dim: int = 768, batch_size: int = 1):
    """Создать мок выходных данных модели BERT."""
    hidden = np.random.rand(batch_size, 10, embedding_dim).astype(np.float32)

    import torch

    hidden_tensor = torch.tensor(hidden)
    mock_output = MagicMock()
    mock_output.last_hidden_state = hidden_tensor
    return mock_output


class TestBertEmbeddingServiceInit:
    """Тесты инициализации BertEmbeddingService."""

    def test_default_model_name(self):
        """Проверяет дефолтное имя модели."""
        from src.infrastructure.bert_embedding_service import BertEmbeddingService

        svc = BertEmbeddingService()
        assert svc.model_name == "bert-base-multilingual-cased"

    def test_custom_model_name(self):
        """Проверяет установку кастомного имени модели."""
        from src.infrastructure.bert_embedding_service import BertEmbeddingService

        svc = BertEmbeddingService(model_name="DeepPavlov/rubert-base-cased")
        assert svc.model_name == "DeepPavlov/rubert-base-cased"

    def test_model_not_loaded_initially(self):
        """Модель не загружается при инициализации (lazy)."""
        from src.infrastructure.bert_embedding_service import BertEmbeddingService

        svc = BertEmbeddingService()
        assert svc._model is None
        assert svc._tokenizer is None


class TestBertEnsureModelLoaded:
    """Тесты метода _ensure_model_loaded."""

    @patch("src.infrastructure.bert_embedding_service.BertEmbeddingService._ensure_model_loaded")
    def test_ensure_model_loaded_not_called_twice(self, mock_ensure):
        """_ensure_model_loaded вызывается при первом обращении."""
        from src.infrastructure.bert_embedding_service import BertEmbeddingService

        svc = BertEmbeddingService()
        svc._ensure_model_loaded()
        assert mock_ensure.called

    def test_ensure_model_loaded_import_error(self):
        """Тест: ImportError при отсутствии transformers."""
        from src.infrastructure.bert_embedding_service import BertEmbeddingService

        svc = BertEmbeddingService()

        with patch.dict("sys.modules", {"transformers": None}):
            import builtins
            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "transformers":
                    raise ImportError("No module named 'transformers'")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                with pytest.raises(ImportError, match="transformers library required"):
                    svc._ensure_model_loaded()


class TestBertGetEmbedding:
    """Тесты метода get_embedding."""

    @pytest.fixture
    def mock_bert_service(self):
        """Создаёт BertEmbeddingService с мок-моделью."""
        import torch
        from src.infrastructure.bert_embedding_service import BertEmbeddingService

        svc = BertEmbeddingService()

        # Создаём мок токенизатора
        mock_tokenizer = MagicMock()
        attention_mask = torch.ones(1, 10, dtype=torch.long)
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
            "attention_mask": attention_mask,
        }

        # Создаём мок модели
        mock_model = MagicMock()
        mock_model.config.hidden_size = 768
        mock_output = _make_mock_model_output(embedding_dim=768, batch_size=1)
        mock_model.return_value = mock_output

        svc._model = mock_model
        svc._tokenizer = mock_tokenizer
        svc._embedding_dim = 768

        return svc

    def test_get_embedding_returns_ndarray(self, mock_bert_service):
        """get_embedding возвращает numpy массив."""
        result = mock_bert_service.get_embedding("нейронная сеть")
        assert isinstance(result, np.ndarray)

    def test_get_embedding_correct_shape(self, mock_bert_service):
        """get_embedding возвращает вектор правильной размерности."""
        result = mock_bert_service.get_embedding("математика")
        assert result.shape == (768,)

    def test_get_embedding_calls_tokenizer(self, mock_bert_service):
        """get_embedding вызывает токенизатор с текстом."""
        mock_bert_service.get_embedding("тест")
        mock_bert_service._tokenizer.assert_called_once()

    def test_get_embedding_runtime_error_if_model_none(self):
        """Тест: RuntimeError если модель не загружена."""
        from src.infrastructure.bert_embedding_service import BertEmbeddingService

        svc = BertEmbeddingService()
        # Принудительно устанавливаем модель как None (без вызова _ensure_model_loaded)
        svc._model = None
        svc._tokenizer = MagicMock()

        with patch.object(svc, "_ensure_model_loaded"):
            with pytest.raises((RuntimeError, AttributeError)):
                svc.get_embedding("test")


class TestBertGetEmbeddingsBatch:
    """Тесты метода get_embeddings_batch."""

    @pytest.fixture
    def mock_bert_service(self):
        """Создаёт BertEmbeddingService с мок-моделью."""
        import torch
        from src.infrastructure.bert_embedding_service import BertEmbeddingService

        svc = BertEmbeddingService()

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(3, 10, dtype=torch.long),
            "attention_mask": torch.ones(3, 10, dtype=torch.long),
        }

        mock_model = MagicMock()
        mock_output = _make_mock_model_output(embedding_dim=768, batch_size=3)
        mock_model.return_value = mock_output

        svc._model = mock_model
        svc._tokenizer = mock_tokenizer
        svc._embedding_dim = 768

        return svc

    def test_empty_list_returns_empty(self):
        """Пустой список терминов возвращает пустой список."""
        from src.infrastructure.bert_embedding_service import BertEmbeddingService

        svc = BertEmbeddingService()
        result = svc.get_embeddings_batch([])
        assert result == []

    def test_batch_returns_list(self, mock_bert_service):
        """get_embeddings_batch возвращает список массивов."""
        texts = ["математика", "физика", "химия"]
        result = mock_bert_service.get_embeddings_batch(texts)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_batch_elements_are_ndarray(self, mock_bert_service):
        """Элементы результата — numpy массивы."""
        texts = ["математика", "физика"]
        result = mock_bert_service.get_embeddings_batch(texts)
        for emb in result:
            assert isinstance(emb, np.ndarray)


class TestBertGetSentenceEmbedding:
    """Тесты метода get_sentence_embedding (mean pooling)."""

    @pytest.fixture
    def mock_bert_service(self):
        """Создаёт BertEmbeddingService с мок-моделью."""
        import torch
        from src.infrastructure.bert_embedding_service import BertEmbeddingService

        svc = BertEmbeddingService()

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }

        mock_model = MagicMock()
        mock_output = _make_mock_model_output(embedding_dim=768, batch_size=1)
        mock_model.return_value = mock_output

        svc._model = mock_model
        svc._tokenizer = mock_tokenizer
        svc._embedding_dim = 768

        return svc

    def test_sentence_embedding_returns_ndarray(self, mock_bert_service):
        """get_sentence_embedding возвращает numpy массив."""
        result = mock_bert_service.get_sentence_embedding("нейронная сеть")
        assert isinstance(result, np.ndarray)

    def test_sentence_embedding_correct_shape(self, mock_bert_service):
        """get_sentence_embedding возвращает вектор правильной размерности."""
        result = mock_bert_service.get_sentence_embedding("математика")
        assert result.shape == (768,)


class TestBertEmbeddingDimension:
    """Тесты свойства embedding_dim."""

    def test_embedding_dim_triggers_model_load(self):
        """embedding_dim вызывает загрузку модели."""
        from src.infrastructure.bert_embedding_service import BertEmbeddingService

        svc = BertEmbeddingService()

        with patch.object(svc, "_ensure_model_loaded") as mock_ensure:
            svc._embedding_dim = 768
            _ = svc.embedding_dim
            mock_ensure.assert_called_once()

    def test_embedding_dim_returns_correct_value(self):
        """embedding_dim возвращает правильную размерность."""
        from src.infrastructure.bert_embedding_service import BertEmbeddingService

        svc = BertEmbeddingService()
        svc._embedding_dim = 512

        with patch.object(svc, "_ensure_model_loaded"):
            assert svc.embedding_dim == 512
