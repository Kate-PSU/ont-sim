# backend/tests/test_embedding_service_extended.py
# Расширенные тесты для EmbeddingService
#
# Версия: 1.0
# Обновлено: 2026-04-15

"""
Расширенные тесты для EmbeddingService.

Покрывают методы:
- preload / _try_load_model
- switch_model / reload_default
- is_loaded
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

backend_root = Path(__file__).parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from src.infrastructure.embedding_service import EmbeddingService

TEST_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


# ============================================================
# Тесты is_loaded
# ============================================================

class TestIsLoaded:
    """Тесты метода is_loaded."""

    def test_is_loaded_initially_false(self):
        """is_loaded возвращает False до загрузки модели."""
        service = EmbeddingService(model_name=TEST_MODEL)
        assert service.is_loaded() is False

    @patch("src.infrastructure.embedding_service.SentenceTransformer")
    def test_is_loaded_true_after_model_access(self, mock_st_cls):
        """is_loaded возвращает True после обращения к model."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_cls.return_value = mock_model

        service = EmbeddingService(model_name=TEST_MODEL)
        _ = service.model  # Триггер lazy-загрузки
        assert service.is_loaded() is True


# ============================================================
# Тесты _try_load_model
# ============================================================

class TestTryLoadModel:
    """Тесты метода _try_load_model."""

    @patch("src.infrastructure.embedding_service.SentenceTransformer")
    def test_try_load_model_success(self, mock_st_cls):
        """_try_load_model возвращает True при успехе."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_st_cls.return_value = mock_model

        service = EmbeddingService(model_name=TEST_MODEL)
        result = service._try_load_model(TEST_MODEL, "cpu")

        assert result is True
        assert service._loaded is True
        assert service._model is not None

    @patch("src.infrastructure.embedding_service.SentenceTransformer")
    def test_try_load_model_failure(self, mock_st_cls):
        """_try_load_model возвращает False при ошибке."""
        mock_st_cls.side_effect = RuntimeError("Cannot load model")

        service = EmbeddingService(model_name=TEST_MODEL)
        result = service._try_load_model("nonexistent-model", "cpu")

        assert result is False
        assert service._loaded is False
        assert service._model is None


# ============================================================
# Тесты preload
# ============================================================

class TestPreload:
    """Тесты метода preload."""

    @patch("src.infrastructure.embedding_service.SentenceTransformer")
    def test_preload_success(self, mock_st_cls):
        """preload загружает основную модель."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_st_cls.return_value = mock_model

        service = EmbeddingService(model_name=TEST_MODEL)
        service.preload(device="cpu")

        assert service.is_loaded() is True

    @patch("src.infrastructure.embedding_service.SentenceTransformer")
    def test_preload_already_loaded_skips(self, mock_st_cls):
        """preload пропускается если модель уже загружена."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_st_cls.return_value = mock_model

        service = EmbeddingService(model_name=TEST_MODEL)
        service._loaded = True  # Симулируем уже загруженную модель

        service.preload(device="cpu")

        # SentenceTransformer не должен вызываться
        mock_st_cls.assert_not_called()

    @patch("src.infrastructure.embedding_service.SentenceTransformer")
    def test_preload_falls_back_to_fallback_model(self, mock_st_cls):
        """preload использует fallback модель при ошибке основной."""
        call_count = [0]

        def mock_st_constructor(model_name, device=None):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Cannot load primary model")
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            return mock_model

        mock_st_cls.side_effect = mock_st_constructor

        service = EmbeddingService(model_name="nonexistent-primary-model")
        service.preload(device="cpu")

        # Должен был переключиться на fallback
        assert service.is_loaded() is True
        assert service.model_name == service.FALLBACK_MODEL

    @patch("src.infrastructure.embedding_service.SentenceTransformer")
    def test_preload_raises_if_both_fail(self, mock_st_cls):
        """preload бросает RuntimeError если обе модели недоступны."""
        mock_st_cls.side_effect = RuntimeError("Cannot load any model")

        service = EmbeddingService(model_name="nonexistent-model")

        with pytest.raises(RuntimeError, match="Не удалось загрузить"):
            service.preload(device="cpu")

    @patch("src.infrastructure.embedding_service.SentenceTransformer")
    def test_preload_uses_env_device(self, mock_st_cls):
        """preload читает DEVICE из env переменной."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_cls.return_value = mock_model

        service = EmbeddingService(model_name=TEST_MODEL)

        with patch.dict("os.environ", {"DEVICE": "cpu"}):
            service.preload()  # device=None

        assert service.is_loaded() is True


# ============================================================
# Тесты switch_model
# ============================================================

class TestSwitchModel:
    """Тесты метода switch_model."""

    @patch("src.infrastructure.embedding_service.SentenceTransformer")
    def test_switch_model_success(self, mock_st_cls):
        """switch_model успешно переключает модель."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_cls.return_value = mock_model

        service = EmbeddingService(model_name="old-model")
        # Первоначально загружена old-model
        service._loaded = True
        service._model = mock_model

        new_model = "sentence-transformers/all-MiniLM-L6-v2"
        result = service.switch_model(new_model, device="cpu")

        assert result is True
        assert service.model_name == new_model

    @patch("src.infrastructure.embedding_service.SentenceTransformer")
    def test_switch_model_failure_keeps_old_name(self, mock_st_cls):
        """switch_model не меняет model_name при ошибке."""
        mock_st_cls.side_effect = RuntimeError("Cannot load")

        service = EmbeddingService(model_name="old-model")
        result = service.switch_model("bad-model", device="cpu")

        assert result is False
        assert service.model_name == "old-model"

    @patch("src.infrastructure.embedding_service.SentenceTransformer")
    def test_switch_model_resets_loaded_state(self, mock_st_cls):
        """switch_model сбрасывает старое состояние перед загрузкой новой модели."""
        mock_st_cls.side_effect = RuntimeError("Cannot load new model")

        service = EmbeddingService(model_name="model-1")
        # Симулируем, что старая модель была загружена
        service._loaded = True
        service._model = MagicMock()

        # Попытка переключения на плохую модель
        result = service.switch_model("bad-model", device="cpu")

        # Переключение должно провалиться
        assert result is False
        # После неудачного переключения loaded должен быть False (reset произошёл)
        assert service._loaded is False
        assert service._model is None


# ============================================================
# Тесты reload_default
# ============================================================

class TestReloadDefault:
    """Тесты метода reload_default."""

    @patch("src.infrastructure.embedding_service.SentenceTransformer")
    def test_reload_default_loads_default_model(self, mock_st_cls):
        """reload_default загружает DEFAULT_MODEL."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_st_cls.return_value = mock_model

        service = EmbeddingService(model_name="some-other-model")
        result = service.reload_default(device="cpu")

        assert result is True
        assert service.model_name == EmbeddingService.DEFAULT_MODEL

    @patch("src.infrastructure.embedding_service.SentenceTransformer")
    def test_reload_default_returns_false_on_failure(self, mock_st_cls):
        """reload_default возвращает False при ошибке."""
        mock_st_cls.side_effect = RuntimeError("Cannot load")

        service = EmbeddingService()
        result = service.reload_default(device="cpu")

        assert result is False
