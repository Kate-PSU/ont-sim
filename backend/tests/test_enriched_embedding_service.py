# backend/tests/test_enriched_embedding_service.py
# Тесты для сервиса обогащённых эмбеддингов
#
# Версия: 1.0
# Обновлено: 2026-04-15

"""
Тесты для EnrichedEmbeddingService.

Мокируем EmbeddingService и WordNetService для изоляции логики
комбинирования эмбеддинга термина с эмбеддингами гиперонимов.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

backend_root = Path(__file__).parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))


def _make_mock_embedding_service(dim: int = 4):
    """Создать мок EmbeddingService."""
    mock = MagicMock()
    mock.get_embedding.side_effect = lambda term: np.ones(dim, dtype=np.float32)
    return mock


def _make_mock_wordnet_service(hypernyms: list[str] | None = None):
    """Создать мок WordNetService."""
    mock = MagicMock()
    mock.get_hypernyms.return_value = hypernyms or []
    return mock


class TestEnrichedEmbeddingServiceInit:
    """Тесты инициализации EnrichedEmbeddingService."""

    def test_default_alpha(self):
        """Проверяет дефолтный alpha."""
        from src.infrastructure.enriched_embedding_service import EnrichedEmbeddingService

        svc = EnrichedEmbeddingService(
            embedding_service=_make_mock_embedding_service(),
            wordnet_service=_make_mock_wordnet_service(),
        )
        assert svc.alpha == 0.7

    def test_default_max_hypernyms(self):
        """Проверяет дефолтный max_hypernyms."""
        from src.infrastructure.enriched_embedding_service import EnrichedEmbeddingService

        svc = EnrichedEmbeddingService(
            embedding_service=_make_mock_embedding_service(),
            wordnet_service=_make_mock_wordnet_service(),
        )
        assert svc.max_hypernyms == 3

    def test_custom_alpha(self):
        """Проверяет установку кастомного alpha."""
        from src.infrastructure.enriched_embedding_service import EnrichedEmbeddingService

        svc = EnrichedEmbeddingService(
            embedding_service=_make_mock_embedding_service(),
            wordnet_service=_make_mock_wordnet_service(),
            alpha=0.5,
        )
        assert svc.alpha == 0.5

    def test_invalid_alpha_negative(self):
        """ValueError при alpha < 0."""
        from src.infrastructure.enriched_embedding_service import EnrichedEmbeddingService

        with pytest.raises(ValueError, match="alpha"):
            EnrichedEmbeddingService(
                embedding_service=_make_mock_embedding_service(),
                alpha=-0.1,
            )

    def test_invalid_alpha_greater_than_one(self):
        """ValueError при alpha > 1."""
        from src.infrastructure.enriched_embedding_service import EnrichedEmbeddingService

        with pytest.raises(ValueError, match="alpha"):
            EnrichedEmbeddingService(
                embedding_service=_make_mock_embedding_service(),
                alpha=1.1,
            )

    def test_invalid_max_hypernyms_zero(self):
        """ValueError при max_hypernyms < 1."""
        from src.infrastructure.enriched_embedding_service import EnrichedEmbeddingService

        with pytest.raises(ValueError, match="max_hypernyms"):
            EnrichedEmbeddingService(
                embedding_service=_make_mock_embedding_service(),
                max_hypernyms=0,
            )

    def test_alpha_boundary_zero(self):
        """alpha=0.0 допускается."""
        from src.infrastructure.enriched_embedding_service import EnrichedEmbeddingService

        svc = EnrichedEmbeddingService(
            embedding_service=_make_mock_embedding_service(),
            alpha=0.0,
        )
        assert svc.alpha == 0.0

    def test_alpha_boundary_one(self):
        """alpha=1.0 допускается."""
        from src.infrastructure.enriched_embedding_service import EnrichedEmbeddingService

        svc = EnrichedEmbeddingService(
            embedding_service=_make_mock_embedding_service(),
            alpha=1.0,
        )
        assert svc.alpha == 1.0


class TestGetEnrichedEmbedding:
    """Тесты метода get_enriched_embedding."""

    def test_no_hypernyms_returns_base_embedding(self):
        """Без гиперонимов возвращает базовый эмбеддинг."""
        from src.infrastructure.enriched_embedding_service import EnrichedEmbeddingService

        base_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        emb_svc = MagicMock()
        emb_svc.get_embedding.return_value = base_emb

        wn_svc = _make_mock_wordnet_service(hypernyms=[])

        svc = EnrichedEmbeddingService(
            embedding_service=emb_svc,
            wordnet_service=wn_svc,
        )

        result = svc.get_enriched_embedding("математика")
        np.testing.assert_array_equal(result, base_emb)

    def test_wordnet_unavailable_returns_base_embedding(self):
        """WordNet недоступен → возвращает базовый эмбеддинг."""
        from src.infrastructure.enriched_embedding_service import EnrichedEmbeddingService

        base_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        emb_svc = MagicMock()
        emb_svc.get_embedding.return_value = base_emb

        svc = EnrichedEmbeddingService(
            embedding_service=emb_svc,
            wordnet_service=None,
        )
        # _init_wordnet не инициализирует (нет реального WordNet)
        with patch.object(svc, "_init_wordnet"):
            svc.wordnet_service = None
            result = svc.get_enriched_embedding("математика")

        np.testing.assert_array_equal(result, base_emb)

    def test_with_hypernyms_combines_embeddings(self):
        """С гиперонимами комбинирует эмбеддинги."""
        from src.infrastructure.enriched_embedding_service import EnrichedEmbeddingService

        term_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        hyp_emb = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        emb_svc = MagicMock()
        emb_svc.get_embedding.side_effect = lambda t: (
            term_emb if t == "нейронная сеть" else hyp_emb
        )

        wn_svc = _make_mock_wordnet_service(hypernyms=["алгоритм"])

        svc = EnrichedEmbeddingService(
            embedding_service=emb_svc,
            wordnet_service=wn_svc,
            alpha=0.5,
        )

        result = svc.get_enriched_embedding("нейронная сеть")

        # Должен быть нормализованный вектор
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-5

    def test_custom_alpha_override(self):
        """Переопределение alpha в get_enriched_embedding."""
        from src.infrastructure.enriched_embedding_service import EnrichedEmbeddingService

        term_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        hyp_emb = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        emb_svc = MagicMock()
        emb_svc.get_embedding.side_effect = lambda t: (
            term_emb if t == "нейронная сеть" else hyp_emb
        )

        wn_svc = _make_mock_wordnet_service(hypernyms=["алгоритм"])

        svc = EnrichedEmbeddingService(
            embedding_service=emb_svc,
            wordnet_service=wn_svc,
            alpha=0.7,
        )

        # Используем alpha=1.0 → должен вернуть только term_emb (нормализованный)
        result_high_alpha = svc.get_enriched_embedding("нейронная сеть", alpha=1.0)
        # result_high_alpha должен быть близок к term_emb
        assert result_high_alpha[0] > 0.9

    def test_max_hypernyms_limit(self):
        """Ограничение max_hypernyms работает правильно."""
        from src.infrastructure.enriched_embedding_service import EnrichedEmbeddingService

        emb_svc = MagicMock()
        emb_svc.get_embedding.return_value = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # 5 гиперонимов, но max_hypernyms=2
        wn_svc = _make_mock_wordnet_service(
            hypernyms=["h1", "h2", "h3", "h4", "h5"]
        )

        svc = EnrichedEmbeddingService(
            embedding_service=emb_svc,
            wordnet_service=wn_svc,
            max_hypernyms=2,
        )

        svc.get_enriched_embedding("тест")
        # Должно быть вызвано get_embedding для термина + 2 гиперонима
        assert emb_svc.get_embedding.call_count == 3  # 1 term + 2 hypernyms

    def test_hypernym_embedding_error_is_skipped(self):
        """Ошибка эмбеддинга гиперонима пропускается."""
        from src.infrastructure.enriched_embedding_service import EnrichedEmbeddingService

        term_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        emb_svc = MagicMock()
        call_count = [0]

        def get_emb(t):
            call_count[0] += 1
            if call_count[0] == 1:
                return term_emb
            raise RuntimeError("Embedding error")

        emb_svc.get_embedding.side_effect = get_emb

        wn_svc = _make_mock_wordnet_service(hypernyms=["bad_hypernym"])

        svc = EnrichedEmbeddingService(
            embedding_service=emb_svc,
            wordnet_service=wn_svc,
        )

        # Все гиперонимы упали → возвращает базовый эмбеддинг
        result = svc.get_enriched_embedding("тест")
        np.testing.assert_array_equal(result, term_emb)


class TestGetEnrichedDomainEmbedding:
    """Тесты метода get_enriched_domain_embedding."""

    def test_empty_terms_raises_value_error(self):
        """Пустой список терминов вызывает ValueError."""
        from src.infrastructure.enriched_embedding_service import EnrichedEmbeddingService

        svc = EnrichedEmbeddingService(
            embedding_service=_make_mock_embedding_service(),
            wordnet_service=_make_mock_wordnet_service(),
        )

        with pytest.raises(ValueError):
            svc.get_enriched_domain_embedding([])

    def test_single_term_returns_normalized(self):
        """Один термин → нормализованный эмбеддинг."""
        from src.infrastructure.enriched_embedding_service import EnrichedEmbeddingService

        emb_svc = MagicMock()
        emb_svc.get_embedding.return_value = np.array([3.0, 4.0, 0.0, 0.0], dtype=np.float32)
        wn_svc = _make_mock_wordnet_service(hypernyms=[])

        svc = EnrichedEmbeddingService(
            embedding_service=emb_svc,
            wordnet_service=wn_svc,
        )

        result = svc.get_enriched_domain_embedding(["математика"])
        assert isinstance(result, np.ndarray)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-5

    def test_multiple_terms_returns_centroid(self):
        """Несколько терминов → усреднённый нормализованный вектор."""
        from src.infrastructure.enriched_embedding_service import EnrichedEmbeddingService

        emb_svc = MagicMock()
        emb_svc.get_embedding.return_value = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        wn_svc = _make_mock_wordnet_service(hypernyms=[])

        svc = EnrichedEmbeddingService(
            embedding_service=emb_svc,
            wordnet_service=wn_svc,
        )

        result = svc.get_enriched_domain_embedding(["term1", "term2", "term3"])
        assert result.shape == (4,)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-5

    def test_custom_alpha_passed_through(self):
        """alpha передаётся в get_enriched_embedding."""
        from src.infrastructure.enriched_embedding_service import EnrichedEmbeddingService

        emb_svc = _make_mock_embedding_service()
        wn_svc = _make_mock_wordnet_service(hypernyms=[])

        svc = EnrichedEmbeddingService(
            embedding_service=emb_svc,
            wordnet_service=wn_svc,
        )

        with patch.object(svc, "get_enriched_embedding", wraps=svc.get_enriched_embedding) as mock_gee:
            svc.get_enriched_domain_embedding(["математика"], alpha=0.3)
            # Убедимся, что alpha передаётся
            calls = mock_gee.call_args_list
            assert all(c.kwargs.get("alpha") == 0.3 or c.args[1:] == (0.3,) for c in calls)


class TestGetEnrichmentInfo:
    """Тесты метода get_enrichment_info."""

    def test_no_wordnet_returns_not_enriched(self):
        """Без WordNet возвращает enriched=False."""
        from src.infrastructure.enriched_embedding_service import EnrichedEmbeddingService

        svc = EnrichedEmbeddingService(
            embedding_service=_make_mock_embedding_service(),
            wordnet_service=None,
        )

        with patch.object(svc, "_init_wordnet"):
            svc.wordnet_service = None
            result = svc.get_enrichment_info("математика")

        assert result["enriched"] is False
        assert result["hypernyms"] == []
        assert result["hypernym_count"] == 0
        assert result["term"] == "математика"

    def test_with_hypernyms_returns_enriched_true(self):
        """С гиперонимами возвращает enriched=True."""
        from src.infrastructure.enriched_embedding_service import EnrichedEmbeddingService

        wn_svc = _make_mock_wordnet_service(hypernyms=["наука", "дисциплина"])

        svc = EnrichedEmbeddingService(
            embedding_service=_make_mock_embedding_service(),
            wordnet_service=wn_svc,
        )

        result = svc.get_enrichment_info("математика")

        assert result["enriched"] is True
        assert result["hypernym_count"] == 2
        assert "наука" in result["hypernyms"]

    def test_hypernym_count_limited_by_max_hypernyms(self):
        """Количество гиперонимов ограничено max_hypernyms."""
        from src.infrastructure.enriched_embedding_service import EnrichedEmbeddingService

        wn_svc = _make_mock_wordnet_service(
            hypernyms=["h1", "h2", "h3", "h4", "h5"]
        )

        svc = EnrichedEmbeddingService(
            embedding_service=_make_mock_embedding_service(),
            wordnet_service=wn_svc,
            max_hypernyms=2,
        )

        result = svc.get_enrichment_info("математика")
        assert result["hypernym_count"] == 2
        assert len(result["hypernyms"]) == 2

    def test_no_hypernyms_returns_enriched_false(self):
        """Без гиперонимов возвращает enriched=False."""
        from src.infrastructure.enriched_embedding_service import EnrichedEmbeddingService

        wn_svc = _make_mock_wordnet_service(hypernyms=[])

        svc = EnrichedEmbeddingService(
            embedding_service=_make_mock_embedding_service(),
            wordnet_service=wn_svc,
        )

        result = svc.get_enrichment_info("неизвестный_термин")
        assert result["enriched"] is False


class TestInitWordNet:
    """Тесты метода _init_wordnet."""

    def test_init_wordnet_skipped_if_already_set(self):
        """_init_wordnet не перезаписывает существующий wordnet_service."""
        from src.infrastructure.enriched_embedding_service import EnrichedEmbeddingService

        wn_svc = _make_mock_wordnet_service()

        svc = EnrichedEmbeddingService(
            embedding_service=_make_mock_embedding_service(),
            wordnet_service=wn_svc,
        )

        with patch("src.infrastructure.enriched_embedding_service.WordNetService") as mock_wn_cls:
            svc._init_wordnet()
            # WordNetService не должен создаваться, т.к. уже установлен
            mock_wn_cls.assert_not_called()

    def test_init_wordnet_creates_service_if_none(self):
        """_init_wordnet создаёт WordNetService если он None."""
        from src.infrastructure.enriched_embedding_service import EnrichedEmbeddingService

        svc = EnrichedEmbeddingService(
            embedding_service=_make_mock_embedding_service(),
            wordnet_service=None,
        )

        mock_wn = MagicMock()
        mock_wn.initialize.return_value = None

        with patch("src.infrastructure.enriched_embedding_service.WordNetService", return_value=mock_wn):
            svc._init_wordnet()

        assert svc.wordnet_service is not None

    def test_init_wordnet_handles_initialization_error(self):
        """_init_wordnet обрабатывает ошибку инициализации WordNet."""
        from src.infrastructure.enriched_embedding_service import EnrichedEmbeddingService

        svc = EnrichedEmbeddingService(
            embedding_service=_make_mock_embedding_service(),
            wordnet_service=None,
        )

        mock_wn = MagicMock()
        mock_wn.initialize.side_effect = Exception("DB error")

        with patch("src.infrastructure.enriched_embedding_service.WordNetService", return_value=mock_wn):
            # Не должно бросать исключение
            svc._init_wordnet()

        # При ошибке wordnet_service устанавливается в None
        assert svc.wordnet_service is None
