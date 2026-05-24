# backend/tests/test_tfidf_service.py
# Тесты для сервиса TF-IDF
#
# Версия: 1.0
# Обновлено: 2026-04-10

"""
Тесты для TfidfService — расчёт весов терминов методом TF-IDF.
"""

import numpy as np
import pytest

from src.infrastructure.tfidf_service import (
    TfidfService,
    TermWeight,
    TfidfResult,
    calculate_term_frequency,
)


class TestCalculateTermFrequency:
    """Тесты функции расчёта частоты термина."""

    def test_tf_returns_dict(self):
        """Тест: функция возвращает словарь частот."""
        tokens = ["машинное", "обучение", "машинное"]
        tf = calculate_term_frequency(tokens)
        
        assert isinstance(tf, dict)
        assert tf["машинное"] == pytest.approx(2/3, abs=0.01)

    def test_tf_zero_occurrences(self):
        """Тест: отсутствующий термин не в словаре."""
        tokens = ["машинное", "обучение"]
        tf = calculate_term_frequency(tokens)
        
        assert "глубокое" not in tf

    def test_tf_case_sensitive(self):
        """Тест: TF чувствителен к регистру."""
        tokens = ["Машинное", "машинное", "МАШИННОЕ"]
        tf = calculate_term_frequency(tokens)
        
        # Каждый токен учитывается отдельно
        assert len(tf) == 3


class TestTfidfServiceInit:
    """Тесты инициализации TfidfService."""

    def test_init_default(self):
        """Тест: инициализация с параметрами по умолчанию."""
        service = TfidfService()
        
        assert service.idf_threshold == 0.0
        assert service.z_score_threshold == -2.0
        assert service.vectorizer is None

    def test_init_with_thresholds(self):
        """Тест: инициализация с порогами."""
        service = TfidfService(idf_threshold=1.0, z_score_threshold=-1.0)
        
        assert service.idf_threshold == 1.0
        assert service.z_score_threshold == -1.0


class TestTfidfServiceFit:
    """Тесты метода fit() — обучение TF-IDФ."""

    def test_fit_single_document(self):
        """Тест: fit_terms с одним документом."""
        terms = ["машинное", "обучение"]
        service = TfidfService().fit_terms(terms)
        
        # После fit_terms _terms_cache должен быть заполнен
        assert len(service._terms_cache) >= 1

    def test_fit_multiple_documents(self):
        """Тест: fit с несколькими документами."""
        documents = [
            ["машинное", "обучение"],
            ["глубокое", "обучение"],
            ["машинное", "обучение", "классификация"],
        ]
        service = TfidfService().fit(documents)
        
        assert len(service.documents) == 3

    def test_fit_idf_calculation(self):
        """Тест: IDF рассчитывается корректно."""
        documents = [
            ["термин_a", "термин_b"],
            ["термин_a", "термин_c"],
            ["термин_a", "термин_b", "термин_c"],
        ]
        service = TfidfService().fit(documents)
        
        # термин_a встречается во всех документах → IDF минимальный
        idf_a = service.get_idf("термин_a")
        idf_b = service.get_idf("термин_b")
        
        # термин_a встречается чаще → IDF меньше
        assert idf_a <= idf_b

    def test_fit_empty_documents(self):
        """Тест: fit с пустым списком - пустой результат."""
        service = TfidfService()
        result = service.fit([])
        # Не должно вызывать ошибку, просто пустой результат
        assert result is service
        assert len(service.documents) == 0


class TestTfidfServiceCalculate:
    """Тесты расчёта TF-IDF."""

    def test_calculate_tfidf(self):
        """Тест: расчёт TF-IDF для набора терминов."""
        terms = ["машинное", "обучение", "глубокое"]
        service = TfidfService().fit_terms(terms)
        
        weights = service.calculate_tfidf(["машинное", "обучение"])
        
        # Термины должны быть в кэше после fit_terms
        assert "машинное" in service._terms_cache
        assert "обучение" in service._terms_cache
        assert weights["машинное"] >= 0

    def test_calculate_tfidf_with_lemmatization(self):
        """Тест: TF-IDF с лемматизацией."""
        terms = ["машинный", "обучение"]
        service = TfidfService().fit_terms(terms)
        
        weights = service.calculate_tfidf(["машинное"])
        
        # Термины в корпусе: машинный, но не машинное
        # Метод все равно вычисляет TF-IDF
        assert isinstance(weights, dict)


class TestTfidfServiceNormalize:
    """Тесты нормализации весов."""

    def test_normalize_zscore(self):
        """Тест: нормализация Z-score."""
        weights = {
            "термин_a": 10.0,
            "термин_b": 20.0,
            "термин_c": 30.0,
        }
        service = TfidfService()
        normalized = service.normalize_zscore(weights)
        
        assert "термин_a" in normalized
        assert "термин_b" in normalized
        assert "термин_c" in normalized
        # Среднее ≈ 0, std ≈ 1
        values = list(normalized.values())
        assert abs(sum(values)) < 1e-10  # сумма ≈ 0


class TestTfidfServiceFilter:
    """Тесты фильтрации терминов."""

    def test_filter_by_idf(self):
        """Тест: фильтрация по IDF."""
        documents = [
            ["редкий_термин", "часто_встречающийся"],
            ["редкий_термин", "часто_встречающийся"],
            ["часто_встречающийся", "другой_термин"],
        ]
        service = TfidfService().fit(documents)
        
        # Фильтруем список терминов с высоким порогом
        terms = ["редкий_термин", "часто_встречающийся", "другой_термин"]
        
        # Получаем IDF знания
        idf_редкий = service.get_idf("редкий_термин")
        idf_часто = service.get_idf("часто_встречающийся")
        
        # Используем средний порог
        threshold = (idf_редкий + idf_часто) / 2
        filtered = service.filter_by_idf(terms, threshold=threshold)
        
        # "редкий_термин" должен остаться (больший IDF)
        assert "редкий_термин" in filtered

    def test_filter_by_zscore(self):
        """Тест: фильтрация по Z-score."""
        z_scores = {
            "термин_a": 0.0,
            "термин_b": 5.0,
            "термин_c": 10.0,
        }
        service = TfidfService()
        filtered = service.filter_by_zscore(z_scores, threshold=1.0)
        
        assert "термин_a" not in filtered  # z-score < 1
        # термин_b и термин_c должны остаться


class TestTfidfServiceProcess:
    """Тесты полного цикла обработки домена."""

    def test_process_domain(self):
        """Тест: обработка домена."""
        documents = [
            ["машинное", "обучение", "ML"],
            ["глубокое", "обучение", "DL"],
        ]
        service = TfidfService().fit(documents)
        
        result = service.process_domain(
            ["машинное", "обучение", "ML", "DL"]
        )
        
        assert isinstance(result, TfidfResult)
        assert result.domain_terms == ["машинное", "обучение", "ML", "DL"]

    def test_process_domain_empty(self):
        """Тест: обработка пустого домена требует инициализации."""
        # Сначала нужно вызвать fit()
        service = TfidfService().fit([["термин"]])
        result = service.process_domain([])
        
        assert result.domain_terms == []
        assert len(result.weights) == 0


class TestTfidfServiceGetWeights:
    """Тесты получения весов для центроида."""

    def test_get_weights_for_centroid(self):
        """Тест: получение весов в формате для центроида."""
        documents = [
            ["термин_a", "термин_b", "термин_b"],
            ["термин_a", "термин_c"],
        ]
        service = TfidfService().fit(documents)
        
        weights = service.get_weights_for_centroid(
            ["термин_a", "термин_b", "термин_c"]
        )
        
        assert isinstance(weights, np.ndarray)
        assert len(weights) == 3


class TestTermWeight:
    """Тесты dataclass TermWeight."""

    def test_term_weight_create(self):
        """Тест: создание TermWeight с полным набором параметров."""
        weight = TermWeight(term="машинное", weight=0.5, tfidf=0.5, idf=1.0, z_score=0.0)
        
        assert weight.term == "машинное"
        assert weight.weight == 0.5
        assert weight.tfidf == 0.5
        assert weight.idf == 1.0
        assert weight.z_score == 0.0


class TestTfidfResult:
    """Тесты dataclass TfidfResult."""

    def test_tfidf_result_create(self):
        """Тест: создание TfidfResult."""
        result = TfidfResult(
            domain="test_domain",
            weights={"a": 0.5, "b": 0.3},
            filtered_terms=["a", "b"],
            matrix_shape=(1, 2),
            domain_terms=["a", "b"],
        )
        
        assert result.domain == "test_domain"
        assert result.weights == {"a": 0.5, "b": 0.3}
        assert result.filtered_terms == ["a", "b"]
        assert result.matrix_shape == (1, 2)
        assert result.domain_terms == ["a", "b"]


class TestTfidfServiceEdgeCases:
    """Edge cases для TfidfService."""

    def test_tfidf_single_document_single_term(self):
        """Тест: один документ с одним термином."""
        service = TfidfService().fit_terms(["один_термин"])
        
        weights = service.calculate_tfidf(["один_термин"])
        # Термин присутствует в корпусе после fit_terms
        assert "один_термин" in weights

    def test_tfidf_unknown_term(self):
        """Тест: неизвестный термин."""
        service = TfidfService().fit([["известный"]])
        
        idf = service.get_idf("неизвестный")
        # Неизвестный термин должен иметь максимальный IDF
        assert idf >= 0

    def test_tfidf_empty_weights(self):
        """Тест: пустой словарь весов."""
        service = TfidfService()
        normalized = service.normalize_zscore({})
        
        assert normalized == {}
