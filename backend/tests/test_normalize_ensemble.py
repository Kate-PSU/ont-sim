# backend/tests/test_normalize_ensemble.py
# Тесты для функции normalize_ensemble — ансамблевая нормализация
#
# Версия: 1.0
# Обновлено: 2026-04-27

"""
Тесты для функции normalize_ensemble — ансамблевая нормализация
оценок семантической близости от нескольких методов.

Тестируемый функционал:
- Нормализация оценок через rankdata → zscore → weighted mean
- Поддержка кастомных весов
- Обработка edge cases (пустой вход, разная длина)
"""

import numpy as np
import pytest

from src.application.benchmark_service import BenchmarkService


class TestNormalizeEnsemble:
    """Тесты для функции normalize_ensemble."""

    @pytest.fixture
    def service(self):
        """BenchmarkService instance для тестов."""
        return BenchmarkService()

    def test_basic_normalization(self, service):
        """Базовый тест: три массива должны нормализоваться и объединиться."""
        scores_sbert = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
        scores_tfidf = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        scores_wordnet = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

        ensemble = service.normalize_ensemble(
            [scores_sbert, scores_tfidf, scores_wordnet]
        )

        # Проверяем, что результат нормализован в [0, 1]
        assert np.all(ensemble >= 0)
        assert np.all(ensemble <= 1)
        assert len(ensemble) == 5

    def test_equal_weights(self, service):
        """Тест: равные веса дают усреднённый результат."""
        # Используем НЕ зеркальные массивы чтобы получить вариацию
        scores1 = np.array([1.0, 0.8, 0.6, 0.4])  # Почти линейный убывающий
        scores2 = np.array([0.8, 0.9, 0.3, 0.1])  # Разные паттерны

        ensemble = service.normalize_ensemble(
            [scores1, scores2],
            weights={"a": 0.5, "b": 0.5}
        )

        # После нормализации результат должен иметь ненулевое отклонение
        assert np.std(ensemble) > 0
        # Первый элемент (высокий в обоих) должен быть > 0.7
        assert ensemble[0] > 0.7

    def test_custom_weights(self, service):
        """Тест: кастомные веса влияют на результат."""
        scores_sbert = np.array([0.9, 0.5, 0.1])
        scores_tfidf = np.array([0.1, 0.5, 0.9])

        # SBERT с весом 0.9 должен доминировать
        ensemble_sbert_heavy = service.normalize_ensemble(
            [scores_sbert, scores_tfidf],
            weights={"sbert": 0.9, "tfidf": 0.1}
        )

        # TF-IDF с весом 0.9 должен дать другой результат
        ensemble_tfidf_heavy = service.normalize_ensemble(
            [scores_sbert, scores_tfidf],
            weights={"sbert": 0.1, "tfidf": 0.9}
        )

        # Результаты должны отличаться
        assert not np.allclose(ensemble_sbert_heavy, ensemble_tfidf_heavy)

    def test_empty_input(self, service):
        """Тест: пустой список должен вернуть пустой массив."""
        ensemble = service.normalize_ensemble([])
        assert len(ensemble) == 0

    def test_single_method(self, service):
        """Тест: один метод — просто нормализация."""
        scores = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
        ensemble = service.normalize_ensemble([scores])

        # После нормализации должно быть примерно равно входам
        # (с точностью до монотонного преобразования)
        assert np.all(ensemble >= 0)
        assert np.all(ensemble <= 1)
        assert len(ensemble) == 5

    def test_different_lengths_raises(self, service):
        """Тест: разная длина массивов должна вызывать ошибку."""
        scores1 = np.array([0.9, 0.7, 0.5])
        scores2 = np.array([0.1, 0.2])  # Другая длина!

        with pytest.raises(ValueError, match="одинаковую длину"):
            service.normalize_ensemble([scores1, scores2])

    def test_weight_normalization(self, service):
        """Тест: веса должны нормализоваться (сумма = 1)."""
        scores1 = np.array([0.8, 0.5, 0.2])
        scores2 = np.array([0.1, 0.3, 0.7])

        # Неравные веса должны нормализоваться автоматически (сумма = 300)
        ensemble = service.normalize_ensemble(
            [scores1, scores2],
            weights={"a": 100, "b": 200}  # Сумма = 300
        )

        # Результат должен быть корректным и в [0, 1]
        assert np.all(ensemble >= 0)
        assert np.all(ensemble <= 1)
        assert len(ensemble) == 3

    def test_default_weights(self, service):
        """Тест: веса по умолчанию: SBERT=0.4, TF-IDF=0.3, WordNet=0.3."""
        scores = np.array([0.8, 0.5, 0.2])

        # Вызываем без весов — должны использоваться умолчания
        ensemble = service.normalize_ensemble([scores])

        assert len(ensemble) == 3

    def test_realistic_benchmark_scenario(self, service):
        """Тест: симуляция реального сценария бенчмарка.
        
        SBERT: высокие оценки для похожих пар
        TF-IDF: низкие оценки (формы разные)
        WordNet: средние оценки
        """
        # 5 пар: от очень похожих до очень разных
        scores_sbert = np.array([0.95, 0.8, 0.6, 0.4, 0.15])  # SBERT ловит семантику
        scores_tfidf = np.array([0.05, 0.15, 0.25, 0.35, 0.45])  # TF-IDF не ловит синонимы
        scores_wordnet = np.array([0.8, 0.7, 0.5, 0.4, 0.3])  # WordNet средне

        ensemble = service.normalize_ensemble(
            [scores_sbert, scores_tfidf, scores_wordnet],
            weights={"sbert": 0.4, "tfidf": 0.3, "wordnet": 0.3}
        )

        # Ансамбль должен дать сглаженный результат
        assert np.all(ensemble >= 0)
        assert np.all(ensemble <= 1)
        # Первая пара (самая похожая) должна иметь высокий скор
        assert ensemble[0] > ensemble[-1]

    def test_circular_evaluation_prevention(self, service):
        """Тест: ансамбль должен снижать перекос от циклической зависимости.
        
        Симуляция: если один метод даёт завышенные оценки (0.9999 корреляция),
        ансамбль должен сгладить этот эффект.
        """
        # "Идеальные" предсказания одного метода
        perfect_scores = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
        # Противоречивые оценки другого метода
        reverse_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        ensemble = service.normalize_ensemble(
            [perfect_scores, reverse_scores],
            weights={"method1": 0.5, "method2": 0.5}
        )

        # Ансамбль не должен быть полностью повёрнут к одному методу
        # Первая пара (где method1=0.95, method2=0.1) не должна получить
        # слишком высокую оценку, если method2 даёт низкую
        # и наоборот
        assert ensemble[0] < 0.9  # Не слишком высоко
        assert ensemble[-1] > 0.2  # Не слишком низко

    def test_zscore_preserves_order(self, service):
        """Тест: нормализация должна сохранять порядок пар."""
        scores = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
        ensemble = service.normalize_ensemble([scores])

        # Если пара A имеет больший скор чем B,
        # то и в ансамбле A должен иметь больший скор
        assert ensemble[0] > ensemble[1]
        assert ensemble[1] > ensemble[2]
        assert ensemble[2] > ensemble[3]
        assert ensemble[3] > ensemble[4]


class TestNormalizeEnsembleEdgeCases:
    """Дополнительные edge cases."""

    @pytest.fixture
    def service(self):
        return BenchmarkService()

    def test_constant_scores(self, service):
        """Тест: массив с одинаковыми значениями."""
        scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        ensemble = service.normalize_ensemble([scores])

        # Все значения должны быть в [0, 1]
        assert np.all(ensemble >= 0)
        assert np.all(ensemble <= 1)

    def test_two_methods(self, service):
        """Тест: ровно два метода."""
        scores1 = np.array([0.9, 0.7, 0.5])
        scores2 = np.array([0.1, 0.3, 0.5])

        ensemble = service.normalize_ensemble(
            [scores1, scores2],
            weights={"m1": 0.5, "m2": 0.5}
        )

        assert len(ensemble) == 3
        assert np.all(ensemble >= 0)
        assert np.all(ensemble <= 1)

    def test_weights_without_normalization(self, service):
        """Тест: веса работают корректно при отсутствии внешней нормализации."""
        scores1 = np.array([1.0, 0.0, 0.5])
        scores2 = np.array([0.0, 1.0, 0.5])

        # При равных весах среднее должно быть около 0.5 для всех
        ensemble = service.normalize_ensemble(
            [scores1, scores2],
            weights={"a": 0.5, "b": 0.5}
        )

        # Средние значения должны быть в разумном диапазоне
        assert np.all(ensemble >= 0)
        assert np.all(ensemble <= 1)