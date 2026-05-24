"""
Тесты для проверки работы весов в ensemble similarity.

Тест доказывает, что:
1. При sbert=1.0, tfidf=0.0 результат ensemble = sbert_score
2. При sbert=0.0, tfidf=1.0 результат ensemble = tfidf_score
3. При разных весах результаты РАЗНЫЕ
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from backend.src.application.similarity_methods import (
    calculate_ensemble_similarity,
    calculate_sbert_similarity,
    calculate_tfidf_similarity,
)
from backend.src.infrastructure.tfidf_service import TfidfService


class TestEnsembleWeights:
    """Тесты на проверку работы весов в ensemble similarity."""
    
    def test_ensemble_weights_sbert_only(self):
        """При весах sbert=1.0, tfidf=0.0 результат должен быть равен sbert_score."""
        # Мок-данные
        terms1 = ["машина", "обучение"]
        terms2 = ["данные", "модель"]
        
        # Мок центроиды с известными значениями
        centroid1 = np.array([0.1, 0.2, 0.3, 0.4])
        centroid2 = np.array([0.1, 0.2, 0.3, 0.4])
        
        # Мок embedding_service
        mock_emb_service = MagicMock()
        mock_emb_service.get_embeddings_batch.return_value = [
            np.array([0.1, 0.2, 0.3, 0.4]),
            np.array([0.1, 0.2, 0.3, 0.4]),
        ]
        
        # TF-IDF service
        tfidf_service = TfidfService()
        tfidf_service.fit_terms(terms1 + terms2)
        
        # Вычисляем эталонные значения
        sbert_score = calculate_sbert_similarity(
            (terms1, centroid1),
            (terms2, centroid2),
            mock_emb_service
        )
        
        tfidf_score = calculate_tfidf_similarity(
            terms1, terms2, tfidf_service
        )
        
        # Ensemble с весами sbert=1.0, tfidf=0.0
        result = calculate_ensemble_similarity(
            (terms1, centroid1),
            (terms2, centroid2),
            mock_emb_service,
            tfidf_service,
            weights={"sbert": 1.0, "tfidf": 0.0}
        )
        
        # При весах sbert=1.0, tfidf=0.0 результат должен быть равен sbert_score
        assert abs(result["similarity"] - sbert_score) < 0.01, \
            f"Expected {sbert_score}, got {result['similarity']}"
        assert abs(result["sbert_score"] - sbert_score) < 0.01  # float precision
        assert abs(result["tfidf_score"] - tfidf_score) < 0.01
    
    def test_ensemble_weights_tfidf_only(self):
        """При весах sbert=0.0, tfidf=1.0 результат должен быть равен tfidf_score."""
        # Мок-данные
        terms1 = ["машина", "обучение"]
        terms2 = ["данные", "модель"]
        
        # Мок центроиды
        centroid1 = np.array([0.1, 0.2, 0.3, 0.4])
        centroid2 = np.array([0.1, 0.2, 0.3, 0.4])
        
        # Мок embedding_service
        mock_emb_service = MagicMock()
        
        # TF-IDF service
        tfidf_service = TfidfService()
        tfidf_service.fit_terms(terms1 + terms2)
        
        # Вычисляем эталонные значения
        sbert_score = calculate_sbert_similarity(
            (terms1, centroid1),
            (terms2, centroid2),
            mock_emb_service
        )
        
        tfidf_score = calculate_tfidf_similarity(
            terms1, terms2, tfidf_service
        )
        
        # Ensemble с весами sbert=0.0, tfidf=1.0
        result = calculate_ensemble_similarity(
            (terms1, centroid1),
            (terms2, centroid2),
            mock_emb_service,
            tfidf_service,
            weights={"sbert": 0.0, "tfidf": 1.0}
        )
        
        # При весах sbert=0.0, tfidf=1.0 результат должен быть равен tfidf_score
        assert abs(result["similarity"] - tfidf_score) < 0.01, \
            f"Expected {tfidf_score}, got {result['similarity']}"
    
    def test_ensemble_weights_different_results(self):
        """При разных весах результаты должны быть РАЗНЫМИ.
        
        Это основной тест, который доказывает что веса РАБОТАЮТ.
        Если результаты одинаковые при разных весах — баг в коде!
        """
        # Мок-данные с РАЗНЫМИ sbert_score и tfidf_score
        terms1 = ["машина", "обучение", "алгоритм"]
        terms2 = ["еда", "ресторан", "кухня"]  # Совершенно разные термины!
        
        # Мок центроиды
        centroid1 = np.array([0.5, 0.5, 0.5, 0.5])  # SBERT близко к самому себе
        centroid2 = np.array([0.5, 0.5, 0.5, 0.5])
        
        # Мок embedding_service - возвращает предсказуемые эмбеддинги
        mock_emb_service = MagicMock()
        # При одинаковых центроидах SBERT similarity будет ~1.0
        mock_emb_service.get_embeddings_batch.return_value = [
            np.array([0.5, 0.5, 0.5, 0.5]),
            np.array([0.5, 0.5, 0.5, 0.5]),
        ]
        
        # TF-IDF service - должен давать другой результат
        tfidf_service = TfidfService()
        tfidf_service.fit_terms(terms1 + terms2)
        
        # Результат с весами 100% SBERT
        result_100_sbert = calculate_ensemble_similarity(
            (terms1, centroid1),
            (terms2, centroid2),
            mock_emb_service,
            tfidf_service,
            weights={"sbert": 1.0, "tfidf": 0.0}
        )
        
        # Результат с весами 100% TF-IDF
        result_100_tfidf = calculate_ensemble_similarity(
            (terms1, centroid1),
            (terms2, centroid2),
            mock_emb_service,
            tfidf_service,
            weights={"sbert": 0.0, "tfidf": 1.0}
        )
        
        # Результат с весами 50/50
        result_50_50 = calculate_ensemble_similarity(
            (terms1, centroid1),
            (terms2, centroid2),
            mock_emb_service,
            tfidf_service,
            weights={"sbert": 0.5, "tfidf": 0.5}
        )
        
        # Печатаем для отладки
        print(f"\n[TEST] sbert_score: {result_100_sbert['sbert_score']:.4f}")
        print(f"[TEST] tfidf_score: {result_100_sbert['tfidf_score']:.4f}")
        print(f"[TEST] 100% SBERT: {result_100_sbert['similarity']:.4f}")
        print(f"[TEST] 100% TF-IDF: {result_100_tfidf['similarity']:.4f}")
        print(f"[TEST] 50/50: {result_50_50['similarity']:.4f}")
        
        # Если sbert_score == tfidf_score, тест не информативен
        if abs(result_100_sbert['sbert_score'] - result_100_sbert['tfidf_score']) < 0.01:
            pytest.skip("sbert_score и tfidf_score слишком близки для данного датасета")
        
        # Основная проверка: результаты должны быть РАЗНЫМИ
        # 100% SBERT vs 100% TF-IDF должны отличаться
        diff_extreme = abs(result_100_sbert['similarity'] - result_100_tfidf['similarity'])
        
        assert diff_extreme > 0.01, (
            f"БАГ: Результаты одинаковые при разных весах!\n"
            f"100% SBERT: {result_100_sbert['similarity']:.4f}\n"
            f"100% TF-IDF: {result_100_tfidf['similarity']:.4f}\n"
            f"Разница: {diff_extreme:.6f}\n"
            f"Это означает что веса НЕ используются в расчёте!"
        )
        
        # 100% SBERT vs 50/50 должны отличаться
        diff_sbert_50 = abs(result_100_sbert['similarity'] - result_50_50['similarity'])
        assert diff_sbert_50 > 0.001, \
            f"100% SBERT и 50/50 дают одинаковый результат!"
        
        # 100% TF-IDF vs 50/50 должны отличаться
        diff_tfidf_50 = abs(result_100_tfidf['similarity'] - result_50_50['similarity'])
        assert diff_tfidf_50 > 0.001, \
            f"100% TF-IDF и 50/50 дают одинаковый результат!"
    
    def test_ensemble_weights_interpolation(self):
        """При линейной интерполяции весов результат должен меняться линейно."""
        terms1 = ["машина", "обучение"]
        terms2 = ["еда", "готовка"]
        
        centroid1 = np.array([0.5, 0.5, 0.5, 0.5])
        centroid2 = np.array([0.5, 0.5, 0.5, 0.5])
        
        mock_emb_service = MagicMock()
        mock_emb_service.get_embeddings_batch.return_value = [
            np.array([0.5, 0.5, 0.5, 0.5]),
            np.array([0.5, 0.5, 0.5, 0.5]),
        ]
        
        tfidf_service = TfidfService()
        tfidf_service.fit_terms(terms1 + terms2)
        
        results = {}
        for sbert_w in [0.0, 0.25, 0.5, 0.75, 1.0]:
            tfidf_w = 1.0 - sbert_w
            result = calculate_ensemble_similarity(
                (terms1, centroid1),
                (terms2, centroid2),
                mock_emb_service,
                tfidf_service,
                weights={"sbert": sbert_w, "tfidf": tfidf_w}
            )
            results[sbert_w] = result["similarity"]
            print(f"  w_sbert={sbert_w:.2f} -> similarity={result['similarity']:.4f}")
        
        # Проверяем что результаты монотонно меняются
        for w in [0.0, 0.25, 0.5, 0.75]:
            # Если sbert_score > tfidf_score, то при увеличении веса SBERT similarity должна расти
            sbert_score = results[1.0]
            tfidf_score = results[0.0]
            
            if sbert_score > tfidf_score:
                assert results[w + 0.25] > results[w], \
                    f"Монотонность нарушена: {results[w]} -> {results[w + 0.25]}"
            else:
                assert results[w + 0.25] < results[w], \
                    f"Монотонность нарушена: {results[w]} -> {results[w + 0.25]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
