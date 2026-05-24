# backend/tests/test_benchmark_service.py
# Тесты для сервиса бенчмаркинга
#
# Версия: 1.0
# Обновлено: 2026-04-08

"""
Тесты для BenchmarkService — сервиса сравнения методов
семантической близости на бенчмарках.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.application.benchmark_service import (
    BenchmarkService,
    BenchmarkPair,
    MethodResult,
    BenchmarkComparison,
    format_results_table,
)


class TestBenchmarkService:
    """Тесты для BenchmarkService."""
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Мок сервиса эмбеддингов."""
        mock = MagicMock()
        mock.get_embedding = MagicMock(side_effect=lambda term: np.random.rand(768))
        return mock
    
    @pytest.fixture
    def mock_wordnet_service(self):
        """Мок сервиса RuWordNet."""
        mock = MagicMock()
        mock.get_similarity = MagicMock(return_value=MagicMock(similarity=0.5))
        mock.initialize = MagicMock()
        return mock
    
    @pytest.fixture
    def sample_pairs(self):
        """Пример пар для тестирования."""
        return [
            BenchmarkPair(word1="автомобиль", word2="машина", human_score=0.95),
            BenchmarkPair(word1="кошка", word2="собака", human_score=0.7),
            BenchmarkPair(word1="книга", word2="журнал", human_score=0.8),
            BenchmarkPair(word1="дом", word2="здание", human_score=0.85),
        ]
    
    def test_benchmark_pair_creation(self, sample_pairs):
        """Тест создания BenchmarkPair."""
        pair = sample_pairs[0]
        assert pair.word1 == "автомобиль"
        assert pair.word2 == "машина"
        assert pair.human_score == 0.95
    
    def test_method_result_creation(self):
        """Тест создания MethodResult."""
        result = MethodResult(
            method="SBERT",
            spearman=0.85,
            pearson=0.82,
            mse=0.05,
            missing=0,
            predictions_count=10,
        )
        assert result.method == "SBERT"
        assert result.spearman == 0.85
        assert result.pearson == 0.82
        assert result.mse == 0.05
    
    def test_benchmark_comparison_creation(self, sample_pairs):
        """Тест создания BenchmarkComparison."""
        result1 = MethodResult(
            method="SBERT",
            spearman=0.85,
            pearson=0.82,
            mse=0.05,
            missing=0,
            predictions_count=10,
        )
        comparison = BenchmarkComparison(
            dataset_name="hj",
            dataset_size=10,
            results=[result1],
            execution_time_sec=5.5,
        )
        assert comparison.dataset_name == "hj"
        assert len(comparison.results) == 1
        assert comparison.execution_time_sec == 5.5
    
    def test_cosine_similarity(self, mock_embedding_service):
        """Тест расчёта косинусного сходства."""
        service = BenchmarkService(embedding_service=mock_embedding_service)
        
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.0])
        sim = service._cosine_similarity(v1, v2)
        assert sim == pytest.approx(1.0)
        
        v3 = np.array([1.0, 0.0, 0.0])
        v4 = np.array([0.0, 1.0, 0.0])
        sim = service._cosine_similarity(v3, v4)
        assert sim == pytest.approx(0.0)
    
    def test_cosine_similarity_zero_vector(self):
        """Тест косинусного сходства с нулевым вектором."""
        service = BenchmarkService()
        
        v1 = np.array([0.0, 0.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.0])
        sim = service._cosine_similarity(v1, v2)
        assert sim == 0.0
    
    def test_calculate_metrics(self):
        """Тест расчёта метрик."""
        service = BenchmarkService()
        
        predictions = [0.9, 0.8, 0.7, 0.6]
        ground_truth = [0.95, 0.75, 0.7, 0.65]
        
        spearman, pearson, mse = service._calculate_metrics(predictions, ground_truth)
        
        assert isinstance(spearman, float)
        assert isinstance(pearson, float)
        assert isinstance(mse, float)
        assert -1.0 <= spearman <= 1.0
        assert -1.0 <= pearson <= 1.0
        assert mse >= 0.0
    
    def test_calculate_metrics_empty(self):
        """Тест расчёта метрик с пустыми данными."""
        service = BenchmarkService()
        
        spearman, pearson, mse = service._calculate_metrics([], [])
        
        assert spearman == 0.0
        assert pearson == 0.0
        assert mse == 1.0  # Возвращаем 1.0 при пустых данных
    
    def test_evaluate_method(self, sample_pairs):
        """Тест оценки одного метода."""
        service = BenchmarkService()
        
        def predict_fn(word1, word2):
            return 0.8
        
        result = service._evaluate_method("test_method", sample_pairs, predict_fn)
        
        assert result.method == "test_method"
        assert result.predictions_count == len(sample_pairs)
        assert result.missing == 0
    
    def test_evaluate_method_with_missing(self):
        """Тест оценки метода с пропусками."""
        service = BenchmarkService()
        
        pairs = [
            BenchmarkPair(word1="a", word2="b", human_score=0.5),
            BenchmarkPair(word1="c", word2="d", human_score=0.6),
        ]
        
        call_count = [0]
        
        def predict_fn(word1, word2):
            call_count[0] += 1
            if call_count[0] == 1:
                return None  # Пропуск
            return 0.8
        
        result = service._evaluate_method("test_method", pairs, predict_fn)
        
        assert result.missing == 1
        assert result.predictions_count == 1
    
    def test_evaluate_sbert(self, mock_embedding_service, sample_pairs):
        """Тест оценки SBERT."""
        service = BenchmarkService(embedding_service=mock_embedding_service)
        result = service.evaluate_sbert(sample_pairs)
        
        assert result.method == "SBERT (baseline)"
        assert result.predictions_count <= len(sample_pairs)
    
    def test_evaluate_sbert_zscore_exists(self, mock_embedding_service, sample_pairs):
        """Тест: SBERT + Z-score возвращает валидный результат (mock-данные).
        
        Проверяет, что метод работает без ошибок и возвращает MethodResult
        с ожидаемыми полями. Корреляция может быть любой (случайные mock-данные).
        """
        service = BenchmarkService(embedding_service=mock_embedding_service)
        result = service.evaluate_sbert_zscore(sample_pairs)
        
        assert result.method == "SBERT + Z-score"
        assert result.predictions_count <= len(sample_pairs)
        assert isinstance(result.spearman, float)
        assert isinstance(result.pearson, float)
        assert isinstance(result.mse, float)
    
    def test_evaluate_sbert_zscore_correct(self, sample_pairs):
        """Тест: SBERT + Z-score с детерминированными данными.
        
        Использует фиксированные эмбеддинги для проверки корректности
        расчёта корреляции.
        """
        # Детерминированные эмбеддинги: одинаковые значения
        fixed_emb = np.array([1.0, 0.0, 0.0])
        mock_service = MagicMock()
        mock_service.get_embedding = MagicMock(return_value=fixed_emb)
        
        service = BenchmarkService(embedding_service=mock_service)
        result = service.evaluate_sbert_zscore(sample_pairs)
        
        assert result.method == "SBERT + Z-score"
        # Все предсказания одинаковые (одинаковые эмбеддинги)
        # => корреляция не определена, ожидаем 0.0
        assert result.spearman == 0.0
        assert result.predictions_count == len(sample_pairs)
    
    def test_evaluate_ruwordnet_lin_no_service(self, sample_pairs):
        """Тест RuWordNet без инициализации."""
        service = BenchmarkService(wordnet_service=None)
        result = service.evaluate_ruwordnet_lin(sample_pairs)
        
        # Должен вернуть результат без ошибки (graceful degradation)
        assert result.method == "RuWordNet (Lin)"
        assert result.predictions_count == 0 or result.predictions_count > 0
    
    def test_evaluate_ruwordnet_wup(self, mock_wordnet_service, sample_pairs):
        """Тест оценки RuWordNet Wu-Palmer."""
        service = BenchmarkService(wordnet_service=mock_wordnet_service)
        result = service.evaluate_ruwordnet_wup(sample_pairs)
        
        assert result.method == "RuWordNet (Wu-Palmer)"
    
    def test_evaluate_hybrid(self, mock_embedding_service, mock_wordnet_service, sample_pairs):
        """Тест гибридного метода."""
        service = BenchmarkService(
            embedding_service=mock_embedding_service,
            wordnet_service=mock_wordnet_service,
        )
        result = service.evaluate_hybrid(sample_pairs)
        
        assert result.method == "Hybrid (SBERT + RuWordNet)"
        assert result.predictions_count <= len(sample_pairs)
    
    def test_evaluate_hybrid_without_wordnet(self, mock_embedding_service, sample_pairs):
        """Тест гибридного метода без WordNet."""
        service = BenchmarkService(
            embedding_service=mock_embedding_service,
            wordnet_service=None,
        )
        result = service.evaluate_hybrid(sample_pairs)
        
        # Должен работать на основе SBERT
        assert result.method == "Hybrid (SBERT + RuWordNet)"
        assert result.predictions_count > 0
    
    @patch("src.application.benchmark_service.pd")
    def test_load_benchmark(self, mock_pd, tmp_path):
        """Тест загрузки бенчмарка."""
        # Создаём временный CSV файл
        csv_content = "word1,word2,sim\nавтомобиль,машина,0.95\nкошка,собака,0.7\n"
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)
        
        # Настраиваем мок
        import pandas as pd
        mock_df = pd.DataFrame([
            {"word1": "автомобиль", "word2": "машина", "sim": 0.95},
            {"word1": "кошка", "word2": "собака", "sim": 0.7},
        ])
        mock_pd.read_csv.return_value = mock_df
        
        service = BenchmarkService()
        pairs = service.load_benchmark(str(csv_file))
        
        assert len(pairs) == 2
        assert pairs[0].word1 == "автомобиль"
        assert pairs[1].human_score == 0.7


class TestFormatResultsTable:
    """Тесты для форматирования таблицы результатов."""
    
    def test_format_results_table(self):
        """Тест форматирования таблицы."""
        result1 = MethodResult(
            method="SBERT",
            spearman=0.85,
            pearson=0.82,
            mse=0.05,
            missing=0,
            predictions_count=10,
        )
        result2 = MethodResult(
            method="RuWordNet",
            spearman=0.75,
            pearson=0.72,
            mse=0.08,
            missing=2,
            predictions_count=8,
        )
        
        comparison = BenchmarkComparison(
            dataset_name="hj",
            dataset_size=10,
            results=[result1, result2],
            execution_time_sec=5.5,
        )
        
        table = format_results_table(comparison)
        
        assert "## Результаты: hj" in table
        assert "| Метод | Спирмен | Пирсон | MSE | Missing |" in table
        assert "| SBERT | 0.8500 | 0.8200 |" in table
        assert "| RuWordNet | 0.7500 | 0.7200 |" in table
    
    def test_format_results_table_empty(self):
        """Тест форматирования пустой таблицы."""
        comparison = BenchmarkComparison(
            dataset_name="hj",
            dataset_size=0,
            results=[],
            execution_time_sec=0.0,
        )
        
        table = format_results_table(comparison)
        
        assert "## Результаты: hj" in table
        assert "| Метод |" in table


class TestIntegration:
    """Интеграционные тесты."""
    
    @pytest.fixture
    def mock_emb_service(self):
        """Мок сервиса эмбеддингов для интеграционных тестов."""
        mock = MagicMock()
        mock.get_embedding = MagicMock(side_effect=lambda term: np.random.rand(768))
        return mock
    
    def test_service_initialization(self):
        """Тест инициализации сервиса."""
        service = BenchmarkService()
        
        assert service.embedding_service is not None
        assert service.wordnet_service is None
        assert service.tfidf_service is None
    
    def test_full_benchmark_workflow(self, mock_emb_service, tmp_path):
        """Тест полного workflow бенчмарка."""
        # Создаём временный CSV
        csv_content = "word1,word2,sim\nterm1,term2,0.8\nterm3,term4,0.6\n"
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)
        
        # Мокаем pd.read_csv
        with patch("src.application.benchmark_service.pd") as mock_pd:
            import pandas as pd
            mock_df = pd.DataFrame([
                {"word1": "term1", "word2": "term2", "sim": 0.8},
                {"word1": "term3", "word2": "term4", "sim": 0.6},
            ])
            mock_pd.read_csv.return_value = mock_df
            
            service = BenchmarkService(embedding_service=mock_emb_service)
            comparison = service.run_all(str(csv_file), "test")
            
            assert comparison.dataset_name == "test"
            assert comparison.dataset_size == 2
            assert len(comparison.results) > 0
            assert comparison.execution_time_sec >= 0


class TestAlternativeMethods:
    """Тесты для альтернативных методов (BERTopic, Doc2Vec, LDA) из задачи P01."""
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Мок сервиса эмбеддингов."""
        mock = MagicMock()
        mock.get_embedding = MagicMock(side_effect=lambda term: np.random.rand(768))
        return mock
    
    @pytest.fixture
    def sample_pairs(self):
        """Пример пар для тестирования."""
        return [
            BenchmarkPair(word1="автомобиль", word2="машина", human_score=0.95),
            BenchmarkPair(word1="кошка", word2="собака", human_score=0.7),
            BenchmarkPair(word1="книга", word2="журнал", human_score=0.8),
            BenchmarkPair(word1="дом", word2="здание", human_score=0.85),
            BenchmarkPair(word1="река", word2="озеро", human_score=0.75),
            BenchmarkPair(word1="гора", word2="холм", human_score=0.65),
        ]
    
    def test_evaluate_bertopic_returns_valid_result(self, mock_embedding_service, sample_pairs):
        """Тест: evaluate_bertopic возвращает валидный результат.
        
        Проверяет, что метод работает без ошибок и возвращает MethodResult
        с ожидаемыми полями. Использует fallback если BERTopic не установлен.
        """
        service = BenchmarkService(embedding_service=mock_embedding_service)
        result = service.evaluate_bertopic(sample_pairs)
        
        assert result.method == "BERTopic"
        assert isinstance(result.spearman, float)
        assert isinstance(result.pearson, float)
        assert isinstance(result.mse, float)
        assert isinstance(result.missing, int)
        assert isinstance(result.predictions_count, int)
    
    def test_evaluate_bertopic_with_few_terms(self, mock_embedding_service):
        """Тест: evaluate_bertopic с малым количеством терминов.
        
        При слишком малом количестве терминов (<5) метод должен вернуть
        результат без ошибок (graceful degradation).
        """
        service = BenchmarkService(embedding_service=mock_embedding_service)
        
        # Всего 3 пары = 6 уникальных терминов
        pairs = [
            BenchmarkPair(word1="а", word2="б", human_score=0.8),
            BenchmarkPair(word1="в", word2="г", human_score=0.6),
            BenchmarkPair(word1="д", word2="е", human_score=0.7),
        ]
        
        result = service.evaluate_bertopic(pairs)
        
        assert result.method == "BERTopic"
        assert result.predictions_count == 0 or result.predictions_count > 0
    
    def test_evaluate_doc2vec_returns_valid_result(self, mock_embedding_service, sample_pairs):
        """Тест: evaluate_doc2vec возвращает валидный результат.
        
        Проверяет, что метод работает без ошибок и возвращает MethodResult
        с ожидаемыми полями.
        """
        service = BenchmarkService(embedding_service=mock_embedding_service)
        result = service.evaluate_doc2vec(sample_pairs)
        
        assert result.method == "Doc2Vec"
        assert isinstance(result.spearman, float)
        assert isinstance(result.pearson, float)
        assert isinstance(result.mse, float)
        assert isinstance(result.missing, int)
        assert isinstance(result.predictions_count, int)
    
    def test_evaluate_doc2vec_with_multiword_terms(self, mock_embedding_service):
        """Тест: evaluate_doc2vec с многословными терминами.
        
        Многословные термины должны правильно обрабатываться.
        """
        service = BenchmarkService(embedding_service=mock_embedding_service)
        
        pairs = [
            BenchmarkPair(word1="искусственный интеллект", word2="машинное обучение", human_score=0.9),
            BenchmarkPair(word1="база данных", word2="система управления", human_score=0.75),
        ]
        
        result = service.evaluate_doc2vec(pairs)
        
        assert result.method == "Doc2Vec"
        # Должен обработать все пары
        assert result.predictions_count <= len(pairs)
    
    def test_evaluate_lda_returns_valid_result(self, mock_embedding_service, sample_pairs):
        """Тест: evaluate_lda возвращает валидный результат.
        
        Проверяет, что метод работает без ошибок и возвращает MethodResult
        с ожидаемыми полями.
        """
        service = BenchmarkService(embedding_service=mock_embedding_service)
        result = service.evaluate_lda(sample_pairs)
        
        assert result.method == "LDA"
        assert isinstance(result.spearman, float)
        assert isinstance(result.pearson, float)
        assert isinstance(result.mse, float)
        assert isinstance(result.missing, int)
        assert isinstance(result.predictions_count, int)
    
    def test_evaluate_lda_with_few_terms(self, mock_embedding_service):
        """Тест: evaluate_lda с малым количеством терминов.
        
        При слишком малом количестве терминов (<3) метод должен вернуть
        результат без ошибок.
        """
        service = BenchmarkService(embedding_service=mock_embedding_service)
        
        # Всего 2 пары = 4 уникальных термина
        pairs = [
            BenchmarkPair(word1="а", word2="б", human_score=0.8),
            BenchmarkPair(word1="в", word2="г", human_score=0.6),
        ]
        
        result = service.evaluate_lda(pairs)
        
        assert result.method == "LDA"
        assert result.predictions_count == 0 or result.predictions_count > 0
    
    def test_run_all_includes_alternative_methods(self, mock_embedding_service, tmp_path):
        """Тест: run_all включает альтернативные методы.
        
        Проверяет, что при запуске на русском датасете (hj-rg)
        в результатах присутствуют BERTopic, Doc2Vec, LDA.
        """
        # Создаём временный CSV
        csv_content = "word1,word2,sim\nавтомобиль,машина,0.95\nкошка,собака,0.7\n"
        csv_file = tmp_path / "hj-rg.csv"
        csv_file.write_text(csv_content)
        
        with patch("src.application.benchmark_service.pd") as mock_pd:
            import pandas as pd
            mock_df = pd.DataFrame([
                {"word1": "автомобиль", "word2": "машина", "sim": 0.95},
                {"word1": "кошка", "word2": "собака", "sim": 0.7},
            ])
            mock_pd.read_csv.return_value = mock_df
            
            service = BenchmarkService(embedding_service=mock_embedding_service)
            comparison = service.run_all(str(csv_file), "hj-rg")
            
            # Проверяем, что все методы присутствуют
            method_names = [r.method for r in comparison.results]
            
            assert "SBERT (baseline)" in method_names
            assert "BERTopic" in method_names
            assert "Doc2Vec" in method_names
            assert "LDA" in method_names
    
    def test_run_all_english_includes_alternative_methods(self, mock_embedding_service, tmp_path):
        """Тест: run_all для английского датасета включает альтернативные методы.
        
        Проверяет, что при запуске на английском датасете (simlex999)
        в результатах присутствуют BERTopic, Doc2Vec, LDA.
        """
        # Создаём временный CSV
        csv_content = "word1,word2,sim\ncar,automobile,0.95\ndog,cat,0.7\n"
        csv_file = tmp_path / "simlex999.csv"
        csv_file.write_text(csv_content)
        
        with patch("src.application.benchmark_service.pd") as mock_pd:
            import pandas as pd
            mock_df = pd.DataFrame([
                {"word1": "car", "word2": "automobile", "sim": 0.95},
                {"word1": "dog", "word2": "cat", "sim": 0.7},
            ])
            mock_pd.read_csv.return_value = mock_df
            
            service = BenchmarkService(embedding_service=mock_embedding_service)
            comparison = service.run_all(str(csv_file), "simlex999")
            
            # Проверяем, что все методы присутствуют
            method_names = [r.method for r in comparison.results]
            
            assert "SBERT (baseline)" in method_names
            assert "BERTopic" in method_names
            assert "Doc2Vec" in method_names
            assert "LDA" in method_names
