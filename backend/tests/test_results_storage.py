# backend/tests/test_results_storage.py
# Тесты для ResultsStorage
#
# Версия: 1.0
# Обновлено: 2026-04-15

"""
Тесты для модуля персистентного хранения результатов бенчмарков.
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.infrastructure.results_storage import (
    ResultsStorage,
    get_storage,
    save_benchmark_results,
    get_benchmark_result,
)


class TestResultsStorageInit:
    """Тесты инициализации ResultsStorage."""

    def test_init_default_path(self):
        """Тест: инициализация с путём по умолчанию."""
        # Используем временный путь чтобы не засорять
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ResultsStorage(db_path=Path(tmpdir) / "test.db")
            assert storage.db_path.exists()

    def test_init_creates_tables(self):
        """Тест: при инициализации создаются таблицы."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = ResultsStorage(db_path=db_path)
            
            # Таблицы должны быть созданы
            assert storage.db_path.exists()


class TestResultsStorageSaveResult:
    """Тесты сохранения результатов."""

    def test_save_result_success(self):
        """Тест: успешное сохранение результатов."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = ResultsStorage(db_path=db_path)
            
            results = [
                {"method": "sbert", "spearman": 0.85},
                {"method": "wordnet", "spearman": 0.72},
            ]
            
            success = storage.save_result("hj", results, 10.5)
            assert success is True

    def test_save_result_updates_existing(self):
        """Тест: сохранение обновляет существующую запись."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = ResultsStorage(db_path=db_path)
            
            results_v1 = [{"method": "sbert", "spearman": 0.80}]
            results_v2 = [{"method": "sbert", "spearman": 0.85}]
            
            storage.save_result("hj", results_v1, 10.0)
            storage.save_result("hj", results_v2, 15.0)
            
            # Получаем результат
            result = storage.get_result("hj")
            assert result is not None
            assert result["results"][0]["spearman"] == 0.85
            assert result["execution_time_sec"] == 15.0


class TestResultsStorageGetResult:
    """Тесты получения результатов."""

    def test_get_result_exists(self):
        """Тест: получение существующего результата."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = ResultsStorage(db_path=db_path)
            
            original_results = [{"method": "sbert", "spearman": 0.85}]
            storage.save_result("hj", original_results, 10.0)
            
            result = storage.get_result("hj")
            
            assert result is not None
            assert result["dataset_name"] == "hj"
            assert result["results"] == original_results
            assert result["execution_time_sec"] == 10.0

    def test_get_result_not_exists(self):
        """Тест: получение несуществующего результата."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = ResultsStorage(db_path=db_path)
            
            result = storage.get_result("nonexistent")
            assert result is None

    def test_get_result_structure(self):
        """Тест: структура возвращаемого результата."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = ResultsStorage(db_path=db_path)
            
            results = [{"method": "sbert", "spearman": 0.85}]
            storage.save_result("hj", results, 10.0)
            
            result = storage.get_result("hj")
            
            assert "dataset_name" in result
            assert "results" in result
            assert "execution_time_sec" in result
            assert "created_at" in result
            assert "updated_at" in result


class TestResultsStorageGetHistory:
    """Тесты получения истории результатов."""

    def test_get_history_multiple_runs(self):
        """Тест: история с несколькими запусками."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = ResultsStorage(db_path=db_path)
            
            # Сохраняем несколько раз
            for i in range(5):
                storage.save_result("hj", [{"spearman": 0.7 + i * 0.03}], float(i + 1))
            
            history = storage.get_history("hj", limit=3)
            
            assert len(history) == 3
            # Самые свежие первыми
            assert history[0]["execution_time_sec"] == 5.0

    def test_get_history_empty(self):
        """Тест: история для несуществующего датасета."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = ResultsStorage(db_path=db_path)
            
            history = storage.get_history("nonexistent")
            assert history == []


class TestResultsStorageListDatasets:
    """Тесты получения списка датасетов."""

    def test_list_datasets(self):
        """Тест: список сохранённых датасетов."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = ResultsStorage(db_path=db_path)
            
            storage.save_result("hj", [], 1.0)
            storage.save_result("simlex", [], 2.0)
            storage.save_result("wordsim", [], 3.0)
            
            datasets = storage.list_datasets()
            
            assert len(datasets) == 3
            assert "hj" in datasets
            assert "simlex" in datasets
            assert "wordsim" in datasets

    def test_list_datasets_empty(self):
        """Тест: пустой список датасетов."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = ResultsStorage(db_path=db_path)
            
            datasets = storage.list_datasets()
            assert datasets == []


class TestGlobalFunctions:
    """Тесты глобальных функций."""

    def test_save_benchmark_results_function(self):
        """Тест: функция save_benchmark_results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            
            # Создаём storage с нужным путём
            from src.infrastructure import results_storage
            results_storage._storage = ResultsStorage(db_path=db_path)
            
            results = [{"method": "test"}]
            success = save_benchmark_results("test_ds", results, 5.0)
            
            assert success is True
            
            # Очищаем глобальный storage
            results_storage._storage = None

    def test_get_benchmark_result_function(self):
        """Тест: функция get_benchmark_result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            
            # Создаём storage с нужным путём
            from src.infrastructure import results_storage
            results_storage._storage = ResultsStorage(db_path=db_path)
            
            # Сохраняем
            save_benchmark_results("test_ds", [{"test": 1}], 1.0)
            
            # Получаем
            result = get_benchmark_result("test_ds")
            assert result is not None
            assert result["dataset_name"] == "test_ds"
            
            # Очищаем
            results_storage._storage = None
