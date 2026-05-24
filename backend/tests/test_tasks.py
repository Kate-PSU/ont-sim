# backend/tests/test_tasks.py
# Тесты для Celery задач
#
# Версия: 1.6
# Обновлено: 2026-04-15

"""
Тесты для модуля tasks.py.

Тестируем конфигурацию Celery и базовые функции.
Полное тестирование bound tasks требует integration tests с Celery worker.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

backend_root = Path(__file__).parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))


# ============================================================
# Тесты init_celery
# ============================================================

class TestInitCelery:
    """Тесты инициализации Celery."""

    def test_init_celery_sets_broker_url(self):
        """Проверяет, что init_celery устанавливает broker_url."""
        from src.application.tasks import celery_app, init_celery

        init_celery(redis_host="myredis", redis_port=6380)

        assert celery_app.conf.broker_url == "redis://myredis:6380/0"
        assert celery_app.conf.result_backend == "redis://myredis:6380/1"

    def test_init_celery_default_values(self):
        """Проверяет значения по умолчанию."""
        from src.application.tasks import celery_app, init_celery

        init_celery()

        assert celery_app.conf.broker_url == "redis://redis:6379/0"

    def test_celery_app_config(self):
        """Проверяет базовую конфигурацию Celery."""
        from src.application.tasks import celery_app

        assert celery_app.conf.task_serializer == "json"
        assert celery_app.conf.accept_content == ["json"]
        assert celery_app.conf.result_serializer == "json"
        assert celery_app.conf.timezone == "Europe/Moscow"
        assert celery_app.conf.enable_utc is True
        assert celery_app.conf.task_track_started is True
        assert celery_app.conf.task_time_limit == 600
        assert celery_app.conf.task_soft_time_limit == 540
        assert celery_app.conf.result_expires == 3600


# ============================================================
# Тесты run_benchmark (signature)
# ============================================================

class TestRunBenchmarkSignature:
    """Тесты сигнатуры функции run_benchmark."""

    def test_run_benchmark_task_has_name(self):
        """Проверяет, что задача имеет правильное имя."""
        from src.application.tasks import run_benchmark

        assert run_benchmark.name == "benchmark.run"

    def test_run_benchmark_is_bound(self):
        """Проверяет, что задача bound."""
        from src.application.tasks import run_benchmark

        # Bound tasks имеют Request
        assert hasattr(run_benchmark, 'request')


# ============================================================
# Тесты build_graph (signature)
# ============================================================

class TestBuildGraphSignature:
    """Тесты сигнатуры функции build_graph."""

    def test_build_graph_task_has_name(self):
        """Проверяет, что задача имеет правильное имя."""
        from src.application.tasks import build_graph

        assert build_graph.name == "graph.build"

    def test_build_graph_is_bound(self):
        """Проверяет, что задача bound."""
        from src.application.tasks import build_graph

        assert hasattr(build_graph, 'request')


# ============================================================
# Тесты dataset_map
# ============================================================

class TestDatasetMap:
    """Тесты маппинга датасетов."""

    def test_dataset_map_keys(self):
        """Проверяет доступные ключи датасетов."""
        from src.application.tasks import run_benchmark

        # Получаем dataset_map из функции через рефлексию
        import inspect
        source = inspect.getsource(run_benchmark)
        
        assert '"hj"' in source
        assert '"hj-rg"' in source
        assert '"hj-mc"' in source
        assert '"simlex"' in source
