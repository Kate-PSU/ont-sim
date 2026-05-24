# backend/tests/test_disk_cache.py
# Тесты для DiskCacheManager
#
# Версия: 1.0
# Обновлено: 2026-04-18

"""
Тесты для DiskCacheManager - persistent cache manager.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest


class TestDiskCacheManager:
    """Тесты DiskCacheManager."""
    
    @pytest.fixture
    def temp_dir(self):
        """Временная директория для тестов."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)
    
    @pytest.fixture
    def disk_cache(self, temp_dir):
        """Создание DiskCacheManager с временной директорией."""
        from backend.src.infrastructure.disk_cache import DiskCacheManager
        return DiskCacheManager(base_dir=temp_dir)
    
    def test_init_creates_directories(self, temp_dir):
        """Тест: инициализация создаёт необходимые директории."""
        from backend.src.infrastructure.disk_cache import DiskCacheManager
        
        cache = DiskCacheManager(base_dir=temp_dir)
        
        assert (Path(temp_dir)).exists()
        for method in ["sbert", "rag", "wordnet", "bert"]:
            assert (Path(temp_dir) / method).exists()
    
    def test_save_and_load_similarity(self, disk_cache, temp_dir):
        """Тест: сохранение и загрузка similarity."""
        # Сохраняем
        disk_cache.save_similarity(
            domain1="ml",
            domain2="dl",
            score=0.847,
            method="sbert",
            metric="cosine"
        )
        
        # Проверяем, что файл создан
        expected_path = Path(temp_dir) / "sbert" / "dl-ml_cosine.json"
        assert expected_path.exists()
        
        # Загружаем
        score = disk_cache.load_similarity(
            domain1="ml",
            domain2="dl",
            method="sbert",
            metric="cosine"
        )
        
        assert score == pytest.approx(0.847)
    
    def test_load_missing_returns_none(self, disk_cache):
        """Тест: загрузка отсутствующего файла возвращает None."""
        score = disk_cache.load_similarity(
            domain1="nonexistent",
            domain2="domain",
            method="sbert",
            metric="cosine"
        )
        
        assert score is None
    
    def test_load_all_similarities(self, disk_cache):
        """Тест: загрузка всех similarity записей."""
        # Сохраняем несколько записей
        disk_cache.save_similarity("ml", "dl", 0.847, "sbert", "cosine")
        disk_cache.save_similarity("cs", "math", 0.65, "sbert", "cosine")
        disk_cache.save_similarity("ml", "dl", 0.82, "rag", "cosine")
        
        all_records = disk_cache.load_all_similarities()
        
        assert len(all_records) == 3
    
    def test_load_all_similarities_by_method(self, disk_cache):
        """Тест: загрузка similarity только для конкретного метода."""
        disk_cache.save_similarity("ml", "dl", 0.847, "sbert", "cosine")
        disk_cache.save_similarity("ml", "dl", 0.82, "rag", "cosine")
        disk_cache.save_similarity("cs", "math", 0.65, "sbert", "cosine")
        
        sbert_records = disk_cache.load_all_similarities(method="sbert")
        
        assert len(sbert_records) == 2
        for record in sbert_records:
            assert record["method"] == "sbert"
    
    def test_get_cache_count(self, disk_cache):
        """Тест: подсчёт количества записей."""
        disk_cache.save_similarity("ml", "dl", 0.847, "sbert", "cosine")
        disk_cache.save_similarity("cs", "math", 0.65, "sbert", "cosine")
        disk_cache.save_similarity("ml", "dl", 0.82, "rag", "cosine")
        
        total = disk_cache.get_cache_count()
        sbert_count = disk_cache.get_cache_count(method="sbert")
        
        assert total == 3
        assert sbert_count == 2
    
    def test_get_stats(self, disk_cache):
        """Тест: получение статистики."""
        disk_cache.save_similarity("ml", "dl", 0.847, "sbert", "cosine")
        disk_cache.save_similarity("ml", "dl", 0.82, "rag", "cosine")
        
        stats = disk_cache.get_stats()
        
        assert stats["total"] == 2
        assert stats["methods"]["sbert"] == 1
        assert stats["methods"]["rag"] == 1
    
    def test_clear_method(self, disk_cache, temp_dir):
        """Тест: очистка кеша для конкретного метода."""
        disk_cache.save_similarity("ml", "dl", 0.847, "sbert", "cosine")
        disk_cache.save_similarity("ml", "dl", 0.82, "rag", "cosine")
        
        count = disk_cache.clear_method("sbert")
        
        assert count == 1
        assert len(list((Path(temp_dir) / "sbert").glob("*.json"))) == 0
        assert len(list((Path(temp_dir) / "rag").glob("*.json"))) == 1
    
    def test_clear_all(self, disk_cache, temp_dir):
        """Тест: очистка всего кеша."""
        disk_cache.save_similarity("ml", "dl", 0.847, "sbert", "cosine")
        disk_cache.save_similarity("ml", "dl", 0.82, "rag", "cosine")
        disk_cache.save_similarity("ml", "dl", 0.55, "wordnet", "cosine")
        
        count = disk_cache.clear_all()
        
        assert count == 3
        for method in ["sbert", "rag", "wordnet", "bert"]:
            assert len(list((Path(temp_dir) / method).glob("*.json"))) == 0
    
    def test_domain_order_normalization(self, disk_cache, temp_dir):
        """Тест: нормализация порядка доменов (alphabetical)."""
        disk_cache.save_similarity("zoo", "alpha", 0.5, "sbert", "cosine")
        
        # Загружаем в обратном порядке - результат тот же
        loaded = disk_cache.load_similarity("alpha", "zoo", "sbert", "cosine")
        
        assert loaded == pytest.approx(0.5)
        
        # Проверяем, что файл один
        json_files = list((Path(temp_dir) / "sbert").glob("*.json"))
        assert len(json_files) == 1
    
    def test_json_format(self, disk_cache, temp_dir):
        """Тест: проверка формата JSON файла."""
        disk_cache.save_similarity(
            domain1="ml",
            domain2="dl",
            score=0.847,
            method="sbert",
            metric="cosine"
        )
        
        json_file = list((Path(temp_dir) / "sbert").glob("*.json"))[0]
        
        with open(json_file, "r") as f:
            data = json.load(f)
        
        assert data["domain1"] == "ml"
        assert data["domain2"] == "dl"
        assert data["score"] == 0.847
        assert data["method"] == "sbert"
        assert data["metric"] == "cosine"
        assert "timestamp" in data


class TestCacheManagerDiskIntegration:
    """Тесты интеграции CacheManager с DiskCacheManager."""
    
    @pytest.fixture
    def temp_dir(self):
        """Временная директория для тестов."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_set_similarity_by_method_saves_to_disk(self, temp_dir):
        """Тест: set_similarity_by_method сохраняет на диск."""
        from backend.src.infrastructure.cache_manager import CacheManager
        import redis.asyncio as redis
        
        cache = CacheManager(
            host="localhost",
            port=6379,
            disk_cache_dir=temp_dir,
        )
        await cache.connect()
        
        # Сохраняем
        await cache.set_similarity_by_method(
            domain1="ml",
            domain2="dl",
            score=0.847,
            method="sbert",
            metric="cosine"
        )
        
        # Проверяем диск
        expected_path = Path(temp_dir) / "sbert" / "dl-ml_cosine.json"
        assert expected_path.exists()
        
        await cache.disconnect()
    
    @pytest.mark.asyncio
    async def test_warm_up_loads_to_redis(self, temp_dir):
        """Тест: warm_up загружает данные в Redis."""
        from backend.src.infrastructure.cache_manager import CacheManager
        
        # Предварительно создаём JSON файл
        json_dir = Path(temp_dir) / "sbert"
        json_dir.mkdir(parents=True, exist_ok=True)
        
        json_file = json_dir / "dl-ml_cosine.json"
        json_file.write_text(json.dumps({
            "domain1": "ml",
            "domain2": "dl",
            "score": 0.847,
            "method": "sbert",
            "metric": "cosine",
            "timestamp": "2026-04-18T20:00:00Z"
        }))
        
        cache = CacheManager(
            host="localhost",
            port=6379,
            disk_cache_dir=temp_dir,
        )
        await cache.connect()
        
        # Warm-up
        loaded = await cache.warm_up()
        
        assert loaded >= 1
        
        # Проверяем, что данные в Redis
        key = f"sim:sbert:ml:dl:cosine"
        data = await cache.client.get(key)
        assert data is not None
        
        await cache.disconnect()
