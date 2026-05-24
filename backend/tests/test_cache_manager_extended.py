# backend/tests/test_cache_manager_extended.py
# Расширенные тесты для CacheManager — блокировки и статусы
#
# Версия: 1.0
# Обновлено: 2026-04-15

"""
Расширенные тесты для CacheManager.

Покрывают методы:
- acquire_lock / release_lock / is_locked / get_lock_ttl
- get_system_status
- set_operation_status / get_operation_status
"""

import sys
from pathlib import Path

import pytest

backend_root = Path(__file__).parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

try:
    import fakeredis.aioredis
    FAKEREDIS_AVAILABLE = True
except ImportError:
    FAKEREDIS_AVAILABLE = False

from src.infrastructure.cache_manager import CacheManager


@pytest.mark.skipif(not FAKEREDIS_AVAILABLE, reason="fakeredis not installed")
class TestCacheManagerLocks:
    """Тесты для методов блокировки (mutex)."""

    @pytest.fixture
    async def cache(self):
        """Фикстура для создания CacheManager с fakeredis."""
        manager = CacheManager(host="fakeredis", port=6379)
        faker = fakeredis.aioredis.FakeRedis(decode_responses=False)
        manager._client = faker
        await faker.flushdb()
        yield manager
        await manager.disconnect()

    @pytest.mark.asyncio
    async def test_acquire_lock_success(self, cache):
        """Тест: успешное получение блокировки."""
        result = await cache.acquire_lock("benchmark")
        assert result is True

    @pytest.mark.asyncio
    async def test_acquire_lock_twice_fails(self, cache):
        """Тест: повторная блокировка возвращает False."""
        await cache.acquire_lock("benchmark")
        result = await cache.acquire_lock("benchmark")
        assert result is False

    @pytest.mark.asyncio
    async def test_release_lock_success(self, cache):
        """Тест: успешное освобождение блокировки."""
        await cache.acquire_lock("upload")
        result = await cache.release_lock("upload")
        assert result is True

    @pytest.mark.asyncio
    async def test_release_lock_not_exists(self, cache):
        """Тест: освобождение несуществующей блокировки."""
        result = await cache.release_lock("graph")
        assert result is False

    @pytest.mark.asyncio
    async def test_is_locked_true(self, cache):
        """Тест: is_locked возвращает True после блокировки."""
        await cache.acquire_lock("benchmark")
        assert await cache.is_locked("benchmark") is True

    @pytest.mark.asyncio
    async def test_is_locked_false(self, cache):
        """Тест: is_locked возвращает False если нет блокировки."""
        assert await cache.is_locked("upload") is False

    @pytest.mark.asyncio
    async def test_is_locked_after_release(self, cache):
        """Тест: is_locked возвращает False после освобождения."""
        await cache.acquire_lock("graph")
        await cache.release_lock("graph")
        assert await cache.is_locked("graph") is False

    @pytest.mark.asyncio
    async def test_acquire_lock_unknown_operation(self, cache):
        """Тест: ValueError для неизвестного типа операции."""
        with pytest.raises(ValueError, match="Неизвестный тип операции"):
            await cache.acquire_lock("unknown_operation")

    @pytest.mark.asyncio
    async def test_acquire_lock_custom_ttl(self, cache):
        """Тест: блокировка с кастомным TTL."""
        result = await cache.acquire_lock("benchmark", ttl=10)
        assert result is True

        ttl = await cache.get_lock_ttl("benchmark")
        assert 0 < ttl <= 10

    @pytest.mark.asyncio
    async def test_get_lock_ttl_no_lock(self, cache):
        """Тест: get_lock_ttl возвращает -2 если блокировки нет."""
        ttl = await cache.get_lock_ttl("upload")
        # -2 если ключ не существует
        assert ttl == -2

    @pytest.mark.asyncio
    async def test_get_lock_ttl_with_lock(self, cache):
        """Тест: get_lock_ttl возвращает положительное значение."""
        await cache.acquire_lock("benchmark")
        ttl = await cache.get_lock_ttl("benchmark")
        assert ttl > 0

    @pytest.mark.asyncio
    async def test_all_lock_operation_types(self, cache):
        """Тест: все типы операций работают."""
        for op in CacheManager.OPERATION_TYPES:
            result = await cache.acquire_lock(op)
            assert result is True
            await cache.release_lock(op)

    @pytest.mark.asyncio
    async def test_lock_after_release_can_reacquire(self, cache):
        """Тест: после освобождения блокировку можно получить снова."""
        await cache.acquire_lock("enrichment")
        await cache.release_lock("enrichment")
        result = await cache.acquire_lock("enrichment")
        assert result is True


@pytest.mark.skipif(not FAKEREDIS_AVAILABLE, reason="fakeredis not installed")
class TestCacheManagerSystemStatus:
    """Тесты для get_system_status."""

    @pytest.fixture
    async def cache(self):
        """Фикстура для создания CacheManager с fakeredis."""
        manager = CacheManager(host="fakeredis", port=6379)
        faker = fakeredis.aioredis.FakeRedis(decode_responses=False)
        manager._client = faker
        await faker.flushdb()
        yield manager
        await manager.disconnect()

    @pytest.mark.asyncio
    async def test_system_status_no_locks(self, cache):
        """Тест: статус без блокировок — busy=False."""
        status = await cache.get_system_status()
        assert status["busy"] is False
        assert status["locked_operations"] == []

    @pytest.mark.asyncio
    async def test_system_status_with_lock(self, cache):
        """Тест: статус с блокировкой — busy=True."""
        await cache.acquire_lock("benchmark")
        status = await cache.get_system_status()
        assert status["busy"] is True
        assert len(status["locked_operations"]) == 1
        assert status["locked_operations"][0]["operation"] == "benchmark"

    @pytest.mark.asyncio
    async def test_system_status_multiple_locks(self, cache):
        """Тест: несколько блокировок."""
        await cache.acquire_lock("benchmark")
        await cache.acquire_lock("upload")
        status = await cache.get_system_status()
        assert status["busy"] is True
        operations = [op["operation"] for op in status["locked_operations"]]
        assert "benchmark" in operations
        assert "upload" in operations

    @pytest.mark.asyncio
    async def test_system_status_lock_released(self, cache):
        """Тест: после освобождения блокировки busy=False."""
        await cache.acquire_lock("graph")
        await cache.release_lock("graph")
        status = await cache.get_system_status()
        assert status["busy"] is False

    @pytest.mark.asyncio
    async def test_system_status_ttl_in_response(self, cache):
        """Тест: в статусе присутствует ttl_remaining."""
        await cache.acquire_lock("benchmark")
        status = await cache.get_system_status()
        op = status["locked_operations"][0]
        assert "ttl_remaining" in op
        assert op["ttl_remaining"] > 0


@pytest.mark.skipif(not FAKEREDIS_AVAILABLE, reason="fakeredis not installed")
class TestCacheManagerOperationStatus:
    """Тесты для set_operation_status и get_operation_status."""

    @pytest.fixture
    async def cache(self):
        """Фикстура для создания CacheManager с fakeredis."""
        manager = CacheManager(host="fakeredis", port=6379)
        faker = fakeredis.aioredis.FakeRedis(decode_responses=False)
        manager._client = faker
        await faker.flushdb()
        yield manager
        await manager.disconnect()

    @pytest.mark.asyncio
    async def test_set_and_get_operation_status(self, cache):
        """Тест: сохранение и получение статуса операции."""
        await cache.set_operation_status("benchmark", "started")
        result = await cache.get_operation_status("benchmark")

        assert result is not None
        assert result["status"] == "started"
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_get_operation_status_not_exists(self, cache):
        """Тест: получение несуществующего статуса → None."""
        result = await cache.get_operation_status("benchmark")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_operation_status_with_progress(self, cache):
        """Тест: статус с прогрессом сохраняется."""
        await cache.set_operation_status("upload", "processing", progress=0.5)
        result = await cache.get_operation_status("upload")

        assert result is not None
        assert result["progress"] == 0.5

    @pytest.mark.asyncio
    async def test_set_operation_status_without_progress(self, cache):
        """Тест: статус без прогресса не содержит поле progress."""
        await cache.set_operation_status("graph", "completed")
        result = await cache.get_operation_status("graph")

        assert result is not None
        assert "progress" not in result

    @pytest.mark.asyncio
    async def test_operation_status_overwrite(self, cache):
        """Тест: обновление статуса перезаписывает предыдущий."""
        await cache.set_operation_status("benchmark", "started")
        await cache.set_operation_status("benchmark", "completed", progress=1.0)
        result = await cache.get_operation_status("benchmark")

        assert result["status"] == "completed"
        assert result["progress"] == 1.0

    @pytest.mark.asyncio
    async def test_multiple_operations_status(self, cache):
        """Тест: разные операции не мешают друг другу."""
        await cache.set_operation_status("benchmark", "started")
        await cache.set_operation_status("upload", "processing", progress=0.3)

        benchmark_status = await cache.get_operation_status("benchmark")
        upload_status = await cache.get_operation_status("upload")

        assert benchmark_status["status"] == "started"
        assert upload_status["status"] == "processing"
        assert upload_status["progress"] == 0.3
