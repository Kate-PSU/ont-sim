# backend/tests/test_cache_manager.py
# Тесты для CacheManager
#
# Версия: 1.1
# Обновлено: 2026-04-08

"""
Тесты для менеджера кеширования с использованием fakeredis.
"""

import numpy as np
import pytest

try:
    import fakeredis.aioredis
    FAKEREDIS_AVAILABLE = True
except ImportError:
    FAKEREDIS_AVAILABLE = False

from src.infrastructure.cache_manager import CacheManager


@pytest.mark.skipif(not FAKEREDIS_AVAILABLE, reason="fakeredis not installed")
class TestCacheManager:
    """Тесты для CacheManager с fakeredis."""

    @pytest.fixture
    async def cache(self):
        """Фикстура для создания CacheManager с fakeredis."""
        manager = CacheManager(host="fakeredis", port=6379)
        # Каждый тест получает свой изолированный FakeRedis
        faker = fakeredis.aioredis.FakeRedis(decode_responses=False)
        manager._client = faker
        await faker.flushdb()
        yield manager
        await manager.disconnect()

    @pytest.mark.asyncio
    async def test_set_and_get_embedding(self, cache):
        """Тест: сохранение и получение эмбеддинга."""
        term = "python"
        embedding = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        
        await cache.set_embedding(term, embedding)
        result = await cache.get_embedding(term)
        
        assert result is not None
        np.testing.assert_array_almost_equal(result, embedding)

    @pytest.mark.asyncio
    async def test_get_nonexistent_embedding(self, cache):
        """Тест: получение несуществующего эмбеддинга возвращает None."""
        result = await cache.get_embedding("nonexistent_term_xyz")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_and_get_centroid(self, cache):
        """Тест: сохранение и получение центроида."""
        domain = "mathematics"
        centroid = np.array([0.5, 0.3, 0.8, 0.1], dtype=np.float32)
        
        await cache.set_centroid(domain, centroid)
        result = await cache.get_centroid(domain)
        
        assert result is not None
        np.testing.assert_array_almost_equal(result, centroid)

    @pytest.mark.asyncio
    async def test_get_nonexistent_centroid(self, cache):
        """Тест: получение несуществующего центроида возвращает None."""
        result = await cache.get_centroid("nonexistent_domain")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_and_get_similarity(self, cache):
        """Тест: сохранение и получение близости."""
        domain1 = "mathematics"
        domain2 = "physics"
        score = 0.856
        
        await cache.set_similarity(domain1, domain2, score)
        result = await cache.get_similarity(domain1, domain2)
        
        assert result is not None
        assert abs(result - score) < 1e-5

    @pytest.mark.asyncio
    async def test_get_nonexistent_similarity(self, cache):
        """Тест: получение несуществующей близости возвращает None."""
        result = await cache.get_similarity("domain1", "domain2")
        assert result is None

    @pytest.mark.asyncio
    async def test_clear_all(self, cache):
        """Тест: очистка всего кеша."""
        # Добавляем данные
        await cache.set_embedding("term1", np.array([1.0, 2.0]))
        await cache.set_centroid("domain1", np.array([3.0, 4.0]))
        
        # Очищаем
        await cache.clear_all()
        
        # Проверяем, что данные удалены
        assert await cache.get_embedding("term1") is None
        assert await cache.get_centroid("domain1") is None

    @pytest.mark.asyncio
    async def test_cache_prefixes(self, cache):
        """Тест: проверка правильности префиксов ключей."""
        assert cache.EMB_PREFIX == "emb:"
        assert cache.CENTROID_PREFIX == "centroid:"
        assert cache.SIM_PREFIX == "sim:"

    @pytest.mark.asyncio
    async def test_multiple_embeddings(self, cache):
        """Тест: сохранение нескольких эмбеддингов."""
        # Используем разные ключи с разными размерами массивов
        await cache.set_embedding("unique_term_1", np.array([1.0, 0.0], dtype=np.float32))
        await cache.set_embedding("unique_term_2", np.array([0.0, 1.0], dtype=np.float32))
        await cache.set_embedding("unique_term_3", np.array([0.707, 0.707], dtype=np.float32))
        
        result1 = await cache.get_embedding("unique_term_1")
        result2 = await cache.get_embedding("unique_term_2")
        result3 = await cache.get_embedding("unique_term_3")
        
        np.testing.assert_array_almost_equal(result1, np.array([1.0, 0.0]))
        np.testing.assert_array_almost_equal(result2, np.array([0.0, 1.0]))
        np.testing.assert_array_almost_equal(result3, np.array([0.707, 0.707]))

    @pytest.mark.asyncio
    async def test_default_ttl(self, cache):
        """Тест: проверка значения TTL по умолчанию."""
        assert cache.default_ttl == 24 * 60 * 60  # 24 часа
        assert cache.DEFAULT_TTL == 24 * 60 * 60

    @pytest.mark.asyncio
    async def test_custom_ttl(self):
        """Тест: использование кастомного TTL."""
        manager = CacheManager(host="fakeredis", port=6379, default_ttl=3600)
        assert manager.default_ttl == 3600

    @pytest.mark.asyncio
    async def test_ttl_on_embedding(self, cache):
        """Тест: проверка TTL при сохранении эмбеддинга."""
        term = "test_term_ttl"
        embedding = np.array([1.0, 2.0], dtype=np.float32)
        
        # Сохраняем с кастомным TTL
        await cache.set_embedding(term, embedding, ttl=60)
        
        # Проверяем, что ключ существует
        result = await cache.get_embedding(term)
        assert result is not None
        
        # Проверяем TTL через fakeredis
        ttl = await cache.client.ttl(f"{cache.EMB_PREFIX}{term}")
        assert ttl > 0
        assert ttl <= 60

    @pytest.mark.asyncio
    async def test_ttl_on_centroid(self, cache):
        """Тест: проверка TTL при сохранении центроида."""
        domain = "test_domain_ttl"
        centroid = np.array([1.0, 2.0], dtype=np.float32)
        
        await cache.set_centroid(domain, centroid, ttl=120)
        
        ttl = await cache.client.ttl(f"{cache.CENTROID_PREFIX}{domain}")
        assert ttl > 0
        assert ttl <= 120

    @pytest.mark.asyncio
    async def test_ttl_on_similarity(self, cache):
        """Тест: проверка TTL при сохранении близости."""
        await cache.set_similarity("domain_x", "domain_y", 0.5, ttl=300)
        
        ttl = await cache.client.ttl(f"{cache.SIM_PREFIX}domain_x:domain_y:cosine")
        assert ttl > 0
        assert ttl <= 300

    @pytest.mark.asyncio
    async def test_clear_prefix(self, cache):
        """Тест: очистка ключей по префиксу."""
        # Добавляем несколько эмбеддингов
        await cache.set_embedding("emb_term1", np.array([1.0, 2.0]))
        await cache.set_embedding("emb_term2", np.array([3.0, 4.0]))
        await cache.set_embedding("emb_term3", np.array([5.0, 6.0]))
        
        # Добавляем центроид (не должен быть удалён)
        await cache.set_centroid("some_domain", np.array([7.0, 8.0]))
        
        # Очищаем только эмбеддинги
        deleted = await cache.clear_prefix(cache.EMB_PREFIX)
        
        assert deleted == 3
        assert await cache.get_embedding("emb_term1") is None
        assert await cache.get_embedding("emb_term2") is None
        assert await cache.get_embedding("emb_term3") is None
        # Центроид должен остаться
        assert await cache.get_centroid("some_domain") is not None

    @pytest.mark.asyncio
    async def test_clear_prefix_empty(self, cache):
        """Тест: очистка несуществующего префикса."""
        deleted = await cache.clear_prefix("nonexistent:")
        assert deleted == 0


class TestCacheManagerClientProperty:
    """Тесты для свойства client без реального подключения."""

    def test_client_raises_error_when_not_connected(self):
        """Тест: client вызывает RuntimeError если не подключено."""
        manager = CacheManager()
        with pytest.raises(RuntimeError, match="Не подключено к Redis"):
            _ = manager.client
