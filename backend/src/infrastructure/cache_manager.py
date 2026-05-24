# backend/src/infrastructure/cache_manager.py
# Менеджер кеширования (Redis + Disk)
#
# Версия: 1.3
# Обновлено: 2026-04-18
# Изменения: добавлен DiskCacheManager для persistent cache

"""
Менеджер для работы с Redis кешем.
Кеширует эмбеддинги, центроиды и результаты близости.
Поддерживает mutex для блокировки тяжёлых операций.
Persistent cache: результаты similarity сохраняются на диск.
"""

import time
from typing import Optional

import numpy as np
import redis.asyncio as redis

from .disk_cache import DiskCacheManager


class CacheManager:
    """Менеджер кеширования в Redis.
    
    Управляет кешированием:
    - Эмбеддингов терминов (emb:{term})
    - Центроидов доменов (centroid:{domain})
    - Попарной близости (sim:{domain1}:{domain2})
    
    Поддерживает mutex для блокировки тяжёлых операций:
    - benchmark - запуск бенчмарка
    - upload - загрузка данных
    - graph - генерация графа
    
    Атрибуты:
        DEFAULT_TTL: TTL по умолчанию в секундах (24 часа).
        LOCK_TTL: TTL для блокировок (5 минут).
    """
    
    # Префиксы ключей
    EMB_PREFIX = "emb:"
    CENTROID_PREFIX = "centroid:"
    SIM_PREFIX = "sim:"
    LOCK_PREFIX = "lock:"
    STATUS_PREFIX = "status:"
    
    # TTL по умолчанию: 24 часа
    DEFAULT_TTL = 24 * 60 * 60
    
    # TTL для блокировок: 5 минут
    LOCK_TTL = 5 * 60
    
    # Типы операций для блокировки
    OPERATION_TYPES = ["benchmark", "upload", "graph", "enrichment"]
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: Optional[str] = None,
        default_ttl: Optional[int] = None,
        disk_cache_dir: Optional[str] = None,
    ) -> None:
        """Инициализация менеджера.
        
        Args:
            host: Хост Redis.
            port: Порт Redis.
            password: Пароль для аутентификации.
            default_ttl: TTL по умолчанию в секундах.
            disk_cache_dir: Директория для disk cache. Если None - не используется.
        """
        self.host = host
        self.port = port
        self.password = password
        self.default_ttl = default_ttl or self.DEFAULT_TTL
        self._client: Optional[redis.Redis] = None
        self._disk_cache: Optional[DiskCacheManager] = None
        self._disk_cache_dir = disk_cache_dir
    
    @property
    def disk_cache(self) -> Optional[DiskCacheManager]:
        """Получение disk cache manager."""
        if self._disk_cache is None and self._disk_cache_dir is not None:
            self._disk_cache = DiskCacheManager(base_dir=self._disk_cache_dir)
        return self._disk_cache
    
    async def warm_up(self, method: Optional[str] = None) -> int:
        """Загрузка данных из disk cache в Redis.
        
        При старте приложения загружает все similarity из JSON файлов
        в Redis для быстрого доступа.
        
        Args:
            method: Метод для загрузки (sbert, rag, wordnet, bert).
                    Если None - все методы.
        
        Returns:
            int: Количество загруженных записей.
        """
        if self.disk_cache is None:
            return 0
        
        records = self.disk_cache.load_all_similarities(method=method)
        loaded = 0
        
        for record in records:
            domain1 = record.get("domain1")
            domain2 = record.get("domain2")
            score = record.get("score")
            metric = record.get("metric", "cosine")
            rec_method = record.get("method", "sbert")
            
            if domain1 and domain2 and score is not None:
                # Используем Redis ключ с методом для sbert/rag/wordnet/bert
                key = f"{self.SIM_PREFIX}{rec_method}:{domain1}:{domain2}:{metric}"
                await self.client.set(key, str(score), ex=self.default_ttl)
                # Симметричная запись
                key_rev = f"{self.SIM_PREFIX}{rec_method}:{domain2}:{domain1}:{metric}"
                await self.client.set(key_rev, str(score), ex=self.default_ttl)
                loaded += 1
        
        return loaded
    
    def _normalize_domain_pair(self, domain1: str, domain2: str) -> tuple[str, str]:
        """Нормализация пары доменов для консистентности ключей.
        
        Сортирует домены по алфавиту, чтобы пара (A, B) и (B, A)
        давала одинаковый ключ.
        
        Args:
            domain1: Первый домен.
            domain2: Второй домен.
        
        Returns:
            Кортеж (d1, d2) где d1 <= d2.
        """
        if domain1.lower() <= domain2.lower():
            return domain1, domain2
        return domain2, domain1
    
    async def get_similarity_by_method(
        self,
        domain1: str,
        domain2: str,
        method: str = "sbert",
        metric: str = "cosine",
    ) -> Optional[float]:
        """Получение близости между доменами с учётом метода.
        
        Сначала проверяет Redis, затем disk cache.
        Использует нормализацию ключей для консистентности.
        
        Args:
            domain1: Первый домен.
            domain2: Второй домен.
            method: Метод расчёта (sbert, rag, wordnet, bert).
            metric: Метрика близости.
        
        Returns:
            Значение близости или None.
        """
        # Нормализуем порядок доменов
        d1, d2 = self._normalize_domain_pair(domain1, domain2)
        
        # Ключ с методом (используем нормализованные домены)
        key = f"{self.SIM_PREFIX}{method}:{d1}:{d2}:{metric}"
        data = await self.client.get(key)
        
        if data:
            return float(data)
        
        # Fallback на disk cache (с нормализацией)
        if self.disk_cache:
            score = self.disk_cache.load_similarity(d1, d2, method, metric)
            if score is not None:
                # Кешируем в Redis
                await self.client.set(key, str(score), ex=self.default_ttl)
                return score
        
        return None
    
    async def set_similarity_by_method(
        self,
        domain1: str,
        domain2: str,
        score: float,
        method: str = "sbert",
        metric: str = "cosine",
        ttl: Optional[int] = None,
    ) -> None:
        """Сохранение близости между доменами (Redis + Disk).
        
        Args:
            domain1: Первый домен.
            domain2: Второй домен.
            score: Значение близости.
            method: Метод расчёта (sbert, rag, wordnet, bert).
            metric: Метрика близости.
            ttl: TTL в секундах (по умолчанию DEFAULT_TTL).
        """
        # Нормализуем порядок доменов для консистентности с get_similarity_by_method
        d1, d2 = self._normalize_domain_pair(domain1, domain2)
        
        # Сохраняем в Redis
        key = f"{self.SIM_PREFIX}{method}:{d1}:{d2}:{metric}"
        ttl_value = ttl if ttl is not None else self.default_ttl
        await self.client.set(key, str(score), ex=ttl_value)
        
        # Сохраняем на диск
        if self.disk_cache:
            self.disk_cache.save_similarity(domain1, domain2, score, method, metric)
    
    async def connect(self) -> None:
        """Подключение к Redis."""
        self._client = redis.Redis(
            host=self.host,
            port=self.port,
            password=self.password,
            decode_responses=False,
        )
    
    async def disconnect(self) -> None:
        """Отключение от Redis."""
        if self._client:
            await self._client.close()
    
    @property
    def client(self) -> redis.Redis:
        """Получение клиента Redis."""
        if self._client is None:
            raise RuntimeError("Не подключено к Redis. Вызовите connect()")
        return self._client
    
    async def get_embedding(self, term: str) -> Optional[np.ndarray]:
        """Получение эмбеддинга термина.
        
        Args:
            term: Термин.
        
        Returns:
            Вектор эмбеддинга или None.
        """
        key = f"{self.EMB_PREFIX}{term}"
        data = await self.client.get(key)
        
        if data:
            return np.frombuffer(data, dtype=np.float32)
        return None
    
    async def set_embedding(
        self,
        term: str,
        embedding: np.ndarray,
        ttl: Optional[int] = None,
    ) -> None:
        """Сохранение эмбеддинга термина с TTL.
        
        Args:
            term: Термин.
            embedding: Вектор эмбеддинга.
            ttl: TTL в секундах (по умолчанию DEFAULT_TTL).
        """
        key = f"{self.EMB_PREFIX}{term}"
        ttl_value = ttl if ttl is not None else self.default_ttl
        await self.client.set(key, embedding.tobytes(), ex=ttl_value)
    
    async def get_centroid(self, domain: str) -> Optional[np.ndarray]:
        """Получение центроида домена.
        
        Args:
            domain: Домен.
        
        Returns:
            Вектор центроида или None.
        """
        key = f"{self.CENTROID_PREFIX}{domain}"
        data = await self.client.get(key)
        
        if data:
            return np.frombuffer(data, dtype=np.float32)
        return None
    
    async def set_centroid(
        self,
        domain: str,
        centroid: np.ndarray,
        ttl: Optional[int] = None,
    ) -> None:
        """Сохранение центроида домена с TTL.
        
        Args:
            domain: Домен.
            centroid: Вектор центроида.
            ttl: TTL в секундах (по умолчанию DEFAULT_TTL).
        """
        key = f"{self.CENTROID_PREFIX}{domain}"
        ttl_value = ttl if ttl is not None else self.default_ttl
        await self.client.set(key, centroid.tobytes(), ex=ttl_value)
    
    async def get_similarity(
        self,
        domain1: str,
        domain2: str,
        metric: str = "cosine",
    ) -> Optional[float]:
        """Получение близости между доменами.
        
        Args:
            domain1: Первый домен.
            domain2: Второй домен.
            metric: Метрика близости.
        
        Returns:
            Значение близости или None.
        """
        key = f"{self.SIM_PREFIX}{domain1}:{domain2}:{metric}"
        data = await self.client.get(key)
        
        if data:
            return float(data)
        return None
    
    async def set_similarity(
        self,
        domain1: str,
        domain2: str,
        score: float,
        metric: str = "cosine",
        ttl: Optional[int] = None,
    ) -> None:
        """Сохранение близости между доменами с TTL.
        
        Args:
            domain1: Первый домен.
            domain2: Второй домен.
            score: Значение близости.
            metric: Метрика близости.
            ttl: TTL в секундах (по умолчанию DEFAULT_TTL).
        """
        key = f"{self.SIM_PREFIX}{domain1}:{domain2}:{metric}"
        ttl_value = ttl if ttl is not None else self.default_ttl
        await self.client.set(key, str(score), ex=ttl_value)
    
    async def clear_prefix(self, prefix: str) -> int:
        """Очистка всех ключей с заданным префиксом.
        
        Args:
            prefix: Префикс ключей для удаления.
        
        Returns:
            Количество удалённых ключей.
        """
        pattern = f"{prefix}*"
        keys = []
        async for key in self.client.scan_iter(match=pattern):
            keys.append(key)
        
        if keys:
            return await self.client.delete(*keys)
        return 0
    
    async def clear_all(self) -> None:
        """Очистка всего кеша."""
        await self.client.flushdb()
    
    # === Методы для блокировки операций (Mutex) ===
    
    async def acquire_lock(
        self,
        operation: str,
        ttl: Optional[int] = None,
    ) -> bool:
        """Получение блокировки для операции.
        
        Использует Redis SET с NX (только если ключ не существует).
        Это гарантирует атомарность - только один процесс может
        установить блокировку.
        
        Args:
            operation: Тип операции (benchmark, upload, graph, enrichment).
            ttl: TTL блокировки в секундах (по умолчанию LOCK_TTL).
        
        Returns:
            True если блокировка получена, False если уже занята.
        """
        if operation not in self.OPERATION_TYPES:
            raise ValueError(f"Неизвестный тип операции: {operation}")
        
        key = f"{self.LOCK_PREFIX}{operation}"
        ttl_value = ttl if ttl is not None else self.LOCK_TTL
        
        # SET с NX (только если не существует) и EX (срок действия)
        result = await self.client.set(
            key,
            str(time.time()),  # Время получения блокировки
            nx=True,  # Только если не существует
            ex=ttl_value,  # TTL
        )
        
        return result is True
    
    async def release_lock(self, operation: str) -> bool:
        """Освобождение блокировки операции.
        
        Args:
            operation: Тип операции.
        
        Returns:
            True если блокировка освобождена, False если её не было.
        """
        key = f"{self.LOCK_PREFIX}{operation}"
        result = await self.client.delete(key)
        return result > 0
    
    async def is_locked(self, operation: str) -> bool:
        """Проверка, заблокирована ли операция.
        
        Args:
            operation: Тип операции.
        
        Returns:
            True если операция заблокирована.
        """
        key = f"{self.LOCK_PREFIX}{operation}"
        return await self.client.exists(key) > 0
    
    async def get_lock_ttl(self, operation: str) -> int:
        """Получение оставшегося времени блокировки.
        
        Args:
            operation: Тип операции.
        
        Returns:
            TTL в секундах, или -1 если блокировки нет, или -2 если ключ не существует.
        """
        key = f"{self.LOCK_PREFIX}{operation}"
        ttl = await self.client.ttl(key)
        return ttl
    
    async def get_system_status(self) -> dict:
        """Получение статуса системы.
        
        Возвращает информацию о заблокированных операциях.
        
        Returns:
            Словарь со статусом системы.
        """
        locked_operations = []
        
        for op in self.OPERATION_TYPES:
            if await self.is_locked(op):
                ttl = await self.get_lock_ttl(op)
                locked_operations.append({
                    "operation": op,
                    "ttl_remaining": ttl,
                })
        
        return {
            "busy": len(locked_operations) > 0,
            "locked_operations": locked_operations,
        }
    
    async def set_operation_status(
        self,
        operation: str,
        status: str,
        progress: Optional[float] = None,
    ) -> None:
        """Установка статуса выполняемой операции.
        
        Args:
            operation: Тип операции.
            status: Статус (started, processing, completed, failed).
            progress: Прогресс от 0.0 до 1.0 (опционально).
        """
        key = f"{self.STATUS_PREFIX}{operation}"
        import json
        data = {
            "status": status,
            "timestamp": time.time(),
        }
        if progress is not None:
            data["progress"] = progress
        
        await self.client.set(key, json.dumps(data), ex=self.LOCK_TTL)
    
    async def get_operation_status(self, operation: str) -> Optional[dict]:
        """Получение статуса операции.
        
        Args:
            operation: Тип операции.
        
        Returns:
            Словарь со статусом или None.
        """
        key = f"{self.STATUS_PREFIX}{operation}"
        data = await self.client.get(key)
        
        if data:
            import json
            return json.loads(data)
        return None
