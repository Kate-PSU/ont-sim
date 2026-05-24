# backend/src/infrastructure/settings.py
# Настройки приложения из переменных окружения
#
# Версия: 1.0
# Обновлено: 2026-04-10

"""
Настройки приложения из переменных окружения.

Модуль загружает конфигурацию из ENV с значениями по умолчанию.
"""

import os
from typing import Literal
from functools import lru_cache


def _resolve_dataset_path() -> str:
    """Разрешить путь к датасету с учётом рабочей директории.
    
    Если путь относительный (не начинается с /), добавляем /app prefix.
    Это нужно для корректной работы в Docker контейнере.
    
    Args:
        Нет.
    
    Returns:
        str: Абсолютный путь к датасету.
    """
    raw_path = os.getenv("DATASET_PATH", "data/terms.csv")
    
    # Если путь абсолютный — используем как есть
    if raw_path.startswith("/"):
        return raw_path
    
    # Если относительный — добавляем /app prefix (Docker)
    return f"/app/{raw_path}"


@lru_cache
def get_settings() -> "Settings":
    """Получить настройки приложения (cached).
    
    Returns:
        Settings: Экземпляр настроек (синглтон через lru_cache).
    """
    return Settings()


class Settings:
    """Настройки приложения.
    
    Все параметры загружаются из переменных окружения с значениями по умолчанию.
    Паттерн Singleton реализован через lru_cache.
    
    Атрибуты:
        DATASET_PATH: Путь к датасету терминов.
        EMBEDDING_MODEL: Название модели эмбеддингов.
        IDF_THRESHOLD: Порог IDF для фильтрации терминов.
        ZSCORE_THRESHOLD: Порог Z-score для нормализации.
        SIMILARITY_METRIC: Метрика близости (cosine или euclidean).
        REDIS_HOST: Хост Redis.
        REDIS_PORT: Порт Redis.
        CACHE_TTL: TTL кеша в секундах.
    """
    
    # Путь к датасету (с автопрефиксом /app для Docker)
    DATASET_PATH: str = _resolve_dataset_path()
    
    # Модель эмбеддингов
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", 
        "ai-forever/sbert_large_nlu_ru"
    )
    
    # Порог IDF для фильтрации терминов
    IDF_THRESHOLD: float = float(os.getenv("IDF_THRESHOLD", "0.0"))
    
    # Порог Z-score для нормализации
    ZSCORE_THRESHOLD: float = float(os.getenv("ZSCORE_THRESHOLD", "3.0"))
    
    # Метрика близости: cosine или euclidean
    SIMILARITY_METRIC: Literal["cosine", "euclidean"] = os.getenv(
        "SIMILARITY_METRIC", 
        "cosine"
    )
    
    # Redis
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    
    # Cache TTL в секундах
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))
    
    @property
    def redis_url(self) -> str:
        """Получить URL для Redis.
        
        Returns:
            URL вида redis://host:port/0.
        """
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/0"
