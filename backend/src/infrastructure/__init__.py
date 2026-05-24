# backend/src/infrastructure/__init__.py
# Инфраструктурный слой (Clean Architecture)
#
# Версия: 1.1
# Обновлено: 2026-04-10
# Изменения: добавлен settings.py

"""
Инфраструктурный слой содержит:
- Кеширование (Redis)
- Загрузка данных
- Эмбеддинги (sentence-transformers)
- Предобработка текста
- TF-IDF веса и нормализация
- Настройки приложения (settings)
"""

from .cache_manager import CacheManager
from .data_loader import CSVDataLoader, DataLoader
from .embedding_service import EmbeddingService
from .enriched_embedding_service import EnrichedEmbeddingService
from .text_preprocessing import (
    lemmatize,
    preprocess_term,
    preprocess_terms_batch,
    remove_stop_words,
    tokenize,
)
from .tfidf_service import TfidfService, calculate_term_frequency
from .sklearn_tfidf import SklearnTfidfSimilarity, create_tfidf_service, SklearnTfidfResult
from .wordnet_service import WordNetService
from .en_wordnet_service import EnglishWordNetService
from .results_storage import ResultsStorage, get_storage, save_benchmark_results, get_benchmark_result
from .settings import Settings, get_settings

__all__ = [
    "CacheManager",
    "CSVDataLoader",
    "DataLoader",
    "EmbeddingService",
    "EnrichedEmbeddingService",
    "EnglishWordNetService",
    "ResultsStorage",
    "SklearnTfidfSimilarity",
    "SklearnTfidfResult",
    "create_tfidf_service",
    "get_storage",
    "save_benchmark_results",
    "get_benchmark_result",
    "TfidfService",
    "WordNetService",
    "calculate_term_frequency",
    "lemmatize",
    "preprocess_term",
    "preprocess_terms_batch",
    "remove_stop_words",
    "tokenize",
    "Settings",
    "get_settings",
]
