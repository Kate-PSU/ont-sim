# backend/src/infrastructure/embedding_service.py
# Сервис эмбеддингов (sentence-transformers)
#
# Версия: 1.2
# Обновлено: 2026-04-10
# Изменения: добавлены методы switch_model и reload_default

"""
Сервис для получения эмбеддингов терминов.
Использует модель ai-forever/sbert_large_nlu_ru.
Поддерживает предварительную загрузку модели при старте.
"""

import logging
import time
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from .cache_manager import CacheManager

# Настройка логирования
logger = logging.getLogger(__name__)


class EmbeddingService:
    """Сервис для работы с эмбеддингами.
    
    Загружает модель SentenceTransformer и предоставляет
    методы для получения векторов терминов с кешированием.
    
    Атрибуты:
        model_name: Название модели эмбеддингов.
        cache: Менеджер кеша (опционально).
    """
    
    DEFAULT_MODEL = "ai-forever/sbert_large_nlu_ru"
    FALLBACK_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ENGLISH_MODEL = "sentence-transformers/all-mpnet-base-v2"
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        cache: Optional[CacheManager] = None,
    ) -> None:
        """Инициализация сервиса.
        
        Args:
            model_name: Название модели (по умолчанию sbert_large_nlu_ru).
            cache: Менеджер кеша для Redis.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.fallback_model_name = self.FALLBACK_MODEL
        self.cache = cache
        self._model: Optional[SentenceTransformer] = None
        self._loaded: bool = False
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy-загрузка модели.
        
        Модель загружается только при первом обращении.
        
        Returns:
            Загруженная модель SentenceTransformer.
        """
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
            self._loaded = True
        return self._model
    
    def preload(self, device: str = None) -> None:
        """Предварительная загрузка модели.
        
        Сначала пытается загрузить основную модель, при ошибке —
        использует fallback-модель.
        
        Args:
            device: Устройство для модели ('cpu', 'cuda', 'mps'). 
                   По умолчанию читается из DEVICE env.
        """
        import os
        
        if self._loaded:
            logger.info(f"[embedding] Модель уже загружена: {self.model_name}")
            return
        
        # Определяем устройство
        if device is None:
            device = os.getenv("DEVICE", "cpu")
        
        # Пытаемся загрузить основную модель
        model_loaded = self._try_load_model(self.model_name, device)
        
        # Если не удалось — пробуем fallback
        if not model_loaded:
            logger.warning(f"[embedding] ⚠️ Переключаемся на fallback модель: {self.fallback_model_name}")
            model_loaded = self._try_load_model(self.fallback_model_name, device)
            if model_loaded:
                self.model_name = self.fallback_model_name  # запоминаем что загружено
        
        if not model_loaded:
            raise RuntimeError(f"Не удалось загрузить ни основную, ни fallback модель")
    
    def _try_load_model(self, model_name: str, device: str) -> bool:
        """Попытка загрузки модели.
        
        Args:
            model_name: Название модели.
            device: Устройство ('cpu', 'cuda', 'mps').
            
        Returns:
            True если модель загружена успешно.
        """
        import os
        start_time = time.time()
        
        logger.info(f"[embedding] Начало загрузки модели: {model_name} на {device}")
        
        # Проверяем доступность CUDA если запрошен GPU
        actual_device = device
        if device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning(f"[embedding] ⚠️ CUDA запрошена, но недоступна. Переключаемся на CPU.")
                    actual_device = "cpu"
            except ImportError:
                logger.warning(f"[embedding] ⚠️ PyTorch не найден или не поддерживает CUDA. Переключаемся на CPU.")
                actual_device = "cpu"
        
        try:
            logger.info(f"[embedding] Шаг 1/3: Инициализация SentenceTransformer...")
            self._model = SentenceTransformer(model_name, device=actual_device)
            logger.info(f"[embedding] Шаг 2/3: SentenceTransformer создан, device={actual_device}")
            
            self._loaded = True
            logger.info(f"[embedding] Шаг 3/3: Получение размерности...")
            dimension = self._model.get_sentence_embedding_dimension()
            
            duration = time.time() - start_time
            logger.info(
                f"[embedding] ✅ Модель загружена за {duration:.1f}с, "
                f"размерность={dimension}, device={actual_device}"
            )
            return True
        except Exception as e:
            logger.warning(f"[embedding] ⚠️ Не удалось загрузить {model_name} на {actual_device}: {type(e).__name__}: {e}")
            self._model = None
            self._loaded = False
            return False
    
    def is_loaded(self) -> bool:
        """Проверка, загружена ли модель.
        
        Returns:
            True если модель загружена.
        """
        return self._loaded
    
    def get_embedding(self, term: str) -> np.ndarray:
        """Получение эмбеддинга термина.
        
        Args:
            term: Термин для векторизации.
        
        Returns:
            Вектор эмбеддинга (numpy array).
        """
        embedding = self.model.encode(term, convert_to_numpy=True)
        return embedding
    
    async def get_embedding_cached(self, term: str) -> np.ndarray:
        """Получение эмбеддинга с кешированием.
        
        Сначала проверяет Redis-кэш, затем вычисляет
        эмбеддинг и сохраняет в кэш.
        
        Args:
            term: Термин для векторизации.
        
        Returns:
            Вектор эмбеддинга (numpy array).
        """
        # Проверяем кэш
        if self.cache is not None:
            cached = await self.cache.get_embedding(term)
            if cached is not None:
                return cached
        
        # Вычисляем эмбеддинг
        embedding = self.get_embedding(term)
        
        # Сохраняем в кэш
        if self.cache is not None:
            await self.cache.set_embedding(term, embedding)
        
        return embedding
    
    def get_embeddings_batch(self, terms: list[str]) -> np.ndarray:
        """Получение эмбеддингов для списка терминов.
        
        Args:
            terms: Список терминов.
        
        Returns:
            Матрица эмбеддингов (N x D).
        """
        embeddings = self.model.encode(terms, convert_to_numpy=True)
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Получение размерности эмбеддинга.
        
        Returns:
            Размерность вектора эмбеддинга.
        """
        return self.model.get_sentence_embedding_dimension()
    
    def switch_model(self, model_name: str, device: str = None) -> bool:
        """Переключение модели эмбеддингов.
        
        Выгружает текущую модель и загружает новую.
        Используется для переключения между RU/EN моделями.
        
        Args:
            model_name: Название новой модели.
            device: Устройство ('cpu', 'cuda', 'mps').
            
        Returns:
            True если переключение успешно.
        """
        import os
        
        if device is None:
            device = os.getenv("DEVICE", "cpu")
        
        logger.info(f"[embedding] Переключение модели: {self.model_name} → {model_name}")
        
        # Выгружаем текущую модель
        self._model = None
        self._loaded = False
        
        # Загружаем новую модель
        success = self._try_load_model(model_name, device)
        
        if success:
            self.model_name = model_name
            logger.info(f"[embedding] ✅ Модель переключена на: {model_name}")
        
        return success
    
    def reload_default(self, device: str = None) -> bool:
        """Перезагрузка модели по умолчанию (RU).
        
        После выполнения EN бенчмарков возвращаемся к русской модели.
        
        Args:
            device: Устройство.
            
        Returns:
            True если перезагрузка успешна.
        """
        return self.switch_model(self.DEFAULT_MODEL, device)
