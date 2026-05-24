# backend/src/infrastructure/enriched_embedding_service.py
# Сервис обогащённых эмбеддингов с гиперонимами
#
# Версия: 1.0
# Обновлено: 2026-04-08

"""
Сервис для получения обогащённых эмбеддингов терминов.

Обогащение происходит путём комбинирования эмбеддинга термина
с эмбеддингами его гиперонимов из RuWordNet.

Формула:
    emb_final = alpha * emb(term) + (1 - alpha) * mean(emb(hypernyms))

Где:
    - alpha — вес оригинального термина (0.0 - 1.0)
    - hypernyms — гиперонимы из WordNet (до max_hypernyms штук)
"""

import logging
from typing import Optional

import numpy as np

from .embedding_service import EmbeddingService
from .wordnet_service import WordNetService

logger = logging.getLogger(__name__)


class EnrichedEmbeddingService:
    """Сервис для получения обогащённых эмбеддингов.
    
    Комбинирует эмбеддинг термина с эмбеддингами его гиперонимов
    для улучшения семантического представления.
    
    Атрибуты:
        embedding_service: Базовый сервис эмбеддингов.
        wordnet_service: Сервис RuWordNet для получения гиперонимов.
        alpha: Вес оригинального термина (0.0 - 1.0).
        max_hypernyms: Максимальное число гиперонимов.
    
    Пример:
        >>> service = EnrichedEmbeddingService()
        >>> emb = service.get_enriched_embedding("нейронная сеть")
        >>> print(emb.shape)
        (1024,)
    """
    
    DEFAULT_ALPHA = 0.7
    DEFAULT_MAX_HYPERNYMS = 3
    
    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        wordnet_service: Optional[WordNetService] = None,
        alpha: float = DEFAULT_ALPHA,
        max_hypernyms: int = DEFAULT_MAX_HYPERNYMS,
    ) -> None:
        """Инициализация сервиса.
        
        Args:
            embedding_service: Базовый сервис эмбеддингов.
                Если None, создаётся новый.
            wordnet_service: Сервис RuWordNet.
                Если None, создаётся новый.
            alpha: Вес оригинального термина (0.0 - 1.0).
                Чем больше, тем больше вес у исходного термина.
            max_hypernyms: Максимальное число гиперонимов для учёта.
        """
        self.embedding_service = embedding_service or EmbeddingService()
        self.wordnet_service = wordnet_service
        
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("alpha должен быть в диапазоне [0.0, 1.0]")
        self.alpha = alpha
        
        if max_hypernyms < 1:
            raise ValueError("max_hypernyms должен быть >= 1")
        self.max_hypernyms = max_hypernyms
    
    def _init_wordnet(self) -> None:
        """Инициализация WordNet если ещё не инициализирован."""
        if self.wordnet_service is None:
            self.wordnet_service = WordNetService()
            try:
                self.wordnet_service.initialize()
                logger.info("RuWordNet инициализирован для обогащения эмбеддингов")
            except Exception as e:
                logger.warning(f"Не удалось инициализировать RuWordNet: {e}")
                self.wordnet_service = None
    
    def get_enriched_embedding(
        self,
        term: str,
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        """Получение обогащённого эмбеддинга термина.
        
        Формула:
            emb_final = alpha * emb(term) + (1 - alpha) * mean(emb(hypernyms))
        
        Args:
            term: Термин для векторизации.
            alpha: Переопределение веса (опционально).
        
        Returns:
            Вектор обогащённого эмбеддинга.
        """
        # Получаем базовый эмбеддинг термина
        term_embedding = self.embedding_service.get_embedding(term)
        
        # Инициализируем WordNet при первом запросе
        self._init_wordnet()
        
        if self.wordnet_service is None:
            # WordNet недоступен, возвращаем базовый эмбеддинг
            return term_embedding
        
        # Получаем гиперонимы
        hypernyms = self.wordnet_service.get_hypernyms(term)
        
        if not hypernyms:
            # Гиперонимов нет, возвращаем базовый эмбеддинг
            return term_embedding
        
        # Ограничиваем число гиперонимов
        hypernyms = hypernyms[: self.max_hypernyms]
        
        # Получаем эмбеддинги гиперонимов
        hypernym_embeddings = []
        for hypernym in hypernyms:
            try:
                emb = self.embedding_service.get_embedding(hypernym)
                hypernym_embeddings.append(emb)
            except Exception as e:
                logger.debug(f"Не удалось получить эмбеддинг для '{hypernym}': {e}")
                continue
        
        if not hypernym_embeddings:
            return term_embedding
        
        # Вычисляем средний эмбеддинг гиперонимов
        mean_hypernym = np.mean(hypernym_embeddings, axis=0)
        
        # Комбинируем
        w = alpha if alpha is not None else self.alpha
        enriched = w * term_embedding + (1 - w) * mean_hypernym
        
        # Нормализуем
        norm = np.linalg.norm(enriched)
        if norm > 0:
            enriched = enriched / norm
        
        return enriched
    
    def get_enriched_domain_embedding(
        self,
        terms: list[str],
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        """Получение обогащённого эмбеддинга домена (набора терминов).
        
        Вычисляет центроид обогащённых эмбеддингов терминов домена.
        
        Args:
            terms: Список терминов домена.
            alpha: Переопределение веса (опционально).
        
        Returns:
            Вектор эмбеддинга домена.
        """
        if not terms:
            raise ValueError("Список терминов не может быть пустым")
        
        embeddings = []
        for term in terms:
            emb = self.get_enriched_embedding(term, alpha)
            embeddings.append(emb)
        
        # Центроид
        centroid = np.mean(embeddings, axis=0)
        
        # Нормализуем
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        
        return centroid
    
    def get_enrichment_info(self, term: str) -> dict:
        """Получение информации об обогащении термина.
        
        Args:
            term: Термин для анализа.
        
        Returns:
            Словарь с информацией:
            - term: исходный термин
            - hypernyms: список гиперонимов
            - hypernym_count: число найденных гиперонимов
            - enriched: True если обогащение применимо
        """
        self._init_wordnet()
        
        result = {
            "term": term,
            "hypernyms": [],
            "hypernym_count": 0,
            "enriched": False,
        }
        
        if self.wordnet_service is None:
            return result
        
        hypernyms = self.wordnet_service.get_hypernyms(term)
        
        if hypernyms:
            result["hypernyms"] = hypernyms[: self.max_hypernyms]
            result["hypernym_count"] = len(result["hypernyms"])
            result["enriched"] = True
        
        return result
