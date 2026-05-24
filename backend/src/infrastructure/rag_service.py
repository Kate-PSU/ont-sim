# backend/src/infrastructure/rag_service.py
# RAG-Centroids сервис
#
# Версия: 1.0
# Обновлено: 2026-04-15

"""
RAG-Centroids сервис для расчёта центроидов с использованием retrieval.

Использует retrieval для улучшения представления терминов через
nearest neighbor search в корпусе эмбеддингов.
"""

from pathlib import Path
from typing import Optional
import numpy as np

from .embedding_service import EmbeddingService
from .retrieval_service import RetrievalService, get_cached_retrieval, set_cached_retrieval


class RAGService:
    """RAG-Centroids сервис.
    
    Вычисляет центроиды доменов с использованием retrieval-augmented подхода:
    
    1. Для каждого термина находим k ближайших соседей в корпусе
    2. Агрегируем эмбеддинги термина и его соседей
    3. Центроид домена = среднее по всем агрегированным эмбеддингам
    
    Attributes:
        embedding_service: Сервис эмбеддингов
        retrieval_service: Retrieval сервис для kNN
        k_neighbors: Количество соседей для retrieval
    """
    
    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        k_neighbors: int = 5,
    ) -> None:
        """Инициализация.
        
        Args:
            embedding_service: Сервис эмбеддингов (создаётся автоматически если None)
            k_neighbors: Количество соседей для retrieval (по умолчанию 5)
        """
        self.embedding_service = embedding_service or EmbeddingService()
        self.k_neighbors = k_neighbors
        self._retrieval: Optional[RetrievalService] = None
        self._corpus_terms: list[str] = []
    
    def set_corpus(self, terms: list[str]) -> None:
        """Установить корпус терминов для retrieval.
        
        Args:
            terms: Список терминов корпуса
        """
        self._corpus_terms = terms
        
        # Создаём retrieval сервис с корпусом
        self._retrieval = RetrievalService(self.embedding_service)
        self._retrieval.build_index(terms, k=self.k_neighbors)
    
    def get_term_centroid(self, term: str) -> np.ndarray:
        """Получить улучшенный эмбеддинг термина через retrieval.
        
        Агрегирует эмбеддинг термина с его k ближайшими соседями:
        
        centroid(term) = mean(emb(term), emb(neighbor1), ..., emb(neighbork))
        
        Args:
            term: Термин для получения эмбеддинга
        
        Returns:
            Агрегированный эмбеддинг
        """
        # Базовый эмбеддинг термина
        base_emb = self.embedding_service.get_embedding(term)
        
        if self._retrieval is None or not self._corpus_terms:
            # Без retrieval - возвращаем базовый эмбеддинг
            return base_emb
        
        try:
            # Получаем соседей
            neighbors = self._retrieval.retrieve_neighbors(term, k=self.k_neighbors)
            
            if not neighbors:
                return base_emb
            
            # Собираем эмбеддинги термина и соседей
            embeddings = [base_emb]
            for neighbor_term, _ in neighbors:
                neighbor_emb = self.embedding_service.get_embedding(neighbor_term)
                embeddings.append(neighbor_emb)
            
            # Центроид = среднее
            centroid = np.mean(embeddings, axis=0)
            
            # Нормализуем
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            
            return centroid
        except Exception:
            # При ошибке возвращаем базовый эмбеддинг
            return base_emb
    
    def get_centroid(self, terms: list[str]) -> np.ndarray:
        """Вычислить центроид списка терминов через RAG-подход.
        
        Args:
            terms: Список терминов домена
        
        Returns:
            Центроид домена (нормализованный вектор)
        """
        if not terms:
            # Пустой список - возвращаем нулевой вектор нужной размерности
            return np.zeros(1024)  # SBERT_large dimension
        
        # Если corpus не установлен - используем термины как корпус
        if not self._corpus_terms:
            self.set_corpus(terms)
        
        # Получаем улучшенные эмбеддинги для всех терминов
        embeddings = []
        for i, term in enumerate(terms):
            emb = self.get_term_centroid(term)
            embeddings.append(emb)
            
            # Очищаем CUDA память каждые 10 терминов
            if i > 0 and i % 10 == 0:
                self._cleanup_memory()
        
        if not embeddings:
            return np.zeros(1024)
        
        # Центроид всех эмбеддингов
        centroid = np.mean(embeddings, axis=0)
        
        # Нормализуем
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        
        # Очищаем память после завершения
        self._cleanup_memory()
        
        return centroid
    
    def _cleanup_memory(self) -> None:
        """Очистка CUDA памяти."""
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def save_state(self, path: Path) -> None:
        """Сохранить состояние RAG сервиса.
        
        Args:
            path: Путь для сохранения
        """
        if self._retrieval is not None:
            cache_key = f"rag_{hash(tuple(self._corpus_terms))}"
            index_path = RetrievalService.get_index_path(cache_key)
            self._retrieval.save_index(index_path)
    
    @classmethod
    def load_state(
        cls,
        path: Path,
        embedding_service: Optional[EmbeddingService] = None,
    ) -> "RAGService":
        """Загрузить состояние RAG сервиса.
        
        Args:
            path: Путь к файлу состояния
            embedding_service: Сервис эмбеддингов
        
        Returns:
            RAGService с загруженным состоянием
        """
        service = cls(embedding_service=embedding_service)
        
        # Пытаемся загрузить индекс
        if path.exists():
            cache_key = f"rag_load_{path.stem}"
            retrieval = RetrievalService.load_index(path, service.embedding_service)
            if retrieval:
                service._retrieval = retrieval
        
        return service
