# backend/src/infrastructure/retrieval_service.py
# Retrieval-сервис для RAG с FAISS
#
# Версия: 2.0
# Обновлено: 2026-04-13

"""
Модуль retrieval-сервиса для RAG.

Использует FAISS для nearest neighbor search по эмбеддингам.
Поддерживает сохранение/загрузку индекса.
"""

from pathlib import Path
from typing import Optional
import pickle
import numpy as np
import faiss


# Глобальный кеш индексов для RAG
_index_cache: dict[str, "RetrievalService"] = {}


def get_cached_retrieval(cache_key: str) -> Optional["RetrievalService"]:
    """Получить кешированный RetrievalService."""
    return _index_cache.get(cache_key)


def set_cached_retrieval(cache_key: str, service: "RetrievalService") -> None:
    """Сохранить RetrievalService в кеш."""
    _index_cache[cache_key] = service


class RetrievalService:
    """Сервис retrieval для RAG-подхода.
    
    Attributes:
        embedding_service: Сервис эмбеддингов
        index: FAISS индекс
        embeddings_matrix: Матрица эмбеддингов
        terms: Список терминов
    """
    
    def __init__(self, embedding_service) -> None:
        """Инициализация.
        
        Args:
            embedding_service: EmbeddingService для получения эмбеддингов
        """
        self.embedding_service = embedding_service
        self._index = None
        self.embeddings_matrix: Optional[np.ndarray] = None
        self.terms: list[str] = []
    
    @property
    def index(self):
        """FAISS индекс."""
        return self._index
    
    def build_index(self, terms: list[str], k: int = 5) -> None:
        """Построение FAISS индекса.
        
        Оптимизация: если terms не изменились, не перестраиваем индекс.
        
        Args:
            terms: Список терминов для индексации
            k: Количество соседей для поиска
        """
        # Оптимизация: если terms те же, не перестраиваем
        if self._index is not None and self.terms == terms:
            return
        
        self.terms = terms
        
        if not terms:
            self._index = None
            self.embeddings_matrix = None
            return
        
        embeddings = []
        
        # Очищаем CUDA память перед построением индекса
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        for term in terms:
            emb = self.embedding_service.get_embedding(term)
            embeddings.append(emb)
            
        # Очищаем CUDA память после построения эмбеддингов
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.embeddings_matrix = np.array(embeddings).astype('float32')
        
        # Нормализуем для cosine similarity
        faiss.normalize_L2(self.embeddings_matrix)
        
        # Создаём IndexFlatIP (inner product = cosine на нормализованных)
        dimension = self.embeddings_matrix.shape[1]
        self._index = faiss.IndexFlatIP(dimension)
        self._index.add(self.embeddings_matrix)
    
    def retrieve_neighbors(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """Поиск k ближайших соседей.
        
        Args:
            query: Запросный термин
            k: Количество соседей
        
        Returns:
            Список кортежей (term, similarity)
        """
        if self._index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        query_emb = self.embedding_service.get_embedding(query)
        query_vec = np.array([query_emb]).astype('float32')
        faiss.normalize_L2(query_vec)
        
        distances, indices = self._index.search(query_vec, k + 1)  # +1 потому что сам термин тоже найдётся
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.terms):
                term = self.terms[idx]
                if term != query:  # Исключаем сам запрос
                    results.append((term, float(dist)))
        
        # Очищаем CUDA память после retrieval
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return results[:k]
    
    def get_retrieved_context(self, term: str, k: int = 5) -> list[str]:
        """Получить контекст retrieved терминов.
        
        Args:
            term: Термин
            k: Количество retrieved
        
        Returns:
            Список retrieved терминов
        """
        neighbors = self.retrieve_neighbors(term, k)
        return [t[0] for t in neighbors]
    
    def retrieve_neighbors_batch(self, queries: list[str], k: int = 5) -> dict[str, list[tuple[str, float]]]:
        """Batch retrieval для нескольких запросов.
        
        Оптимизация: все эмбеддинги и FAISS search выполняются за один раз.
        
        Args:
            queries: Список запросных терминов
            k: Количество соседей
        
        Returns:
            Словарь {query_term: [(neighbor_term, similarity), ...]}
        """
        if self._index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        if not queries:
            return {}
        
        # Batch эмбеддинги для всех запросов
        query_embs = self.embedding_service.get_embeddings_batch(queries)
        query_vecs = np.array(query_embs).astype('float32')
        faiss.normalize_L2(query_vecs)
        
        # FAISS batch search для всех запросов сразу
        n_queries = len(queries)
        distances, indices = self._index.search(query_vecs, k + 1)  # +1 потому что сам термин тоже найдётся
        
        # Собираем результаты
        results = {}
        for i, query in enumerate(queries):
            query_results = []
            for dist, idx in zip(distances[i], indices[i]):
                if idx < len(self.terms):
                    term = self.terms[idx]
                    if term != query:  # Исключаем сам запрос
                        query_results.append((term, float(dist)))
            results[query] = query_results[:k]
        
        return results
    
    def get_retrieved_context_batch(self, terms: list[str], k: int = 5) -> dict[str, list[str]]:
        """Batch получение контекста retrieved для нескольких терминов.
        
        Args:
            terms: Список терминов
            k: Количество retrieved
        
        Returns:
            Словарь {term: [retrieved_terms...]}
        """
        neighbors_dict = self.retrieve_neighbors_batch(terms, k)
        return {term: [t[0] for t in neighbors] for term, neighbors in neighbors_dict.items()}
    
    def save_index(self, path: Path) -> None:
        """Сохранить FAISS индекс на диск.
        
        Args:
            path: Путь для сохранения файла индекса (.index)
        """
        if self._index is None:
            raise ValueError("Index not built. Cannot save None index.")
        
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))
        
        # Сохраняем метаданные: terms и embeddings (для валидации при загрузке)
        metadata_path = path.with_suffix('.meta.pkl')
        metadata = {
            'terms': self.terms,
            'embeddings_shape': self.embeddings_matrix.shape if self.embeddings_matrix is not None else None,
        }
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
    
    @classmethod
    def load_index(cls, path: Path, embedding_service) -> Optional["RetrievalService"]:
        """Загрузить FAISS индекс с диска.
        
        Args:
            path: Путь к файлу индекса (.index)
            embedding_service: EmbeddingService для работы с эмбеддингами
        
        Returns:
            RetrievalService с загруженным индексом или None при ошибке
        """
        if not path.exists():
            return None
        
        metadata_path = path.with_suffix('.meta.pkl')
        if not metadata_path.exists():
            return None
        
        try:
            # Загружаем метаданные
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            # Загружаем FAISS индекс
            index = faiss.read_index(str(path))
            
            # Восстанавливаем embeddings из terms
            service = cls(embedding_service)
            service._index = index
            service.terms = metadata['terms']
            
            # Восстанавливаем embeddings матрицу
            if metadata['embeddings_shape'] and service.terms:
                embeddings = []
                for term in service.terms:
                    emb = embedding_service.get_embedding(term)
                    embeddings.append(emb)
                service.embeddings_matrix = np.array(embeddings).astype('float32')
            
            return service
        except Exception as e:
            print(f"[RetrievalService] Failed to load index: {e}")
            return None
    
    @staticmethod
    def get_index_path(cache_key: str, base_dir: Path = None) -> Path:
        """Получить путь для сохранения индекса.
        
        Args:
            cache_key: Уникальный ключ (dataset + lang + hash)
            base_dir: Базовая директория для хранения
        
        Returns:
            Путь к файлу индекса
        """
        if base_dir is None:
            base_dir = Path(__file__).parent.parent.parent.parent / "data" / "rag_indices"
        return base_dir / f"{cache_key}.index"
