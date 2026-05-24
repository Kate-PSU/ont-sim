# backend/src/application/centroid_service.py
# Сервис расчёта центроидов доменов
#
# Версия: 1.2
# Обновлено: 2026-04-15
# Изменения: calculate_centroid теперь принимает list[np.ndarray] на вход

"""
Модуль для расчёта центроидов предметных областей.

Центроид — взвешенное или невзвешенное среднее векторов терминов домена.
Используется для расчёта близости между доменами.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Union

import numpy as np

from ..domain import Domain


@dataclass
class CentroidResult:
    """Результат расчёта центроида."""
    domain: str
    centroid: np.ndarray
    terms_count: int
    used_weights: bool


class CentroidService:
    """Сервис расчёта центроидов.
    
    Центроид домена — это усреднённый вектор всех его терминов.
    Может быть взвешенным (TF-IDF) или простым средним.
    
    Атрибуты:
        weights: Опциональные веса для взвешенного среднего.
        idf_threshold: Порог IDF для фильтрации терминов.
        z_score_threshold: Порог Z-score для нормализации.
        similarity_metric: Метрика близости ('cosine' или 'euclidean').
    """
    
    def __init__(
        self,
        weights: Optional[np.ndarray] = None,
        idf_threshold: Optional[float] = None,
        z_score_threshold: Optional[float] = None,
        similarity_metric: Literal["cosine", "euclidean"] = "cosine",
    ) -> None:
        """Инициализация сервиса.
        
        Args:
            weights: Веса для взвешенного среднего (N,).
                     Если None — используется простое среднее.
            idf_threshold: Порог IDF для фильтрации (из настроек).
            z_score_threshold: Порог Z-score для нормализации (из настроек).
            similarity_metric: Метрика близости ('cosine' или 'euclidean').
        """
        self.weights = weights
        self.idf_threshold = idf_threshold
        self.z_score_threshold = z_score_threshold
        self.similarity_metric = similarity_metric
    
    def calculate_centroid(
        self,
        embeddings: Union[np.ndarray, list[np.ndarray]],
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Расчёт центроида для массива эмбеддингов.
        
        Args:
            embeddings: Матрица эмбеддингов (N x D) или список np.ndarray.
            weights: Веса для взвешенного среднего (N,).
                    Если None — используется self.weights.
        
        Returns:
            Центроид (вектор D,).
        
        Raises:
            ValueError: Если embeddings пустой.
        """
        # Конвертируем список в numpy массив если нужно
        if isinstance(embeddings, list):
            if len(embeddings) == 0:
                raise ValueError("Embeddings не может быть пустым")
            embeddings = np.array(embeddings)
        
        if embeddings.size == 0:
            raise ValueError("Embeddings не может быть пустым")
        
        if embeddings.ndim == 1:
            # Один вектор — возвращаем его
            return embeddings
        
        # Используем переданные веса или self.weights
        w = weights if weights is not None else self.weights
        
        if w is None:
            # Простое среднее
            return embeddings.mean(axis=0)
        else:
            # Взвешенное среднее
            return np.average(embeddings, axis=0, weights=w)
    
    def calculate_domain_centroid(
        self,
        domain: Domain,
        embeddings: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> CentroidResult:
        """Расчёт центроида для домена.
        
        Args:
            domain: Домен с терминами.
            embeddings: Матрица эмбеддингов терминов (N x D).
            weights: Веса (N,).
        
        Returns:
            Результат с центроидом.
        """
        centroid = self.calculate_centroid(embeddings, weights)
        used_weights = weights is not None or self.weights is not None
        
        return CentroidResult(
            domain=domain.name,
            centroid=centroid,
            terms_count=len(domain.terms),
            used_weights=used_weights,
        )
    
    def calculate_all_centroids(
        self,
        domains: list[Domain],
        domain_embeddings: dict[str, np.ndarray],
        domain_weights: Optional[dict[str, np.ndarray]] = None,
    ) -> dict[str, np.ndarray]:
        """Расчёт центроидов для всех доменов.
        
        Args:
            domains: Список доменов.
            domain_embeddings: Словарь {домен: эмбеддинги} (N x D).
            domain_weights: Опциональные веса {домен: веса}.
        
        Returns:
            Словарь {домен: центроид}.
        """
        centroids = {}
        
        for domain in domains:
            embeddings = domain_embeddings.get(domain.name)
            if embeddings is None:
                continue
            
            weights = None
            if domain_weights is not None:
                weights = domain_weights.get(domain.name)
            
            centroids[domain.name] = self.calculate_centroid(embeddings, weights)
        
        return centroids


def calculate_centroid_batch(
    embeddings_list: list[np.ndarray],
    weights_list: Optional[list[np.ndarray]] = None,
) -> list[np.ndarray]:
    """Расчёт центроидов для списка массивов эмбеддингов.
    
    Args:
        embeddings_list: Список матриц эмбеддингов.
        weights_list: Опциональный список весов.
    
    Returns:
        Список центроидов.
    """
    centroids = []
    weights = weights_list
    
    for i, embeddings in enumerate(embeddings_list):
        w = None
        if weights is not None and i < len(weights):
            w = weights[i]
        
        if embeddings.size == 0:
            continue
        
        if embeddings.ndim == 1:
            centroids.append(embeddings)
        else:
            centroid = embeddings.mean(axis=0) if w is None else np.average(embeddings, axis=0, weights=w)
            centroids.append(centroid)
    
    return centroids


def cosine_similarity_between_centroids(
    centroids: dict[str, np.ndarray],
) -> dict[tuple[str, str], float]:
    """Расчёт попарной близости центроидов.
    
    Args:
        centroids: Словарь {домен: центроид}.
    
    Returns:
        Словарь {(домен1, домен2): близость}.
    """
    from ..application.similarity_service import SimilarityService
    
    service = SimilarityService(metric="cosine")
    result = {}
    
    domain_names = list(centroids.keys())
    for i, d1 in enumerate(domain_names):
        for d2 in domain_names[i + 1:]:
            similarity = service.calculate_similarity(
                centroids[d1],
                centroids[d2],
            )
            result[(d1, d2)] = similarity
    
    return result
