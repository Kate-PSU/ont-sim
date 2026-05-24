# backend/src/application/similarity_service.py
# Сервис расчёта близости доменов
#
# Версия: 1.0
# Обновлено: 2026-04-06

"""
Сервис для расчёта семантической близости между предметными областями.
Использует эмбеддинги и центроиды доменов.
"""

from typing import Optional

import numpy as np

from ..domain import Domain, Similarity, GraphData


class SimilarityService:
    """Сервис расчёта близости.
    
    Вычисляет попарную близость доменов на основе
    эмбеддингов их терминов.
    
    Атрибуты:
        metric: Метрика близости ('cosine' или 'euclidean').
    """
    
    def __init__(self, metric: str = "cosine") -> None:
        """Инициализация сервиса.
        
        Args:
            metric: Метрика близости ('cosine' или 'euclidean').
        """
        self.metric = metric
    
    def calculate_similarity(
        self,
        centroid1: np.ndarray,
        centroid2: np.ndarray,
    ) -> float:
        """Расчёт близости между двумя центроидами.
        
        Args:
            centroid1: Центроид первого домена.
            centroid2: Центроид второго домена.
        
        Returns:
            Значение близости в диапазоне [0, 1].
        """
        if self.metric == "cosine":
            return self._cosine_similarity(centroid1, centroid2)
        elif self.metric == "euclidean":
            return self._euclidean_similarity(centroid1, centroid2)
        else:
            raise ValueError(f"Неизвестная метрика: {self.metric}")
    
    def _cosine_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray,
    ) -> float:
        """Косинусное сходство.
        
        Args:
            vec1: Первый вектор.
            vec2: Второй вектор.
        
        Returns:
            Косинус угла между векторами.
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def _euclidean_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray,
    ) -> float:
        """Евклидово сходство.
        
        Args:
            vec1: Первый вектор.
            vec2: Второй вектор.
        
        Returns:
            Значение в диапазоне [0, 1].
        """
        distance = np.linalg.norm(vec1 - vec2)
        
        # Special case: если оба вектора нулевые
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 and norm2 == 0:
            return 1.0  # Идентичные нулевые векторы
        
        # Если только один нулевой — максимальное расстояние
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(1 / (1 + distance))
    
    def build_graph(
        self,
        domains: list[Domain],
        centroids: dict[str, np.ndarray],
        threshold: float = 0.5,
    ) -> GraphData:
        """Построение графа близости.
        
        Args:
            domains: Список доменов.
            centroids: Словарь центроидов {домен: вектор}.
            threshold: Порог близости для включения ребра.
        
        Returns:
            Данные графа для визуализации.
        """
        nodes = [{"id": d.name, "label": d.name} for d in domains]
        edges = []
        
        domain_names = list(centroids.keys())
        for i, dom1 in enumerate(domain_names):
            for dom2 in domain_names[i + 1:]:
                sim = self.calculate_similarity(
                    centroids[dom1],
                    centroids[dom2],
                )
                if sim >= threshold:
                    edges.append({
                        "source": dom1,
                        "target": dom2,
                        "weight": round(sim, 4),
                    })
        
        return GraphData(nodes=nodes, edges=edges)
    
    def calculate_similarity_matrix(
        self,
        centroids: dict[str, np.ndarray],
    ) -> tuple[list[str], np.ndarray]:
        """Расчёт матрицы попарной близости.
        
        Args:
            centroids: Словарь {домен: центроид}.
        
        Returns:
            Кортеж (список_доменов, матрица_близостей).
            Матрица размером N x N.
        """
        domain_names = list(centroids.keys())
        n = len(domain_names)
        
        if n == 0:
            return [], np.array([])
        
        # Инициализируем матрицу
        matrix = np.zeros((n, n))
        
        # Заполняем матрицу
        for i, d1 in enumerate(domain_names):
            for j, d2 in enumerate(domain_names):
                if i == j:
                    # Близость домена к себе = 1
                    matrix[i][j] = 1.0
                elif i < j:
                    sim = self.calculate_similarity(
                        centroids[d1],
                        centroids[d2],
                    )
                    matrix[i][j] = sim
                    matrix[j][i] = sim  # Симметричность
                # j < i уже заполнен
        
        return domain_names, matrix
    
    def get_similarity_pairs(
        self,
        centroids: dict[str, np.ndarray],
        threshold: float = 0.0,
        top_n: int | None = None,
    ) -> list[dict]:
        """Получение пар доменов с близостью выше порога.
        
        Args:
            centroids: Словарь {домен: центроид}.
            threshold: Минимальная близость.
            top_n: Вернуть только top-N пар.
        
        Returns:
            Список пар [{domain1, domain2, score}, ...], отсортированный по score.
        """
        domain_names = list(centroids.keys())
        pairs = []
        
        for i, d1 in enumerate(domain_names):
            for d2 in domain_names[i + 1:]:
                sim = self.calculate_similarity(
                    centroids[d1],
                    centroids[d2],
                )
                if sim >= threshold:
                    pairs.append({
                        "domain1": d1,
                        "domain2": d2,
                        "score": round(sim, 4),
                    })
        
        # Сортируем по убыванию близости
        pairs.sort(key=lambda x: x["score"], reverse=True)
        
        if top_n is not None:
            pairs = pairs[:top_n]
        
        return pairs
