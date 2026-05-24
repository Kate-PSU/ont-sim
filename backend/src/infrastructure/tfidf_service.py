# backend/src/infrastructure/tfidf_service.py
# Сервис TF-IDF весов и Z-score нормализации
#
# Версия: 6.0
# Обновлено: 2026-04-12

"""
Модуль для расчёта TF-IDF весов и нормализации терминов.

Использует комбинированный подход:
- Символьные n-граммы для булевского TF-IDF
- Jaccard similarity по символам для оценки сходства
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TermWeight:
    """Вес термина с метаданными."""
    term: str
    weight: float
    tfidf: float
    idf: float
    z_score: float


@dataclass
class TfidfResult:
    """Результат TF-IDF расчёта для домена."""
    domain: str
    weights: dict[str, float]
    filtered_terms: list[str]
    matrix_shape: tuple[int, int]
    domain_terms: list[str]


def _jaccard_similarity(s1: str, s2: str) -> float:
    """Вычисление Jaccard similarity по символам."""
    set1 = set(s1.lower())
    set2 = set(s2.lower())
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


class TfidfService:
    """Сервис расчёта TF-IDF весов на основе символьных n-грамм."""
    
    def __init__(
        self,
        idf_threshold: float = 0.0,
        z_score_threshold: float = -2.0,
        documents: Optional[list[list[str]]] = None,
    ) -> None:
        self.idf_threshold = idf_threshold
        self.z_score_threshold = z_score_threshold
        self.vectorizer = None
        self._term_idf: Optional[dict[str, float]] = None
        self._documents: list[list[str]] = documents if documents else []
        self._terms_cache: dict[str, np.ndarray] = {}
        self._ngram_vocab: list[str] = []
        
        self.documents = self._documents
        self.idf_scores = self._term_idf if self._term_idf else {}
    
    @property
    def idf_scores(self) -> dict[str, float]:
        return self._term_idf if self._term_idf else {}
    
    @idf_scores.setter
    def idf_scores(self, value: dict[str, float]) -> None:
        self._term_idf = value
    
    def fit(self, documents: list[list[str]]) -> "TfidfService":
        """Обучение на корпусе документов."""
        self._documents = documents
        self.documents = self._documents
        return self
    
    def _compute_idf(self) -> None:
        """Вычисление IDF для всех терминов."""
        pass
    
    def get_idf(self, term: str) -> float:
        if self._term_idf is None:
            return 0.0
        return self._term_idf.get(term, 0.0)
    
    def _get_char_ngrams(self, term: str) -> set:
        """Получение символьных n-грамм."""
        ngrams = set()
        for n in range(2, 4):
            for i in range(len(term) - n + 1):
                ngrams.add(term[i:i+n])
        return ngrams
    
    def fit_terms(self, terms: list[str]) -> "TfidfService":
        """Обучение на списке терминов.
        
        Используем комбинацию бинарных n-gram векторов и Jaccard similarity.
        """
        self._documents = terms
        self._terms = terms
        self._terms_cache.clear()
        
        # Собираем все уникальные n-граммы
        all_ngrams = set()
        term_ngrams_list = {}
        for term in terms:
            ngrams = self._get_char_ngrams(term)
            term_ngrams_list[term] = ngrams
            all_ngrams.update(ngrams)
        
        # Сортируем для консистентности
        self._ngram_vocab = sorted(all_ngrams)
        ngram_to_idx = {ng: i for i, ng in enumerate(self._ngram_vocab)}
        
        # Создаём бинарные вектора для каждого термина
        for term in terms:
            ngrams = term_ngrams_list[term]
            vec = np.zeros(len(self._ngram_vocab))
            for ng in ngrams:
                if ng in ngram_to_idx:
                    vec[ngram_to_idx[ng]] = 1.0
            self._terms_cache[term] = vec
        
        return self
    
    def get_vector(self, term: str) -> Optional[np.ndarray]:
        """Получение char-ngram вектора для термина."""
        if not self._terms_cache:
            return None
        
        if term in self._terms_cache:
            return self._terms_cache[term]
        
        ngrams = self._get_char_ngrams(term)
        vec = np.zeros(len(self._ngram_vocab))
        for ng in ngrams:
            if ng in self._ngram_vocab:
                idx = self._ngram_vocab.index(ng)
                vec[idx] = 1.0
        return vec
    
    def get_similarity(self, term1: str, term2: str) -> float:
        """Вычисление сходства между терминами.
        
        Использует комбинацию:
        - Jaccard similarity по символам (для случаев без общих n-gram)
        - Cosine similarity по n-gram векторам
        """
        # Сначала пробуем cosine similarity по n-gram
        vec1 = self.get_vector(term1)
        vec2 = self.get_vector(term2)
        
        if vec1 is None or vec2 is None:
            return 0.0
        
        # Cosine similarity
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            # Если один из векторов пустой, используем Jaccard
            return _jaccard_similarity(term1, term2)
        
        cos_sim = np.dot(vec1, vec2) / (norm1 * norm2)
        
        # Если cosine даёт 0, пробуем Jaccard по символам
        if cos_sim == 0 and norm1 > 0 and norm2 > 0:
            return _jaccard_similarity(term1, term2)
        
        return cos_sim
    
    def calculate_tfidf(self, domain_terms: list[str]) -> dict[str, float]:
        """Расчёт TF-IDF весов для терминов домена."""
        if not self._terms_cache:
            return {}
        
        weights = {}
        for term in domain_terms:
            vec = self.get_vector(term)
            weights[term] = float(np.sum(vec)) if vec is not None else 0.0
        
        return weights
    
    def normalize_zscore(self, weights: dict[str, float]) -> dict[str, float]:
        """Z-score нормализация весов."""
        if not weights:
            return {}
        
        values = np.array(list(weights.values()))
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return {term: 0.0 for term in weights}
        
        z_scores = (values - mean) / std
        return dict(zip(weights.keys(), z_scores))
    
    def filter_by_idf(
        self,
        terms: list[str],
        threshold: Optional[float] = None,
    ) -> list[str]:
        """Фильтрация терминов по IDF порогу."""
        thr = threshold if threshold is not None else self.idf_threshold
        return [t for t in terms if self.get_idf(t) >= thr]
    
    def filter_by_zscore(
        self,
        z_scores: dict[str, float],
        threshold: Optional[float] = None,
    ) -> list[str]:
        """Фильтрация терминов по Z-score."""
        thr = threshold if threshold is not None else self.z_score_threshold
        return [t for t, z in z_scores.items() if z >= thr]
    
    def process_domain(
        self,
        terms: list[str],
        domain: str = "default",
        apply_filters: bool = True,
    ) -> TfidfResult:
        """Полная обработка домена."""
        weights = self.calculate_tfidf(terms)
        z_scores = self.normalize_zscore(weights)
        
        filtered = terms
        if apply_filters:
            filtered = self.filter_by_idf(filtered)
            filtered = self.filter_by_zscore(z_scores)
        
        final_weights = {t: weights[t] for t in filtered if t in weights}
        
        return TfidfResult(
            domain=domain,
            weights=final_weights,
            filtered_terms=filtered,
            matrix_shape=(1, len(terms)),
            domain_terms=terms,
        )
    
    def get_weights_for_centroid(
        self,
        terms: list[str],
    ) -> np.ndarray:
        """Получение весов в виде массива для расчёта центроида."""
        weights = self.calculate_tfidf(terms)
        return np.array([weights.get(t, 0.0) for t in terms])


def calculate_term_frequency(
    terms: list[str],
    normalize: bool = True,
) -> dict[str, float]:
    """Расчёт частоты терминов."""
    if not terms:
        return {}
    
    from collections import Counter
    counts = Counter(terms)
    total = sum(counts.values())
    
    frequencies = {}
    for term, count in counts.items():
        freq = count / total if normalize else count
        frequencies[term] = freq
    
    return frequencies