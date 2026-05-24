# backend/src/infrastructure/sklearn_tfidf.py
# Sklearn-based TF-IDF similarity service
#
# Версия: 1.1
# Обновлено: 2026-04-23
#
# =============================================================================
# TF-IDF BENCHMARK RECOMMENDATION
# =============================================================================
# Дата: 2026-04-23
# Датасеты: hj-rg (n=65), simlex999_ru (n=984)
#
# РЕЗУЛЬТАТЫ СРАВНЕНИЯ:
# +------------------+----------+----------+----------+
# | Датасет         | Метод    | N-grams   | Spearman |
# +------------------+----------+----------+----------+
# | hj-rg           | TfidfSvc | (2,4)    | -0.0576 |
# | hj-rg           | Sklearn  | (2, 4)   | 0.3313* |
# | hj-rg           | Sklearn  | (2, 5)   | 0.3307  |
# | simlex999_ru    | TfidfSvc | (2,4)    | 0.0545  |
# | simlex999_ru    | Sklearn  | (2, 4)   | 0.1280* |
# | simlex999_ru    | Sklearn  | (2, 5)   | 0.1277  |
# +------------------+----------+----------+----------+
#
# ВЫВОДЫ:
# - SklearnTfidfSimilarity значительно превосходит TfidfService
# - Лучший n-gram диапазон: (2, 4) для обоих датасетов
#
# PRIMARY CHOICE: SklearnTfidfSimilarity(ngram_range=(2, 4))
# =============================================================================

"""
Модуль для расчёта TF-IDF сходства с использованием sklearn TfidfVectorizer.

Поддерживает:
- Символьные n-граммы (для морфологически богатых языков, например русский)
- Словесные n-граммы (для английского)
- Автоматическое вычисление IDF из корпуса
- Косинусное сходство между терминами

Используется для задачи 113: исправление TF-IDF, который возвращает 0.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class SklearnTfidfResult:
    """Результат TF-IDF вычисления для корпуса."""
    corpus_size: int
    vocabulary_size: int
    idf_values: dict[str, float]
    tfidf_matrix_shape: tuple[int, int]


class SklearnTfidfSimilarity:
    """Сервис расчёта TF-IDF сходства на основе sklearn TfidfVectorizer.
    
    Attributes:
        ngram_range: Диапазон n-грамм (по умолчанию (1, 2))
        analyzer: 'word' для словесных n-грамм, 'char' для символьных
        min_df: Минимальная частота документа для включения в словарь
        max_df: Максимальная частота документа для включения в словарь
        language: Язык ('ru' или 'en')
    
    Example:
        >>> service = SklearnTfidfSimilarity(analyzer='char', ngram_range=(2, 4))
        >>> service.fit(['кот', 'собака', 'кошка'])
        >>> sim = service.get_similarity('кот', 'кошка')
        >>> print(f"Similarity: {sim:.3f}")
    """
    
    def __init__(
        self,
        ngram_range: tuple[int, int] = (1, 2),
        analyzer: str = 'char',  # 'word' или 'char'
        min_df: int = 1,
        max_df: float = 1.0,
        lowercase: bool = True,
        language: str = 'ru',
    ) -> None:
        self.ngram_range = ngram_range
        self.analyzer = analyzer
        self.min_df = min_df
        self.max_df = max_df
        self.lowercase = lowercase
        self.language = language
        
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._corpus: list[str] = []
        self._term_to_idx: dict[str, int] = {}
        self._fitted: bool = False
    
    def fit(self, corpus: list[str]) -> "SklearnTfidfSimilarity":
        """Обучение TF-IDF векторизатора на корпусе терминов.
        
        Args:
            corpus: Список терминов для обучения
            
        Returns:
            self для цепочки вызовов
        """
        if not corpus:
            return self
        
        # Сохраняем корпус
        self._corpus = list(corpus)
        self._term_to_idx = {term: i for i, term in enumerate(self._corpus)}
        
        # Создаём и обучаем векторизатор
        self._vectorizer = TfidfVectorizer(
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            lowercase=self.lowercase,
            norm='l2',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True,  # Используем 1 + log(tf) для TF
        )
        
        # Обучаем на корпусе терминов (каждый термин = "документ")
        self._vectorizer.fit(self._corpus)
        self._fitted = True
        
        return self
    
    def fit_from_domains(
        self,
        domains: dict[str, list[str]],
    ) -> "SklearnTfidfSimilarity":
        """Обучение на словаре доменов с терминами.
        
        Args:
            domains: Словарь {domain_name: [term1, term2, ...]}
            
        Returns:
            self для цепочки вызовов
        """
        all_terms: list[str] = []
        for domain_terms in domains.values():
            all_terms.extend(domain_terms)
        
        return self.fit(all_terms)
    
    @property
    def is_fitted(self) -> bool:
        """Проверка, обучен ли векторизатор."""
        return self._fitted and self._vectorizer is not None
    
    @property
    def vocabulary_size(self) -> int:
        """Размер словаря TF-IDF."""
        if self._vectorizer is None:
            return 0
        return len(self._vectorizer.vocabulary_)
    
    @property
    def corpus_size(self) -> int:
        """Размер корпуса (количество терминов)."""
        return len(self._corpus)
    
    def get_idf(self, term: str) -> float:
        """Получение IDF веса термина.
        
        Args:
            term: Термин
            
        Returns:
            IDF вес (0.0 если термин не в словаре)
        """
        if not self.is_fitted or self._vectorizer is None:
            return 0.0
        
        try:
            idx = self._vectorizer.vocabulary_.get(term.lower())
            if idx is None:
                return 0.0
            return self._vectorizer.idf_[idx]
        except Exception:
            return 0.0
    
    def get_tfidf_vector(self, term: str) -> Optional[np.ndarray]:
        """Получение TF-IDF вектора для термина.
        
        Args:
            term: Термин
            
        Returns:
            Вектор TF-IDF или None
        """
        if not self.is_fitted or self._vectorizer is None:
            return None
        
        try:
            vectors = self._vectorizer.transform([term])
            return vectors.toarray()[0]
        except Exception:
            return None
    
    def get_similarity(self, term1: str, term2: str) -> float:
        """Вычисление косинусного сходства между терминами.
        
        Args:
            term1: Первый термин
            term2: Второй термин
            
        Returns:
            Косинусное сходство в диапазоне [0, 1]
        """
        if not self.is_fitted or self._vectorizer is None:
            return 0.0
        
        try:
            # Трансформируем оба термина в TF-IDF векторы
            vectors = self._vectorizer.transform([term1, term2])
            vec1 = vectors[0].toarray().flatten()
            vec2 = vectors[1].toarray().flatten()
            
            # Косинусное сходство
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            cos_sim = np.dot(vec1, vec2) / (norm1 * norm2)
            
            # Нормализуем в [0, 1] (cosine ∈ [-1, 1])
            return float((cos_sim + 1) / 2)
            
        except Exception:
            return 0.0
    
    def get_similarities_batch(
        self,
        terms: list[str],
    ) -> np.ndarray:
        """Вычисление матрицы сходств между терминами.
        
        Args:
            terms: Список терминов
            
        Returns:
            Матрица сходств (n x n)
        """
        if not self.is_fitted or self._vectorizer is None:
            return np.zeros((len(terms), len(terms)))
        
        try:
            vectors = self._vectorizer.transform(terms)
            
            # Косинусное сходство между всеми парами
            from sklearn.metrics.pairwise import cosine_similarity
            sim_matrix = cosine_similarity(vectors)
            
            # Нормализуем в [0, 1]
            sim_matrix = (sim_matrix + 1) / 2
            
            return sim_matrix
            
        except Exception:
            return np.zeros((len(terms), len(terms)))
    
    def get_info(self) -> SklearnTfidfResult:
        """Получение информации о модели."""
        if not self.is_fitted or self._vectorizer is None:
            return SklearnTfidfResult(
                corpus_size=0,
                vocabulary_size=0,
                idf_values={},
                tfidf_matrix_shape=(0, 0),
            )
        
        idf_values = {
            term: float(self._vectorizer.idf_[idx])
            for term, idx in self._vectorizer.vocabulary_.items()
        }
        
        return SklearnTfidfResult(
            corpus_size=self.corpus_size,
            vocabulary_size=self.vocabulary_size,
            idf_values=idf_values,
            tfidf_matrix_shape=(self.corpus_size, self.vocabulary_size),
        )
    
    def most_similar(
        self,
        term: str,
        top_k: int = 5,
        exclude_self: bool = True,
    ) -> list[tuple[str, float]]:
        """Найти наиболее похожие термины.
        
        Args:
            term: Целевой термин
            top_k: Количество результатов
            exclude_self: Исключить сам термин из результатов
            
        Returns:
            Список кортежей (term, similarity)
        """
        if not self.is_fitted:
            return []
        
        try:
            vectors = self._vectorizer.transform([term] + self._corpus)
            target_vec = vectors[0].toarray().flatten()
            corpus_vecs = vectors[1:].toarray()
            
            similarities = []
            for i, vec in enumerate(corpus_vecs):
                norm1 = np.linalg.norm(target_vec)
                norm2 = np.linalg.norm(vec)
                if norm1 > 0 and norm2 > 0:
                    sim = np.dot(target_vec, vec) / (norm1 * norm2)
                    sim = (sim + 1) / 2  # Нормализация в [0, 1]
                    similarities.append((self._corpus[i], float(sim)))
            
            # Сортируем по убыванию сходства
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Исключаем сам термин если нужно
            if exclude_self:
                similarities = [(t, s) for t, s in similarities if t != term]
            
            return similarities[:top_k]
            
        except Exception:
            return []
    
    def clear(self) -> None:
        """Очистка модели."""
        self._vectorizer = None
        self._corpus = []
        self._term_to_idx = {}
        self._fitted = False
    
    def set_idf_from(self, other: "SklearnTfidfSimilarity") -> "SklearnTfidfSimilarity":
        """Установить IDF веса из другой модели TF-IDF.
        
        Используется для применения IDF весов от Wikipedia корпуса
        к TF-IDF для терминов.
        
        Args:
            other: Другая модель SklearnTfidfSimilarity
            
        Returns:
            self для цепочки вызовов
        """
        if not other.is_fitted or other._vectorizer is None:
            return self
        
        # Получаем IDF веса из другой модели
        other_idf = other._vectorizer.idf_
        other_vocab = other._vectorizer.vocabulary_
        
        # Создаём новый векторизатор с этими IDF весами
        self._vectorizer = TfidfVectorizer(
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            lowercase=self.lowercase,
            norm='l2',
            use_idf=True,
            smooth_idf=False,  # IDF уже из другой модели
            sublinear_tf=True,
        )
        
        # Устанавливаем словарь и IDF веса после создания
        self._vectorizer.vocabulary_ = other_vocab.copy()
        self._vectorizer.idf_ = other_idf.copy()
        self._vectorizer._tfidf._idf_diag = None  # Сброс кэша
        
        self._fitted = True
        return self
    
    def get_idf_weights(self) -> Optional[np.ndarray]:
        """Получить IDF веса текущей модели.
        
        Returns:
            Массив IDF весов или None
        """
        if self._vectorizer is None:
            return None
        return self._vectorizer.idf_.copy()


def create_tfidf_service(
    language: str = 'ru',
    ngram_range: Optional[tuple[int, int]] = None,
) -> SklearnTfidfSimilarity:
    """Фабричная функция для создания TF-IDF сервиса.

    WARNING: Для английского языка char n-grams (2,4) неэффективны.
    Семантически синонимичные слова часто морфологически различны
    ("smart" vs "intelligent" = 0.5). Ожидаемые метрики: Spearman ~0.04.
    Рекомендация: использовать word n-grams или контекстные эмбеддинги для English.
    
    Args:
        language: Язык ('ru' для русского, 'en' для английского)
        ngram_range: Переопределение диапазона n-грамм
        
    Returns:
        Настроенный SklearnTfidfSimilarity
    """
    if language == 'ru':
        # Для русского используем символьные n-граммы
        # Они лучше работают с морфологией
        ngram = ngram_range or (2, 4)  # RECOMMENDED: benchmark showed (2,4) > (2,5)
        analyzer = 'char'
    else:
        # Для английского используем символьные n-граммы (как для русского)
        ngram = ngram_range or (2, 4)
        analyzer = 'char'
    
    return SklearnTfidfSimilarity(
        ngram_range=ngram,
        analyzer=analyzer,
        min_df=1,
        max_df=1.0,
        lowercase=True,
        language=language,
    )


__all__ = [
    "SklearnTfidfSimilarity",
    "SklearnTfidfResult",
    "create_tfidf_service",
]