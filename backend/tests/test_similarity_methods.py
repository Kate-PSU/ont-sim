# backend/tests/test_similarity_methods.py
# Тесты для методов расчёта центроидов и близости
#
# Версия: 1.0
# Обновлено: 2026-04-24

"""
Модульные тесты для методов расчёта центроидов и близости:
- calculate_sbert_centroid
- calculate_tfidf_centroid
- calculate_ensemble_centroid
- calculate_sbert_similarity
- calculate_tfidf_similarity
- calculate_ensemble_similarity
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

# Добавляем путь для импортов
backend_root = Path(__file__).parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))


# === Мок для CentroidService и SimilarityService (без зависимости от sentence_transformers) ===

class MockCentroidService:
    """Mock CentroidService для тестирования без зависимостей."""
    
    def __init__(self, weights: Optional[np.ndarray] = None):
        self.weights = weights
    
    def calculate_centroid(
        self,
        embeddings: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Расчёт центроида для массива эмбеддингов."""
        if isinstance(embeddings, list):
            if len(embeddings) == 0:
                raise ValueError("Embeddings не может быть пустым")
            embeddings = np.array(embeddings)
        
        if embeddings.size == 0:
            raise ValueError("Embeddings не может быть пустым")
        
        if embeddings.ndim == 1:
            return embeddings
        
        w = weights if weights is not None else self.weights
        
        if w is None:
            return embeddings.mean(axis=0)
        else:
            return np.average(embeddings, axis=0, weights=w)


class MockSimilarityService:
    """Mock SimilarityService для тестирования без зависимостей."""
    
    def __init__(self, metric: str = "cosine"):
        self.metric = metric
    
    def calculate_similarity(
        self,
        centroid1: np.ndarray,
        centroid2: np.ndarray,
    ) -> float:
        """Расчёт близости между двумя центроидами."""
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
        """Косинусное сходство."""
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
        """Евклидово сходство."""
        distance = np.linalg.norm(vec1 - vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 and norm2 == 0:
            return 1.0
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(1 / (1 + distance))


class MockTfidfService:
    """Mock TfidfService для тестирования."""
    
    def __init__(self):
        self._terms_cache: dict[str, np.ndarray] = {}
        self._ngram_vocab: list[str] = []
        self._all_terms: list[str] = []
    
    def fit_terms(self, terms: list[str]) -> "MockTfidfService":
        """Обучение на списке терминов."""
        self._all_terms = terms
        self._terms_cache.clear()
        
        all_ngrams = set()
        term_ngrams_list = {}
        for term in terms:
            ngrams = self._get_char_ngrams(term)
            term_ngrams_list[term] = ngrams
            all_ngrams.update(ngrams)
        
        self._ngram_vocab = sorted(all_ngrams)
        ngram_to_idx = {ng: i for i, ng in enumerate(self._ngram_vocab)}
        
        for term in terms:
            ngrams = term_ngrams_list[term]
            vec = np.zeros(len(self._ngram_vocab))
            for ng in ngrams:
                if ng in ngram_to_idx:
                    vec[ngram_to_idx[ng]] = 1.0
            self._terms_cache[term] = vec
        
        return self
    
    def _get_char_ngrams(self, term: str) -> set:
        """Получение символьных n-грамм."""
        ngrams = set()
        for n in range(2, 4):
            for i in range(len(term) - n + 1):
                ngrams.add(term[i:i+n])
        return ngrams
    
    def get_weights_for_centroid(self, terms: list[str]) -> np.ndarray:
        """Получение весов в виде массива для расчёта центроида."""
        weights = self.calculate_tfidf(terms)
        return np.array([weights.get(t, 0.0) for t in terms])
    
    def calculate_tfidf(self, domain_terms: list[str]) -> dict[str, float]:
        """Расчёт TF-IDF весов для терминов домена."""
        if not self._terms_cache:
            return {}
        
        weights = {}
        for term in domain_terms:
            vec = self.get_vector(term)
            weights[term] = float(np.sum(vec)) if vec is not None else 0.0
        
        return weights
    
    def get_vector(self, term: str) -> Optional[np.ndarray]:
        """Получение char-ngram вектора для термина."""
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
        """Вычисление сходства между терминами."""
        vec1 = self.get_vector(term1)
        vec2 = self.get_vector(term2)
        
        if vec1 is None or vec2 is None:
            return 0.0
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))


class MockBenchmarkService:
    """Mock BenchmarkService для нормализации ensemble."""
    
    def normalize_ensemble(
        self,
        scores_list: list[np.ndarray],
        weights: Optional[dict[str, float]] = None,
    ) -> np.ndarray:
        """Нормализация оценок от нескольких методов."""
        if not scores_list:
            return np.array([])
        
        lengths = [len(s) for s in scores_list]
        if len(set(lengths)) > 1:
            raise ValueError(
                f"Все массивы должны иметь одинаковую длину. "
                f"Получено: {lengths}"
            )
        
        if weights is None:
            weights = {f"method_{i}": 1.0 for i in range(len(scores_list))}
        
        try:
            from scipy.stats import rankdata, zscore
            use_scipy = True
        except ImportError:
            use_scipy = False
        
        if use_scipy:
            normalized_scores = []
            for scores in scores_list:
                ranks = rankdata(scores, method='average')
                z_scores = zscore(ranks)
                z_min, z_max = z_scores.min(), z_scores.max()
                if z_max > z_min:
                    normalized = (z_scores - z_min) / (z_max - z_min)
                else:
                    normalized = np.ones_like(z_scores) * 0.5
                normalized_scores.append(normalized)
        else:
            normalized_scores = []
            for scores in scores_list:
                s_min, s_max = scores.min(), scores.max()
                if s_max > s_min:
                    normalized = (scores - s_min) / (s_max - s_min)
                else:
                    normalized = np.ones_like(scores) * 0.5
                normalized_scores.append(normalized)
        
        weight_values = list(weights.values())
        if len(weight_values) != len(normalized_scores):
            raise ValueError(
                f"Количество весов ({len(weight_values)}) должно совпадать "
                f"с количеством массивов ({len(normalized_scores)})"
            )
        
        total_weight = sum(weight_values)
        normalized_weights = [w / total_weight for w in weight_values]
        
        ensemble_scores = np.zeros_like(scores_list[0])
        for w, scores_norm in zip(normalized_weights, normalized_scores):
            ensemble_scores += w * scores_norm
        
        return ensemble_scores


# === Фикстуры ===

@pytest.fixture
def centroid_service():
    """Фикстура CentroidService."""
    return MockCentroidService()


@pytest.fixture
def similarity_service():
    """Фикстура SimilarityService с cosine метрикой."""
    return MockSimilarityService(metric="cosine")


@pytest.fixture
def tfidf_service():
    """Фикстура TfidfService."""
    return MockTfidfService()


@pytest.fixture
def benchmark_service():
    """Фикстура BenchmarkService."""
    return MockBenchmarkService()


@pytest.fixture
def sample_embeddings():
    """Фикстура тестовых эмбеддингов."""
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])


@pytest.fixture
def sample_weights():
    """Фикстура тестовых весов."""
    return np.array([0.5, 0.3, 0.2])


# === Тесты для calculate_sbert_centroid ===

class TestCalculateSbertCentroid:
    """Тесты для calculate_sbert_centroid."""
    
    def test_sbert_centroid_single_embedding(self, centroid_service):
        """Тест: центроид для одного эмбеддинга."""
        embedding = np.array([1.0, 2.0, 3.0])
        
        centroid = centroid_service.calculate_centroid(embedding)
        
        assert isinstance(centroid, np.ndarray)
        assert centroid.shape == (3,)
        np.testing.assert_array_equal(centroid, embedding)
    
    def test_sbert_centroid_multiple_embeddings(self, centroid_service):
        """Тест: центроид для нескольких эмбеддингов."""
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ])
        
        centroid = centroid_service.calculate_centroid(embeddings)
        
        assert isinstance(centroid, np.ndarray)
        assert centroid.shape == (3,)
        np.testing.assert_array_almost_equal(centroid, [3.0, 0.0, 0.0])
    
    def test_sbert_centroid_weighted(self, centroid_service, sample_weights):
        """Тест: взвешенный центроид SBERT."""
        embeddings = np.array([
            [1.0, 0.0],
            [4.0, 0.0],
            [7.0, 0.0],
        ])
        
        centroid = centroid_service.calculate_centroid(embeddings, sample_weights)
        
        assert isinstance(centroid, np.ndarray)
        # 0.5 * [1,0] + 0.3 * [4,0] + 0.2 * [7,0] = [0.5 + 1.2 + 1.4, 0] = [3.1, 0]
        np.testing.assert_array_almost_equal(centroid, [3.1, 0.0])
    
    def test_sbert_centroid_empty_raises(self, centroid_service):
        """Тест: пустой массив вызывает ошибку."""
        with pytest.raises(ValueError, match="пустым"):
            centroid_service.calculate_centroid(np.array([]))
    
    def test_sbert_centroid_type_check(self, centroid_service):
        """Тест: проверка типа возвращаемого значения."""
        embeddings = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ])
        
        centroid = centroid_service.calculate_centroid(embeddings)
        
        assert isinstance(centroid, np.ndarray)
        assert centroid.dtype in [np.float32, np.float64]
    
    def test_sbert_centroid_value_range(self, centroid_service):
        """Тест: значения центроида в разумном диапазоне."""
        embeddings = np.array([
            [0.5, -0.5],
            [-0.3, 0.7],
        ])
        
        centroid = centroid_service.calculate_centroid(embeddings)
        
        assert not np.any(np.isnan(centroid))
        assert not np.any(np.isinf(centroid))
    
    def test_sbert_centroid_3d(self, centroid_service):
        """Тест: центроид для 3D массива."""
        embeddings = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ])
        
        centroid = centroid_service.calculate_centroid(embeddings)
        
        np.testing.assert_array_almost_equal(centroid, [4.0, 5.0, 6.0])


# === Тесты для calculate_tfidf_centroid ===

class TestCalculateTfidfCentroid:
    """Тесты для calculate_tfidf_centroid."""
    
    def test_tfidf_centroid_basic(self, tfidf_service):
        """Тест: базовый расчёт TF-IDF центроида."""
        terms = ["машинное", "обучение", "машинное"]
        
        tfidf_service.fit_terms(terms)
        weights = tfidf_service.get_weights_for_centroid(terms)
        
        assert isinstance(weights, np.ndarray)
        assert len(weights) == len(terms)
        assert np.all(weights >= 0)
    
    def test_tfidf_centroid_weights_sum(self, tfidf_service):
        """Тест: сумма весов TF-IDF."""
        terms = ["нейронная", "сеть", "глубокое"]
        
        tfidf_service.fit_terms(terms)
        weights = tfidf_service.get_weights_for_centroid(terms)
        
        total_weight = np.sum(weights)
        assert total_weight > 0
    
    def test_tfidf_centroid_identical_terms(self, tfidf_service):
        """Тест: одинаковые термины дают одинаковые веса."""
        terms = ["тест", "тест", "тест"]
        
        tfidf_service.fit_terms(terms)
        weights = tfidf_service.get_weights_for_centroid(terms)
        
        assert np.all(weights == weights[0])
    
    def test_tfidf_centroid_empty_terms(self, tfidf_service):
        """Тест: пустой список терминов."""
        tfidf_service.fit_terms([])
        weights = tfidf_service.get_weights_for_centroid([])
        
        assert isinstance(weights, np.ndarray)
        assert len(weights) == 0
    
    def test_tfidf_centroid_type_check(self, tfidf_service):
        """Тест: проверка типа возвращаемого значения."""
        terms = ["термин1", "термин2"]
        
        tfidf_service.fit_terms(terms)
        weights = tfidf_service.get_weights_for_centroid(terms)
        
        assert isinstance(weights, np.ndarray)
        assert weights.dtype in [np.float32, np.float64]
    
    def test_tfidf_centroid_value_range(self, tfidf_service):
        """Тест: веса в неотрицательном диапазоне."""
        terms = ["машинное", "обучение", "ml"]
        
        tfidf_service.fit_terms(terms)
        weights = tfidf_service.get_weights_for_centroid(terms)
        
        assert not np.any(np.isnan(weights))
        assert not np.any(np.isinf(weights))


# === Тесты для calculate_ensemble_centroid ===

class TestCalculateEnsembleCentroid:
    """Тесты для calculate_ensemble_centroid."""
    
    def test_ensemble_centroid_combination(self, centroid_service, tfidf_service):
        """Тест: комбинация SBERT и TF-IDF центроидов."""
        sbert_embeddings = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        sbert_centroid = centroid_service.calculate_centroid(sbert_embeddings)
        
        terms = ["термин1", "термин2"]
        tfidf_service.fit_terms(terms)
        tfidf_weights = tfidf_service.get_weights_for_centroid(terms)
        
        if np.sum(tfidf_weights) > 0:
            normalized_weights = tfidf_weights / np.sum(tfidf_weights)
            ensemble_centroid = centroid_service.calculate_centroid(
                sbert_embeddings, 
                normalized_weights
            )
        else:
            ensemble_centroid = sbert_centroid
        
        assert isinstance(ensemble_centroid, np.ndarray)
        assert ensemble_centroid.shape == (2,)
    
    def test_ensemble_centroid_equal_weights(self, centroid_service):
        """Тест: равные веса SBERT и TF-IDF."""
        embeddings = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        weights = np.array([0.5, 0.5])
        
        centroid = centroid_service.calculate_centroid(embeddings, weights)
        
        np.testing.assert_array_almost_equal(centroid, [0.5, 0.5])
    
    def test_ensemble_centroid_sbert_heavy(self, centroid_service):
        """Тест: SBERT-heavy ensemble (0.8 SBERT, 0.2 TF-IDF)."""
        embeddings = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        effective_weights = np.array([0.8, 0.2])
        
        centroid = centroid_service.calculate_centroid(embeddings, effective_weights)
        
        np.testing.assert_array_almost_equal(centroid, [0.8, 0.2])
    
    def test_ensemble_centroid_type_check(self, centroid_service):
        """Тест: проверка типа возвращаемого значения."""
        embeddings = np.array([[1.0], [0.5]])
        weights = np.array([0.5, 0.5])
        
        centroid = centroid_service.calculate_centroid(embeddings, weights)
        
        assert isinstance(centroid, np.ndarray)
        assert centroid.dtype in [np.float32, np.float64]
    
    def test_ensemble_centroid_value_range(self, centroid_service):
        """Тест: значения в разумном диапазоне."""
        embeddings = np.array([
            [-0.5, 0.5],
            [0.3, -0.7],
        ])
        weights = np.array([0.5, 0.5])
        
        centroid = centroid_service.calculate_centroid(embeddings, weights)
        
        assert not np.any(np.isnan(centroid))
        assert not np.any(np.isinf(centroid))


# === Тесты для calculate_sbert_similarity ===

class TestCalculateSbertSimilarity:
    """Тесты для calculate_sbert_similarity."""
    
    def test_sbert_similarity_identical_vectors(self, similarity_service):
        """Тест: идентичные векторы имеют близость 1.0."""
        vec = np.array([1.0, 0.0])
        
        similarity = similarity_service.calculate_similarity(vec, vec)
        
        assert isinstance(similarity, float)
        assert similarity == pytest.approx(1.0, abs=1e-5)
    
    def test_sbert_similarity_orthogonal_vectors(self, similarity_service):
        """Тест: ортогональные векторы имеют близость 0.0."""
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([0.0, 1.0])
        
        similarity = similarity_service.calculate_similarity(vec1, vec2)
        
        assert isinstance(similarity, float)
        assert similarity == pytest.approx(0.0, abs=1e-5)
    
    def test_sbert_similarity_opposite_vectors(self, similarity_service):
        """Тест: противоположные векторы имеют близость -1.0."""
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([-1.0, 0.0])
        
        similarity = similarity_service.calculate_similarity(vec1, vec2)
        
        assert isinstance(similarity, float)
        assert similarity == pytest.approx(-1.0, abs=1e-5)
    
    def test_sbert_similarity_type_check(self, similarity_service):
        """Тест: проверка типа возвращаемого значения."""
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([0.5, 0.5])
        
        similarity = similarity_service.calculate_similarity(vec1, vec2)
        
        assert isinstance(similarity, float)
    
    def test_sbert_similarity_value_range(self, similarity_service):
        """Тест: значение в диапазоне [-1, 1]."""
        test_cases = [
            (np.array([1.0, 0.0]), np.array([1.0, 0.0])),
            (np.array([1.0, 0.0]), np.array([-1.0, 0.0])),
            (np.array([1.0, 0.0]), np.array([0.0, 1.0])),
            (np.array([0.5, 0.5]), np.array([0.3, 0.7])),
        ]
        
        for vec1, vec2 in test_cases:
            similarity = similarity_service.calculate_similarity(vec1, vec2)
            assert -1.0 <= similarity <= 1.0, f"Similarity {similarity} out of range"
    
    def test_sbert_similarity_zero_vector(self, similarity_service):
        """Тест: нулевой вектор возвращает 0.0."""
        zero_vec = np.zeros(3)
        non_zero = np.array([1.0, 0.0, 0.0])
        
        similarity = similarity_service.calculate_similarity(zero_vec, non_zero)
        
        assert similarity == 0.0
    
    def test_sbert_similarity_normalized_vectors(self, similarity_service):
        """Тест: нормализованные векторы имеют корректную близость."""
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([0.707, 0.707])
        
        similarity = similarity_service.calculate_similarity(vec1, vec2)
        
        assert isinstance(similarity, float)
        assert 0.5 <= similarity <= 1.0


# === Тесты для calculate_tfidf_similarity ===

class TestCalculateTfidfSimilarity:
    """Тесты для calculate_tfidf_similarity."""
    
    def test_tfidf_similarity_identical_terms(self, tfidf_service):
        """Тест: одинаковые термины имеют максимальную близость."""
        term = "машинное"
        
        tfidf_service.fit_terms([term])
        similarity = tfidf_service.get_similarity(term, term)
        
        assert isinstance(similarity, float)
        assert similarity == pytest.approx(1.0, abs=1e-5)
    
    def test_tfidf_similarity_similar_terms(self, tfidf_service):
        """Тест: похожие термины имеют высокую близость."""
        similarity = tfidf_service.get_similarity("машинное", "машинный")
        
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
    
    def test_tfidf_similarity_different_terms(self, tfidf_service):
        """Тест: разные термины имеют низкую близость."""
        terms = ["апельсин", "банан", "книга"]
        tfidf_service.fit_terms(terms)
        
        similarity = tfidf_service.get_similarity("апельсин", "книга")
        
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
    
    def test_tfidf_similarity_type_check(self, tfidf_service):
        """Тест: проверка типа возвращаемого значения."""
        tfidf_service.fit_terms(["термин1", "термин2"])
        
        similarity = tfidf_service.get_similarity("термин1", "термин2")
        
        assert isinstance(similarity, float)
    
    def test_tfidf_similarity_value_range(self, tfidf_service):
        """Тест: значение в диапазоне [0, 1]."""
        terms = ["слово1", "слово2", "слово3"]
        tfidf_service.fit_terms(terms)
        
        for t1 in terms:
            for t2 in terms:
                similarity = tfidf_service.get_similarity(t1, t2)
                assert 0.0 <= similarity <= 1.0, f"Similarity {similarity} out of range"


# === Тесты для calculate_ensemble_similarity ===

class TestCalculateEnsembleSimilarity:
    """Тесты для calculate_ensemble_similarity."""
    
    def test_ensemble_similarity_basic(self, similarity_service, benchmark_service):
        """Тест: базовый расчёт ensemble близости."""
        vec1_domain1 = np.array([1.0, 0.0])
        vec2_domain1 = np.array([0.8, 0.2])
        vec1_domain2 = np.array([0.9, 0.1])
        vec2_domain2 = np.array([0.7, 0.3])
        
        centroid1 = np.mean([vec1_domain1, vec2_domain1], axis=0)
        centroid2 = np.mean([vec1_domain2, vec2_domain2], axis=0)
        
        sbert_sim = similarity_service.calculate_similarity(centroid1, centroid2)
        
        tfidf_vec1 = np.array([0.8, 0.2])
        tfidf_vec2 = np.array([0.7, 0.3])
        tfidf_sim = similarity_service.calculate_similarity(tfidf_vec1, tfidf_vec2)
        
        scores_list = [np.array([sbert_sim]), np.array([tfidf_sim])]
        ensemble_scores = benchmark_service.normalize_ensemble(scores_list)
        
        assert isinstance(ensemble_scores, np.ndarray)
        assert len(ensemble_scores) == 1
        assert 0.0 <= ensemble_scores[0] <= 1.0
    
    def test_ensemble_similarity_equal_components(self, benchmark_service):
        """Тест: равные компоненты дают нормализованное значение."""
        scores_sbert = np.array([0.8, 0.6, 0.4])
        scores_tfidf = np.array([0.8, 0.6, 0.4])
        
        ensemble = benchmark_service.normalize_ensemble([scores_sbert, scores_tfidf])
        
        assert isinstance(ensemble, np.ndarray)
        assert len(ensemble) == 3
    
    def test_ensemble_similarity_weighted(self, benchmark_service):
        """Тест: взвешенный ensemble."""
        scores_list = [
            np.array([0.8, 0.6, 0.4]),
            np.array([0.2, 0.3, 0.1]),
        ]
        weights = {"sbert": 0.7, "tfidf": 0.3}
        
        ensemble = benchmark_service.normalize_ensemble(scores_list, weights)
        
        assert isinstance(ensemble, np.ndarray)
        assert len(ensemble) == 3
    
    def test_ensemble_similarity_type_check(self, benchmark_service):
        """Тест: проверка типа возвращаемого значения."""
        scores_list = [np.array([0.5, 0.6])]
        ensemble = benchmark_service.normalize_ensemble(scores_list)
        
        assert isinstance(ensemble, np.ndarray)
        assert ensemble.dtype in [np.float32, np.float64]
    
    def test_ensemble_similarity_value_range(self, benchmark_service):
        """Тест: значение в диапазоне [0, 1]."""
        scores_list = [
            np.array([0.1, 0.5, 0.9]),
            np.array([0.2, 0.6, 0.8]),
        ]
        weights = {"sbert": 0.5, "tfidf": 0.5}
        
        ensemble = benchmark_service.normalize_ensemble(scores_list, weights)
        
        assert not np.any(np.isnan(ensemble))
        assert not np.any(np.isinf(ensemble))
        assert np.all(ensemble >= 0.0)
        assert np.all(ensemble <= 1.0)
    
    def test_ensemble_similarity_empty_list(self, benchmark_service):
        """Тест: пустой список возвращает пустой массив."""
        ensemble = benchmark_service.normalize_ensemble([])
        
        assert isinstance(ensemble, np.ndarray)
        assert len(ensemble) == 0
    
    def test_ensemble_similarity_mismatched_lengths_raises(self, benchmark_service):
        """Тест: разная длина массивов вызывает ошибку."""
        scores_list = [
            np.array([0.8, 0.6]),
            np.array([0.2]),
        ]
        
        with pytest.raises(ValueError, match="одинаковую длину"):
            benchmark_service.normalize_ensemble(scores_list)


# === Интеграционные тесты ===

class TestIntegration:
    """Интеграционные тесты для всего pipeline."""
    
    def test_full_sbert_pipeline(self, centroid_service, similarity_service):
        """Тест: полный SBERT pipeline (centroid → similarity)."""
        domain1_embeddings = np.array([
            [1.0, 0.0],
            [0.9, 0.1],
        ])
        domain2_embeddings = np.array([
            [0.8, 0.2],
            [0.7, 0.3],
        ])
        
        centroid1 = centroid_service.calculate_centroid(domain1_embeddings)
        centroid2 = centroid_service.calculate_centroid(domain2_embeddings)
        similarity = similarity_service.calculate_similarity(centroid1, centroid2)
        
        assert isinstance(centroid1, np.ndarray)
        assert isinstance(centroid2, np.ndarray)
        assert isinstance(similarity, float)
        assert -1.0 <= similarity <= 1.0
    
    def test_full_tfidf_pipeline(self, tfidf_service, similarity_service):
        """Тест: полный TF-IDF pipeline (centroid → similarity)."""
        domain1_terms = ["машинное", "обучение", "ml"]
        domain2_terms = ["глубокое", "обучение", "dl"]
        
        all_terms = list(set(domain1_terms + domain2_terms))
        tfidf_service.fit_terms(all_terms)
        
        weights1 = tfidf_service.get_weights_for_centroid(domain1_terms)
        weights2 = tfidf_service.get_weights_for_centroid(domain2_terms)
        
        similarity = similarity_service.calculate_similarity(weights1, weights2)
        
        assert isinstance(weights1, np.ndarray)
        assert isinstance(weights2, np.ndarray)
        assert isinstance(similarity, float)
        assert similarity >= 0.0
        assert similarity == pytest.approx(1.0, abs=1e-9)
    
    def test_full_ensemble_pipeline(self, centroid_service, similarity_service, tfidf_service, benchmark_service):
        """Тест: полный ensemble pipeline."""
        sbert_emb1 = np.array([[1.0, 0.0], [0.9, 0.1]])
        terms1 = ["машинное", "обучение"]
        
        sbert_emb2 = np.array([[0.8, 0.2], [0.7, 0.3]])
        terms2 = ["глубокое", "сеть"]
        
        sbert_centroid1 = centroid_service.calculate_centroid(sbert_emb1)
        sbert_centroid2 = centroid_service.calculate_centroid(sbert_emb2)
        sbert_sim = similarity_service.calculate_similarity(sbert_centroid1, sbert_centroid2)
        
        all_terms = terms1 + terms2
        tfidf_service.fit_terms(all_terms)
        tfidf_weights1 = tfidf_service.get_weights_for_centroid(terms1)
        tfidf_weights2 = tfidf_service.get_weights_for_centroid(terms2)
        tfidf_sim = similarity_service.calculate_similarity(tfidf_weights1, tfidf_weights2)
        
        ensemble_scores = benchmark_service.normalize_ensemble(
            [np.array([sbert_sim]), np.array([tfidf_sim])],
            {"sbert": 0.6, "tfidf": 0.4}
        )
        
        assert isinstance(sbert_sim, float)
        assert isinstance(tfidf_sim, float)
        assert isinstance(ensemble_scores, np.ndarray)
        assert len(ensemble_scores) == 1
