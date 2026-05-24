# backend/src/application/benchmark_service.py
# Сервис бенчмаркинга для сравнения методов семантической близости
#
# Версия: 1.3
# Обновлено: 2026-04-09

"""
Модуль для бенчмаркинга методов семантической близости.

Сравнивает различные подходы:
- SBERT (baseline)
- SBERT + TF-IDF
- SBERT + Z-score
- RuWordNet (Lin)
- RuWordNet (Wu-Palmer)
- English WordNet (Lin)
- English WordNet (Wu-Palmer)
- Hybrid (SBERT + RuWordNet)

Метрики:
- Корреляция Спирмена
- Корреляция Пирсона
- MSE (Mean Squared Error)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from ..infrastructure.embedding_service import EmbeddingService
from ..infrastructure.wordnet_service import WordNetService
from ..infrastructure.en_wordnet_service import EnglishWordNetService
from ..infrastructure.tfidf_service import TfidfService


@dataclass
class BenchmarkPair:
    """Пара терминов с экспертной оценкой близости."""
    word1: str
    word2: str
    human_score: float


@dataclass
class MethodResult:
    """Результат оценки одного метода на бенчмарке."""
    method: str
    spearman: float
    pearson: float
    mse: float
    missing: int  # Количество пар, которые метод не смог обработать
    predictions_count: int


@dataclass
class BenchmarkComparison:
    """Сравнение всех методов на бенчмарке."""
    dataset_name: str
    dataset_size: int
    results: list[MethodResult] = field(default_factory=list)
    execution_time_sec: float = 0.0


class BenchmarkService:
    """Сервис для бенчмаркинга методов семантической близости.
    
    Сравнивает различные подходы к измерению семантической близости
    на стандартных бенчмарках с экспертными оценками.
    
    Атрибуты:
        embedding_service: Сервис SBERT эмбеддингов.
        wordnet_service: Сервис RuWordNet.
        tfidf_service: Сервис TF-IDF весов.
    """
    
    # Поддерживаемые форматы колонок с оценками
    SCORE_COLUMNS = ['sim', 'distance', 'distanceij', 'similarity']
    
    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        wordnet_service: Optional[WordNetService] = None,
        tfidf_service: Optional[TfidfService] = None,
    ) -> None:
        """Инициализация сервиса бенчмаркинга.
        
        Args:
            embedding_service: Сервис эмбеддингов (создаётся автоматически).
            wordnet_service: Сервис RuWordNet (создаётся автоматически).
            tfidf_service: Сервис TF-IDF (создаётся автоматически).
        """
        self.embedding_service = embedding_service or EmbeddingService()
        self.wordnet_service = wordnet_service
        self.tfidf_service = tfidf_service
        self.en_wordnet_service: Optional[EnglishWordNetService] = None
    
    def load_benchmark(self, path: str | Path) -> list[BenchmarkPair]:
        """Загрузка бенчмарка из CSV файла.
        
        Поддерживает форматы:
        - English SimLex-999: word1,word2,sim
        - Russian SimLex: word1,word2,distanceij
        
        Колонки с оценками:
        - distance: 0 = синонимы, 10 = антонимы → ИНВЕРТИРУЕТСЯ
        - distanceij: 8-10 = синонимы, 0-2 = антонимы → это similarity, НЕ инвертируется
        - sim/similarity: уже близость → НЕ инвертируется
        
        Args:
            path: Путь к CSV файлу.
        
        Returns:
            Список пар с экспертными оценками.
        """
        df = pd.read_csv(path)
        
        # Определяем колонку с оценкой
        score_col = None
        for col in self.SCORE_COLUMNS:
            if col in df.columns:
                score_col = col
                break
        
        if score_col is None:
            raise ValueError(f"Не найдена колонка с оценкой. Доступные: {df.columns.tolist()}")
        
        # Определяем, нужно ли инвертировать
        # distance: 0 = синонимы, 10 = антонимы → инвертируем
        # distanceij: 8-10 = синонимы, 0-2 = антонимы → это similarity, НЕ инвертируем
        # sim/similarity: уже близость
        invert = score_col == 'distance'
        max_dist = df[score_col].max() if invert else 10
        
        pairs = []
        for _, row in df.iterrows():
            score = float(row[score_col])
            # Инвертируем только для колонки 'distance'
            if invert:
                score = max_dist - score
            pairs.append(BenchmarkPair(
                word1=str(row['word1']),
                word2=str(row['word2']),
                human_score=score
            ))
        return pairs
    
    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Расчёт косинусного сходства между векторами."""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))
    
    def _calculate_metrics(
        self,
        predictions: list[float],
        ground_truth: list[float],
    ) -> tuple[float, float, float]:
        """Расчёт метрик качества."""
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        if len(predictions) < 2 or len(ground_truth) < 2:
            return 0.0, 0.0, 1.0
        
        if np.std(predictions) == 0 or np.std(ground_truth) == 0:
            return 0.0, 0.0, 1.0
        
        spearman_corr, _ = spearmanr(predictions, ground_truth)
        if np.isnan(spearman_corr):
            spearman_corr = 0.0
        
        pearson_corr, _ = pearsonr(predictions, ground_truth)
        if np.isnan(pearson_corr):
            pearson_corr = 0.0
        
        mse = float(np.mean((predictions - ground_truth) ** 2))
        
        return spearman_corr, pearson_corr, mse
    
    def _evaluate_method(
        self,
        method_name: str,
        pairs: list[BenchmarkPair],
        predict_fn,
    ) -> MethodResult:
        """Оценка одного метода на бенчмарке."""
        predictions: list[float] = []
        ground_truth: list[float] = []
        missing = 0
        
        for pair in pairs:
            try:
                sim = predict_fn(pair.word1, pair.word2)
                if sim is not None and not np.isnan(sim):
                    predictions.append(sim)
                    ground_truth.append(pair.human_score)
                else:
                    missing += 1
            except Exception:
                missing += 1
        
        if not predictions:
            return MethodResult(
                method=method_name,
                spearman=0.0,
                pearson=0.0,
                mse=1.0,
                missing=len(pairs),
                predictions_count=0,
            )
        
        spearman, pearson, mse = self._calculate_metrics(predictions, ground_truth)
        
        return MethodResult(
            method=method_name,
            spearman=spearman,
            pearson=pearson,
            mse=mse,
            missing=missing,
            predictions_count=len(predictions),
        )
    
    def evaluate_sbert(self, pairs: list[BenchmarkPair]) -> MethodResult:
        """Оценка SBERT (baseline)."""
        def predict(word1: str, word2: str) -> float:
            emb1 = self.embedding_service.get_embedding(word1)
            emb2 = self.embedding_service.get_embedding(word2)
            return (self._cosine_similarity(emb1, emb2) + 1) / 2
        
        return self._evaluate_method("SBERT (baseline)", pairs, predict)
    
    def evaluate_sbert_tfidf(self, pairs: list[BenchmarkPair]) -> MethodResult:
        """Оценка SBERT + TF-IDF.
        
        Использует IDF веса для модуляции косинусного сходства.
        IDF нормализуется в диапазоне [0, 1] перед использованием.
        """
        all_terms = list({p.word1 for p in pairs} | {p.word2 for p in pairs})
        
        if self.tfidf_service is None:
            self.tfidf_service = TfidfService()
        self.tfidf_service.fit(all_terms)
        
        # Вычисляем min/max IDF для нормализации
        idf_values = [self.tfidf_service.get_idf(t) for t in all_terms]
        min_idf = min(idf_values)
        max_idf = max(idf_values)
        idf_range = max_idf - min_idf if max_idf > min_idf else 1.0
        
        def predict(word1: str, word2: str) -> float:
            emb1 = self.embedding_service.get_embedding(word1)
            emb2 = self.embedding_service.get_embedding(word2)
            
            # Получаем IDF веса
            idf1 = self.tfidf_service.get_idf(word1)
            idf2 = self.tfidf_service.get_idf(word2)
            
            # Нормализуем IDF в [0, 1]
            norm_idf1 = (idf1 - min_idf) / idf_range
            norm_idf2 = (idf2 - min_idf) / idf_range
            
            # Косинусное сходство
            cos_sim = self._cosine_similarity(emb1, emb2)
            
            # Комбинированная мера: взвешенное косинусное сходство
            # Используем среднее нормализованных IDF как вес
            avg_idf_weight = (norm_idf1 + norm_idf2) / 2
            
            # Итоговая формула: cos_sim * (0.5 + 0.5 * avg_idf_weight)
            # Это даёт диапазон [0.5, 1.0] для сходства
            sim = cos_sim * (0.5 + 0.5 * avg_idf_weight)
            
            return float(np.clip(sim, 0, 1))
        
        return self._evaluate_method("SBERT + TF-IDF", pairs, predict)
    
    def evaluate_sbert_zscore(self, pairs: list[BenchmarkPair]) -> MethodResult:
        """Оценка SBERT + Z-score нормализация."""
        def predict(word1: str, word2: str) -> float:
            emb1 = self.embedding_service.get_embedding(word1)
            emb2 = self.embedding_service.get_embedding(word2)
            raw_sim = self._cosine_similarity(emb1, emb2)
            normalized = (raw_sim + 1) / 2
            mean_val = 0.5
            std_val = 0.15
            z = (normalized - mean_val) / std_val
            return float(1 / (1 + np.exp(-z)))
        
        return self._evaluate_method("SBERT + Z-score", pairs, predict)
    
    def evaluate_ruwordnet_lin(self, pairs: list[BenchmarkPair]) -> MethodResult:
        """Оценка RuWordNet (Lin similarity)."""
        if self.wordnet_service is None:
            self.wordnet_service = WordNetService()
            try:
                self.wordnet_service.initialize()
            except Exception:
                return MethodResult(
                    method="RuWordNet (Lin)",
                    spearman=0.0,
                    pearson=0.0,
                    mse=1.0,
                    missing=len(pairs),
                    predictions_count=0,
                )
        
        def predict(word1: str, word2: str) -> Optional[float]:
            result = self.wordnet_service.get_similarity(word1, word2, "lin")
            return result.similarity
        
        return self._evaluate_method("RuWordNet (Lin)", pairs, predict)
    
    def evaluate_ruwordnet_wup(self, pairs: list[BenchmarkPair]) -> MethodResult:
        """Оценка RuWordNet (Wu-Palmer similarity)."""
        if self.wordnet_service is None:
            self.wordnet_service = WordNetService()
            try:
                self.wordnet_service.initialize()
            except Exception:
                return MethodResult(
                    method="RuWordNet (Wu-Palmer)",
                    spearman=0.0,
                    pearson=0.0,
                    mse=1.0,
                    missing=len(pairs),
                    predictions_count=0,
                )
        
        def predict(word1: str, word2: str) -> Optional[float]:
            result = self.wordnet_service.get_similarity(word1, word2, "wup")
            return result.similarity
        
        return self._evaluate_method("RuWordNet (Wu-Palmer)", pairs, predict)
    
    def evaluate_en_wordnet_lin(self, pairs: list[BenchmarkPair]) -> MethodResult:
        """Оценка English WordNet (Lin similarity)."""
        if self.en_wordnet_service is None:
            self.en_wordnet_service = EnglishWordNetService()
            try:
                self.en_wordnet_service.initialize()
            except Exception:
                return MethodResult(
                    method="English WordNet (Lin)",
                    spearman=0.0,
                    pearson=0.0,
                    mse=1.0,
                    missing=len(pairs),
                    predictions_count=0,
                )
        
        def predict(word1: str, word2: str) -> Optional[float]:
            result = self.en_wordnet_service.get_similarity(word1, word2, "lin")
            return result.similarity
        
        return self._evaluate_method("English WordNet (Lin)", pairs, predict)
    
    def evaluate_en_wordnet_wup(self, pairs: list[BenchmarkPair]) -> MethodResult:
        """Оценка English WordNet (Wu-Palmer similarity)."""
        if self.en_wordnet_service is None:
            self.en_wordnet_service = EnglishWordNetService()
            try:
                self.en_wordnet_service.initialize()
            except Exception:
                return MethodResult(
                    method="English WordNet (Wu-Palmer)",
                    spearman=0.0,
                    pearson=0.0,
                    mse=1.0,
                    missing=len(pairs),
                    predictions_count=0,
                )
        
        def predict(word1: str, word2: str) -> Optional[float]:
            result = self.en_wordnet_service.get_similarity(word1, word2, "wup")
            return result.similarity
        
        return self._evaluate_method("English WordNet (Wu-Palmer)", pairs, predict)
    
    def evaluate_hybrid(self, pairs: list[BenchmarkPair]) -> MethodResult:
        """Оценка гибридного метода (SBERT + RuWordNet)."""
        if self.wordnet_service is None:
            self.wordnet_service = WordNetService()
            try:
                self.wordnet_service.initialize()
            except Exception:
                self.wordnet_service = None
        
        def predict(word1: str, word2: str) -> float:
            emb1 = self.embedding_service.get_embedding(word1)
            emb2 = self.embedding_service.get_embedding(word2)
            sbert_sim = (self._cosine_similarity(emb1, emb2) + 1) / 2
            
            if self.wordnet_service is not None:
                try:
                    wn_result = self.wordnet_service.get_similarity(word1, word2, "lin")
                    wn_sim = wn_result.similarity
                    if wn_sim > 0:
                        return (sbert_sim + wn_sim) / 2
                except Exception:
                    pass
            
            return sbert_sim
        
        return self._evaluate_method("Hybrid (SBERT + RuWordNet)", pairs, predict)
    
    def evaluate_bertopic(self, pairs: list[BenchmarkPair]) -> MethodResult:
        """Оценка BERTopic для кластеризации терминов.
        
        Использует BERTopic для кластеризации эмбеддингов терминов.
        Близость определяется по принадлежности к одному кластеру
        и расстоянию между центроидами кластеров.
        """
        try:
            from bertopic import BERTopic
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import normalize
        except (ImportError, Exception):
            # Если BERTopic не установлен — возвращаем заглушку
            return MethodResult(
                method="BERTopic",
                spearman=0.0,
                pearson=0.0,
                mse=1.0,
                missing=len(pairs),
                predictions_count=0,
            )
        
        # Собираем уникальные термины
        all_terms = list({p.word1 for p in pairs} | {p.word2 for p in pairs})
        
        if len(all_terms) < 5:
            return MethodResult(
                method="BERTopic",
                spearman=0.0,
                pearson=0.0,
                mse=1.0,
                missing=len(pairs),
                predictions_count=0,
            )
        
        # Получаем эмбеддинги
        embeddings = []
        for term in all_terms:
            emb = self.embedding_service.get_embedding(term)
            embeddings.append(emb)
        
        embeddings_array = np.array(embeddings)
        embeddings_normalized = normalize(embeddings_array)
        
        # Обучаем BERTopic с KMeans кластеризацией
        nr_clusters = min(10, max(3, len(all_terms) // 5))
        
        try:
            topic_model = BERTopic(
                embedding_model=None,  # Эмбеддинги уже готовы
                nr_topics=nr_clusters,
                hdbscan_model=KMeans(n_clusters=nr_clusters, random_state=42, n_init=10),
                verbose=False,
            )
            
            # BERTopic ожидает список строк или эмбеддинги в формате numpy array
            # Передаём пустую строку как placeholder, эмбеддинги уже готовы
            embeddings_list = embeddings_normalized.tolist()
            topic_model.embeddings = np.array(embeddings_list)
            topics, probs = topic_model.fit_transform(all_terms, y=embeddings_normalized)
        except Exception:
            # Если BERTopic несовместим с текущей версией sentence-transformers
            return MethodResult(
                method="BERTopic",
                spearman=0.0,
                pearson=0.0,
                mse=1.0,
                missing=len(pairs),
                predictions_count=0,
            )
        
        # Вычисляем центроиды кластеров
        term_to_idx = {term: i for i, term in enumerate(all_terms)}
        cluster_centroids: dict[int, np.ndarray] = {}
        
        for cluster_id in set(topics):
            if cluster_id == -1:
                continue
            cluster_indices = [i for i, t in enumerate(topics) if t == cluster_id]
            if cluster_indices:
                cluster_embs = embeddings_normalized[cluster_indices]
                cluster_centroids[cluster_id] = cluster_embs.mean(axis=0)
        
        def predict(word1: str, word2: str) -> Optional[float]:
            idx1 = term_to_idx.get(word1)
            idx2 = term_to_idx.get(word2)
            
            if idx1 is None or idx2 is None:
                return None
            
            topic1 = topics[idx1]
            topic2 = topics[idx2]
            
            # Если один из терминов не в кластере (-1) — возвращаем None
            if topic1 == -1 or topic2 == -1:
                return None
            
            # Если в одном кластере — максимальная близость
            if topic1 == topic2:
                return 0.9
            
            # Разные кластеры — считаем расстояние между центроидами
            if topic1 in cluster_centroids and topic2 in cluster_centroids:
                centroid1 = cluster_centroids[topic1]
                centroid2 = cluster_centroids[topic2]
                cos_sim = self._cosine_similarity(centroid1, centroid2)
                return float((cos_sim + 1) / 2)
            
            return None
        
        return self._evaluate_method("BERTopic", pairs, predict)
    
    def _load_terms_for_doc2vec(self, is_russian: bool = True) -> list[str]:
        """Загрузка терминов из terms.csv для расширения корпуса Doc2Vec.
        
        Args:
            is_russian: Если True, загружает русские термины, иначе английские.
        
        Returns:
            Список терминов для обучения Doc2Vec.
        """
        terms = []
        
        # Загружаем основной файл terms.csv
        terms_path = Path(__file__).parent.parent.parent.parent / "data" / "terms.csv"
        if terms_path.exists():
            try:
                df = pd.read_csv(terms_path)
                if 'term' in df.columns:
                    terms.extend(df['term'].dropna().astype(str).tolist())
            except Exception:
                pass  # Игнорируем ошибки чтения
        
        # Для английских датасетов загружаем английские термины
        if not is_russian:
            terms_en_path = Path(__file__).parent.parent.parent.parent / "data" / "terms_eng.csv"
            if terms_en_path.exists():
                try:
                    df_en = pd.read_csv(terms_en_path)
                    if 'term' in df_en.columns:
                        terms.extend(df_en['term'].dropna().astype(str).tolist())
                except Exception:
                    pass
        
        # Загружаем термины из wiki_ru и wiki_en для дополнительного контекста
        base_path = Path(__file__).parent.parent.parent.parent / "data"
        wiki_path = base_path / "wiki_ru" / "all_domains.csv" if is_russian else base_path / "wiki_en" / "all_domains.csv"
        if wiki_path.exists():
            try:
                df_wiki = pd.read_csv(wiki_path)
                if 'term' in df_wiki.columns:
                    terms.extend(df_wiki['term'].dropna().astype(str).tolist())
                elif len(df_wiki.columns) > 0:
                    # Первый столбец может быть терминами
                    terms.extend(df_wiki.iloc[:, 0].dropna().astype(str).tolist())
            except Exception:
                pass
        
        # Убираем дубликаты и пустые строки
        terms = list(set([t.strip() for t in terms if t.strip() and len(t.strip()) > 1]))
        return terms
    
    def evaluate_doc2vec(self, pairs: list[BenchmarkPair]) -> MethodResult:
        """Оценка Doc2Vec для эмбеддингов документов.
        
        Обучает Doc2Vec на расширенном корпусе терминов (terms.csv + wiki),
        затем использует косинусное сходство между векторами документов.
        
        Исправлено (2026-04-23):
        - Расширен корпус обучения через terms.csv
        - Исправлена токенизация: word-level вместо char-level
        - Улучшен fallback для infer_vector
        """
        try:
            from gensim.models.doc2vec import Doc2Vec, TaggedDocument
        except ImportError:
            return MethodResult(
                method="Doc2Vec",
                spearman=0.0,
                pearson=0.0,
                mse=1.0,
                missing=len(pairs),
                predictions_count=0,
            )
        
        # Собираем уникальные термины из пар
        pair_terms = list({p.word1 for p in pairs} | {p.word2 for p in pairs})
        
        if len(pair_terms) < 3:
            return MethodResult(
                method="Doc2Vec",
                spearman=0.0,
                pearson=0.0,
                mse=1.0,
                missing=len(pairs),
                predictions_count=0,
            )
        
        # Определяем язык по первой паре (русский если есть кириллица)
        is_russian = any('\u0430' <= c <= '\u044f' or '\u0410' <= c <= '\u042f' for c in ' '.join(pair_terms[:5]))
        
        # Загружаем дополнительные термины для расширения корпуса
        extra_terms = self._load_terms_for_doc2vec(is_russian)
        
        # Объединяем термины пар с дополнительными
        all_terms = pair_terms.copy()
        
        # Добавляем дополнительные термины (до 5000 для баланса скорости и качества)
        for term in extra_terms:
            if len(all_terms) >= 5000:
                break
            if term not in all_terms:
                all_terms.append(term)
        
        # Создаём корпус из терминов (каждый термин = документ)
        # Используем word-level токенизацию вместо char-level
        documents = []
        for term in all_terms:
            # Word-level: разбиваем на слова (разделители: пробелы, дефисы, точки)
            words = term.lower().replace('-', ' ').replace('.', ' ').split()
            # Фильтруем пустые токены
            words = [w for w in words if w and len(w) > 0]
            # Если термин однословный без разделителей - используем его как есть
            if len(words) == 0:
                words = [term.lower()]
            documents.append(TaggedDocument(words=words, tags=[term]))
        
        # Обучаем Doc2Vec
        # Параметры оптимизированы для небольших корпусов
        model = Doc2Vec(
            documents=documents,
            vector_size=100,
            window=min(5, len(documents)),
            min_count=1,
            workers=1,
            epochs=200,  # Увеличиваем эпохи для лучшей сходимости
            dm=1,  # PV-DM (Paragraph Vector - Distributed Memory)
            seed=42,
        )
        
        # Строим индекс обученных слов для fallback
        trained_words = set(model.dv.index_to_key)
        
        def predict(word1: str, word2: str) -> Optional[float]:
            try:
                # Токенизация word-level
                words1 = word1.lower().replace('-', ' ').replace('.', ' ').split()
                words2 = word2.lower().replace('-', ' ').replace('.', ' ').split()
                words1 = [w for w in words1 if w]
                words2 = [w for w in words2 if w]
                
                if not words1 or not words2:
                    return None
                
                # infer_vector с достаточным количеством итераций
                vec1 = model.infer_vector(words1, epochs=50, min_alpha=0.001)
                vec2 = model.infer_vector(words2, epochs=50, min_alpha=0.001)
                
                # Проверяем качество векторов
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                
                # Если один из векторов нулевой - используем most_similar как fallback
                if norm1 < 1e-6 or norm2 < 1e-6:
                    # Находим наиболее похожие обученные слова
                    if norm1 < 1e-6:
                        # Ищем ближайшее слово из пары в обученном словаре
                        word_candidates = [w for w in words1 if w in trained_words]
                        if word_candidates:
                            # Находим самое часто встречающееся
                            main_word = word_candidates[0]
                            similar = model.dv.most_similar(main_word, topn=5)
                            # Усредняем top-5 ближайших векторов
                            vec1 = np.mean([model.dv[word] for word, _ in similar], axis=0)
                        else:
                            return None
                    if norm2 < 1e-6:
                        word_candidates = [w for w in words2 if w in trained_words]
                        if word_candidates:
                            main_word = word_candidates[0]
                            similar = model.dv.most_similar(main_word, topn=5)
                            vec2 = np.mean([model.dv[word] for word, _ in similar], axis=0)
                        else:
                            return None
                
                cos_sim = self._cosine_similarity(vec1, vec2)
                # Нормализация в [0, 1]: (cos_sim + 1) / 2
                return float((cos_sim + 1) / 2)
            except Exception:
                return None
        
        return self._evaluate_method("Doc2Vec", pairs, predict)
    
    def evaluate_lda(self, pairs: list[BenchmarkPair]) -> MethodResult:
        """Оценка LDA (Latent Dirichlet Allocation) для тематического моделирования.
        
        Обучает LDA на корпусе, затем использует косинусное сходство
        между распределениями тем для каждой пары терминов.
        """
        try:
            from gensim.corpora import Dictionary
            from gensim.models import LdaModel
            import numpy as np
        except ImportError:
            return MethodResult(
                method="LDA",
                spearman=0.0,
                pearson=0.0,
                mse=1.0,
                missing=len(pairs),
                predictions_count=0,
            )
        
        # Собираем уникальные термины
        all_terms = list({p.word1 for p in pairs} | {p.word2 for p in pairs})
        
        if len(all_terms) < 3:
            return MethodResult(
                method="LDA",
                spearman=0.0,
                pearson=0.0,
                mse=1.0,
                missing=len(pairs),
                predictions_count=0,
            )
        
        # Создаём корпус из терминов
        texts = []
        for term in all_terms:
            # Токенизируем термин
            words = term.lower().split()
            if len(words) == 1:
                words = list(term.lower())
            texts.append(words)
        
        # Создаём словарь и корпус
        dictionary = Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        
        # Обучаем LDA
        nr_topics = min(10, max(2, len(all_terms) // 3))
        
        model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=nr_topics,
            random_state=42,
            passes=10,
            alpha='auto',
            per_word_topics=False,
        )
        
        # Получаем распределения тем для каждого термина
        term_topics: dict[str, np.ndarray] = {}
        for i, term in enumerate(all_terms):
            bow = corpus[i]
            topic_dist = model.get_document_topics(bow, minimum_probability=0.0)
            # Создаём вектор распределения тем
            topic_vector = np.zeros(nr_topics)
            for topic_id, prob in topic_dist:
                topic_vector[topic_id] = prob
            term_topics[term] = topic_vector
        
        def predict(word1: str, word2: str) -> Optional[float]:
            vec1 = term_topics.get(word1)
            vec2 = term_topics.get(word2)
            
            if vec1 is None or vec2 is None:
                return None
            
            # Проверяем, что есть нетривиальное распределение
            if np.sum(vec1) < 0.1 or np.sum(vec2) < 0.1:
                return None
            
            # Косинусное сходство между распределениями тем
            cos_sim = self._cosine_similarity(vec1, vec2)
            return float((cos_sim + 1) / 2)
        
        return self._evaluate_method("LDA", pairs, predict)
    
    def normalize_ensemble(
        self,
        scores_list: list[np.ndarray],
        weights: Optional[dict[str, float]] = None,
    ) -> np.ndarray:
        """Нормализация оценок от нескольких методов через rankdata → zscore → weighted mean.
        
        Используется для создания ансамблевой разметки без привлечения экспертов.
        
        Args:
            scores_list: Список массивов scores от разных методов.
                        Каждый массив содержит предсказания для N пар терминов.
            weights: Опциональные веса для каждого метода.
                    По умолчанию: SBERT=0.4, TF-IDF=0.3, WordNet=0.3
        
        Returns:
            Ансамблевый вектор оценок (N, )
        
        Пример использования:
            >>> scores_sbert = np.array([0.8, 0.6, 0.4])
            >>> scores_tfidf = np.array([0.2, 0.3, 0.1])
            >>> scores_wordnet = np.array([0.7, 0.5, 0.3])
            >>> ensemble = service.normalize_ensemble(
            ...     [scores_sbert, scores_tfidf, scores_wordnet]
            ... )
        """
        if not scores_list:
            return np.array([])
        
        # Проверка на одинаковую длину
        lengths = [len(s) for s in scores_list]
        if len(set(lengths)) > 1:
            raise ValueError(
                f"Все массивы должны иметь одинаковую длину. "
                f"Получено: {lengths}"
            )
        
        # Веса по умолчанию: равные веса для каждого массива
        if weights is None:
            weights = {f"method_{i}": 1.0 for i in range(len(scores_list))}
        
        # Нормализуем каждый массив: rank → z-score → [0, 1]
        from scipy.stats import rankdata, zscore
        
        normalized_scores = []
        for scores in scores_list:
            # Ранги от 1 до N
            ranks = rankdata(scores, method='average')
            # Z-score нормализация
            z_scores = zscore(ranks)
            # Min-max scaling к [0, 1]
            z_min, z_max = z_scores.min(), z_scores.max()
            if z_max > z_min:
                normalized = (z_scores - z_min) / (z_max - z_min)
            else:
                normalized = np.ones_like(z_scores) * 0.5
            normalized_scores.append(normalized)
        
        # Взвешенное усреднение
        weight_values = list(weights.values())
        if len(weight_values) != len(normalized_scores):
            raise ValueError(
                f"Количество весов ({len(weight_values)}) должно совпадать "
                f"с количеством массивов ({len(normalized_scores)})"
            )
        
        # Нормализация весов (сумма = 1)
        total_weight = sum(weight_values)
        normalized_weights = [w / total_weight for w in weight_values]
        
        ensemble_scores = np.zeros_like(scores_list[0])
        for w, scores_norm in zip(normalized_weights, normalized_scores):
            ensemble_scores += w * scores_norm
        
        return ensemble_scores
    
    def run_all(
        self,
        dataset_path: str | Path,
        dataset_name: Optional[str] = None,
    ) -> BenchmarkComparison:
        """Запуск экспериментов на бенчмарке.
        
        Автоматически определяет язык датасета и запускает только
        релевантные методы:
        - Для русских (hj-rg, simlex999_rus): SBERT + RuWordNet + Hybrid
        - Для английских (simlex999): SBERT + English WordNet
        """
        import time
        start_time = time.time()
        
        pairs = self.load_benchmark(dataset_path)
        name = dataset_name or Path(dataset_path).stem
        
        # Определяем язык датасета
        russian_datasets = ['hj-rg', 'simlex999_rus']
        english_datasets = ['simlex999']
        is_russian = name in russian_datasets
        is_english = name in english_datasets
        
        if is_russian:
            # Русские методы: SBERT + TF-IDF + Z-score + RuWordNet + Hybrid + BERTopic + Doc2Vec + LDA
            results = [
                self.evaluate_sbert(pairs),
                self.evaluate_sbert_tfidf(pairs),
                self.evaluate_sbert_zscore(pairs),
                self.evaluate_ruwordnet_lin(pairs),
                self.evaluate_ruwordnet_wup(pairs),
                self.evaluate_hybrid(pairs),
                self.evaluate_bertopic(pairs),
                self.evaluate_doc2vec(pairs),
                self.evaluate_lda(pairs),
            ]
        elif is_english:
            # Английские методы: SBERT + English WordNet + BERTopic + Doc2Vec + LDA
            results = [
                self.evaluate_sbert(pairs),
                self.evaluate_en_wordnet_lin(pairs),
                self.evaluate_en_wordnet_wup(pairs),
                self.evaluate_bertopic(pairs),
                self.evaluate_doc2vec(pairs),
                self.evaluate_lda(pairs),
            ]
        else:
            # Неизвестный датасет - запускаем SBERT как baseline
            results = [
                self.evaluate_sbert(pairs),
            ]
        
        execution_time = time.time() - start_time
        
        return BenchmarkComparison(
            dataset_name=name,
            dataset_size=len(pairs),
            results=results,
            execution_time_sec=execution_time,
        )
    
    def run_multiple_datasets(
        self,
        dataset_paths: dict[str, str | Path],
    ) -> dict[str, BenchmarkComparison]:
        """Запуск экспериментов на нескольких датасетах."""
        return {
            name: self.run_all(path, name)
            for name, path in dataset_paths.items()
        }


def format_results_table(comparison: BenchmarkComparison) -> str:
    """Форматирование результатов в виде таблицы."""
    lines = [
        f"## Результаты: {comparison.dataset_name}",
        f"(n={comparison.dataset_size}, время={comparison.execution_time_sec:.1f}s)",
        "",
        "| Метод | Спирмен | Пирсон | MSE | Missing |",
        "|-------|---------|--------|-----|--------|",
    ]
    
    for r in comparison.results:
        lines.append(
            f"| {r.method} | {r.spearman:.4f} | {r.pearson:.4f} | "
            f"{r.mse:.4f} | {r.missing} |"
        )
    
    return "\n".join(lines)
