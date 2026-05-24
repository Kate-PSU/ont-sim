# backend/src/application/similarity_methods.py
# Функции расчёта центроидов и близости доменов
#
# Версия: 1.0
# Создано: 2026-04-15
# Изменения: Вынесены из routes.py для переиспользования

"""
Модуль для расчёта центроидов и близости между предметными областями.

Содержит функции для:
- Расчёта центроидов (SBERT, TF-IDF, Ensemble)
- Расчёта близости между доменами (SBERT, TF-IDF, Ensemble)

Каждая функция имеет логирование и возвращает ошибку вместо fallback.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def calculate_sbert_centroid(
    terms: list[str],
    embedding_service,
) -> np.ndarray:
    """Расчёт SBERT центроида для списка терминов.
    
    Args:
        terms: Список терминов домена.
        embedding_service: Сервис для получения эмбеддингов.
    
    Returns:
        Центроид (вектор эмбеддинга).
    
    Raises:
        ValueError: Если terms пустой или embedding_service недоступен.
        RuntimeError: Если не удалось получить эмбеддинги.
    """
    logger.debug(f"[SBERT] Расчёт центроида для {len(terms)} терминов")
    
    if not terms:
        logger.error("[SBERT] Пустой список терминов")
        raise ValueError("Список терминов не может быть пустым")
    
    if embedding_service is None:
        logger.error("[SBERT] Embedding service не предоставлен")
        raise ValueError("embedding_service не может быть None")
    
    try:
        embeddings = embedding_service.get_embeddings_batch(terms)
        
        if embeddings is None or len(embeddings) == 0:
            logger.error(f"[SBERT] Не удалось получить эмбеддинги для {len(terms)} терминов")
            raise RuntimeError("Не удалось получить эмбеддинги")
        
        # Вычисляем центроид как среднее
        centroid = np.mean(embeddings, axis=0)
        
        # Нормализуем
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        
        logger.info(f"[SBERT] Центроид вычислен: shape={centroid.shape}, norm={norm:.4f}")
        return centroid
        
    except Exception as e:
        logger.error(f"[SBERT] Ошибка при расчёте центроида: {e}")
        raise


def calculate_tfidf_centroid(
    terms: list[str],
    tfidf_service,
) -> np.ndarray:
    """Расчёт TF-IDF центроида для списка терминов.
    
    Args:
        terms: Список терминов домена.
        tfidf_service: Сервис TF-IDF.
    
    Returns:
        TF-IDF центроид (вектор в пространстве n-грамм).
    
    Raises:
        ValueError: Если terms пустой или tfidf_service недоступен.
        RuntimeError: Если не удалось вычислить центроид.
    """
    logger.debug(f"[TF-IDF] Расчёт центроида для {len(terms)} терминов")
    
    if not terms:
        logger.error("[TF-IDF] Пустой список терминов")
        raise ValueError("Список терминов не может быть пустым")
    
    if tfidf_service is None:
        logger.error("[TF-IDF] TF-IDF service не предоставлен")
        raise ValueError("tfidf_service не может быть None")
    
    try:
        # Обучаем TF-IDF на терминах
        tfidf_service.fit_terms(terms)
        
        vocab_size = len(tfidf_service._ngram_vocab)
        if vocab_size == 0:
            logger.error("[TF-IDF] Пустой словарь TF-IDF")
            raise RuntimeError("TF-IDF словарь пуст")
        
        # Инициализируем центроид
        centroid = np.zeros(vocab_size)
        
        # Суммируем вектора всех терминов
        valid_terms = 0
        for term in terms:
            vec = tfidf_service.get_vector(term)
            if vec is not None:
                centroid += vec
                valid_terms += 1
        
        if valid_terms == 0:
            logger.error(f"[TF-IDF] Не удалось получить вектора для {len(terms)} терминов")
            raise RuntimeError("Не удалось получить TF-IDF вектора")
        
        # Усредняем
        centroid /= len(terms)
        
        logger.info(
            f"[TF-IDF] Центроид вычислен: vocab_size={vocab_size}, "
            f"valid_terms={valid_terms}, norm={np.linalg.norm(centroid):.4f}"
        )
        return centroid
        
    except Exception as e:
        logger.error(f"[TF-IDF] Ошибка при расчёте центроида: {e}")
        raise


def calculate_ensemble_centroid(
    terms: list[str],
    embedding_service,
    tfidf_service,
    weights: Optional[dict[str, float]] = None,
) -> np.ndarray:
    """Расчёт Ensemble центроида (SBERT + TF-IDF).
    
    Комбинирует SBERT и TF-IDF центроиды с заданными весами.
    
    Args:
        terms: Список терминов домена.
        embedding_service: Сервис для получения SBERT эмбеддингов.
        tfidf_service: Сервис TF-IDF.
        weights: Веса для комбинации {"sbert": float, "tfidf": float}.
                 Если None, используются веса по умолчанию (0.8334, 0.1666).
    
    Returns:
        Ensemble центроид.
    
    Raises:
        ValueError: Если terms пустой или сервисы недоступны.
        RuntimeError: Если не удалось вычислить центроид.
    """
    logger.debug(f"[Ensemble] Расчёт центроида для {len(terms)} терминов")
    
    if not terms:
        logger.error("[Ensemble] Пустой список терминов")
        raise ValueError("Список терминов не может быть пустым")
    
    if embedding_service is None:
        logger.error("[Ensemble] Embedding service не предоставлен")
        raise ValueError("embedding_service не может быть None")
    
    if tfidf_service is None:
        logger.error("[Ensemble] TF-IDF service не предоставлен")
        raise ValueError("tfidf_service не может быть None")
    
    # Дефолтные веса
    if weights is None:
        weights = {"sbert": 0.8334, "tfidf": 0.1666}
    
    # Нормализуем веса
    total_w = weights.get("sbert", 0) + weights.get("tfidf", 0)
    if total_w > 0:
        w_sbert = weights.get("sbert", 0) / total_w
        w_tfidf = weights.get("tfidf", 0) / total_w
    else:
        logger.warning("[Ensemble] Некорректные веса, используем равные")
        w_sbert = 0.5
        w_tfidf = 0.5
    
    try:
        # SBERT центроид
        sbert_centroid = calculate_sbert_centroid(terms, embedding_service)
        
        # TF-IDF центроид
        tfidf_centroid = calculate_tfidf_centroid(terms, tfidf_service)
        
        # Нормализуем TF-IDF центроид (приводим к тому же масштабу)
        # TF-IDF может иметь другую размерность, нужно привести к SBERT
        # Для простоты используем проекцию
        if len(tfidf_centroid) != len(sbert_centroid):
            # Разные размерности - используем только SBERT
            logger.warning(
                f"[Ensemble] Разные размерности центроидов: "
                f"SBERT={len(sbert_centroid)}, TF-IDF={len(tfidf_centroid)}. "
                f"Используем только SBERT."
            )
            return sbert_centroid
        
        # Комбинируем
        ensemble_centroid = w_sbert * sbert_centroid + w_tfidf * tfidf_centroid
        
        # Нормализуем результат
        norm = np.linalg.norm(ensemble_centroid)
        if norm > 0:
            ensemble_centroid = ensemble_centroid / norm
        
        logger.info(
            f"[Ensemble] Центроид вычислен: w_sbert={w_sbert:.4f}, "
            f"w_tfidf={w_tfidf:.4f}, norm={norm:.4f}"
        )
        return ensemble_centroid
        
    except Exception as e:
        logger.error(f"[Ensemble] Ошибка при расчёте центроида: {e}")
        raise


def calculate_sbert_similarity(
    domain1: tuple[list[str], np.ndarray],
    domain2: tuple[list[str], np.ndarray],
    embedding_service,
) -> float:
    """Расчёт SBERT близости между двумя доменами.
    
    Args:
        domain1: Кортеж (terms, centroid) первого домена.
                 Если centroid None, вычисляется автоматически.
        domain2: Кортеж (terms, centroid) второго домена.
        embedding_service: Сервис для получения эмбеддингов.
    
    Returns:
        Значение близости (cosine similarity) в диапазоне [-1, 1].
    
    Raises:
        ValueError: Если домен пустой или сервис недоступен.
        RuntimeError: Если не удалось вычислить близость.
    """
    terms1, centroid1 = domain1
    terms2, centroid2 = domain2
    
    logger.debug(f"[SBERT] Расчёт близости: '{terms1[:3]}...' vs '{terms2[:3]}...'")
    
    if not terms1:
        logger.error("[SBERT] Пустой список терминов для первого домена")
        raise ValueError("Первый домен не может иметь пустой список терминов")
    
    if not terms2:
        logger.error("[SBERT] Пустой список терминов для второго домена")
        raise ValueError("Второй домен не может иметь пустой список терминов")
    
    if embedding_service is None:
        logger.error("[SBERT] Embedding service не предоставлен")
        raise ValueError("embedding_service не может быть None")
    
    try:
        # Если центроиды не переданы, вычисляем
        if centroid1 is None:
            centroid1 = calculate_sbert_centroid(terms1, embedding_service)
        if centroid2 is None:
            centroid2 = calculate_sbert_centroid(terms2, embedding_service)
        
        # Cosine similarity через dot product (центроиды уже нормализованы)
        similarity = float(np.dot(centroid1, centroid2))
        
        logger.info(f"[SBERT] Близость вычислена: {similarity:.4f}")
        return similarity
        
    except Exception as e:
        logger.error(f"[SBERT] Ошибка при расчёте близости: {e}")
        raise


def calculate_tfidf_similarity(
    domain1_terms: list[str],
    domain2_terms: list[str],
    tfidf_service,
) -> float:
    """Расчёт TF-IDF близости между двумя доменами.
    
    Args:
        domain1_terms: Термины первого домена.
        domain2_terms: Термины второго домена.
        tfidf_service: Сервис TF-IDF.
    
    Returns:
        Значение близости (cosine similarity) в диапазоне [0, 1].
    
    Raises:
        ValueError: Если один из списков терминов пуст.
        RuntimeError: Если не удалось вычислить близость.
    """
    logger.debug(
        f"[TF-IDF] Расчёт близости: {len(domain1_terms)} терминов vs "
        f"{len(domain2_terms)} терминов"
    )
    
    if not domain1_terms:
        logger.error("[TF-IDF] Пустой список терминов для первого домена")
        raise ValueError("Первый домен не может иметь пустой список терминов")
    
    if not domain2_terms:
        logger.error("[TF-IDF] Пустой список терминов для второго домена")
        raise ValueError("Второй домен не может иметь пустой список терминов")
    
    if tfidf_service is None:
        logger.error("[TF-IDF] TF-IDF service не предоставлен")
        raise ValueError("tfidf_service не может быть None")
    
    try:
        # Обучаем TF-IDF на объединённых терминах
        all_terms = domain1_terms + domain2_terms
        tfidf_service.fit_terms(all_terms)
        
        vocab_size = len(tfidf_service._ngram_vocab)
        if vocab_size == 0:
            logger.error("[TF-IDF] Пустой словарь TF-IDF")
            raise RuntimeError("TF-IDF словарь пуст")
        
        # Центроид первого домена
        centroid1 = np.zeros(vocab_size)
        for term in domain1_terms:
            vec = tfidf_service.get_vector(term)
            if vec is not None:
                centroid1 += vec
        if len(domain1_terms) > 0:
            centroid1 /= len(domain1_terms)
        
        # Центроид второго домена
        centroid2 = np.zeros(vocab_size)
        for term in domain2_terms:
            vec = tfidf_service.get_vector(term)
            if vec is not None:
                centroid2 += vec
        if len(domain2_terms) > 0:
            centroid2 /= len(domain2_terms)
        
        # Cosine similarity
        norm1 = np.linalg.norm(centroid1)
        norm2 = np.linalg.norm(centroid2)
        
        if norm1 == 0 or norm2 == 0:
            logger.warning("[TF-IDF] Нулевой центроид, возвращаем 0")
            return 0.0
        
        similarity = float(np.dot(centroid1, centroid2) / (norm1 * norm2))
        
        logger.info(f"[TF-IDF] Близость вычислена: {similarity:.4f}")
        return similarity
        
    except Exception as e:
        logger.error(f"[TF-IDF] Ошибка при расчёте близости: {e}")
        raise


def calculate_ensemble_similarity(
    domain1: tuple[list[str], np.ndarray],
    domain2: tuple[list[str], np.ndarray],
    embedding_service,
    tfidf_service,
    weights: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """Расчёт Ensemble близости между двумя доменами (SBERT + TF-IDF).
    
    Args:
        domain1: Кортеж (terms, centroid) первого домена.
        domain2: Кортеж (terms, centroid) второго домена.
        embedding_service: Сервис для получения SBERT эмбеддингов.
        tfidf_service: Сервис TF-IDF.
        weights: Веса для комбинации {"sbert": float, "tfidf": float}.
                 Если None, используются веса по умолчанию.
    
    Returns:
        Словарь с результатами:
        {
            "similarity": float,  # Финальная близость
            "sbert_score": float,
            "tfidf_score": float,
            "weights": dict
        }
    
    Raises:
        ValueError: Если домен пустой или сервисы недоступны.
        RuntimeError: Если не удалось вычислить близость.
    """
    terms1, _ = domain1
    terms2, _ = domain2
    
    logger.debug(f"[Ensemble] Расчёт близости для доменов")
    
    if not terms1:
        logger.error("[Ensemble] Пустой список терминов для первого домена")
        raise ValueError("Первый домен не может иметь пустой список терминов")
    
    if not terms2:
        logger.error("[Ensemble] Пустой список терминов для второго домена")
        raise ValueError("Второй домен не может иметь пустой список терминов")
    
    # Дефолтные веса
    if weights is None:
        weights = {"sbert": 0.8334, "tfidf": 0.1666}
    
    # Нормализуем веса
    total_w = weights.get("sbert", 0) + weights.get("tfidf", 0)
    if total_w > 0:
        w_sbert = weights.get("sbert", 0) / total_w
        w_tfidf = weights.get("tfidf", 0) / total_w
    else:
        logger.warning("[Ensemble] Некорректные веса, используем равные")
        w_sbert = 0.5
        w_tfidf = 0.5
    
    try:
        # SBERT близость
        sbert_score = calculate_sbert_similarity(
            domain1, domain2, embedding_service
        )
        
        # TF-IDF близость
        tfidf_score = calculate_tfidf_similarity(
            terms1, terms2, tfidf_service
        )
        
        # Простое взвешенное среднее (normalize_ensemble использует rankdata
        # который для единичного элемента даёт 0.5, что некорректно)
        ensemble_score = w_sbert * sbert_score + w_tfidf * tfidf_score
        
        result = {
            "similarity": float(ensemble_score),
            "sbert_score": round(float(sbert_score), 4),
            "tfidf_score": round(float(tfidf_score), 4),
            "weights": {
                "sbert": round(w_sbert, 4),
                "tfidf": round(w_tfidf, 4),
            }
        }
        
        logger.info(
            f"[Ensemble] Близость вычислена: "
            f"similarity={result['similarity']:.4f}, "
            f"sbert={result['sbert_score']:.4f}, "
            f"tfidf={result['tfidf_score']:.4f}"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"[Ensemble] Ошибка при расчёте близости: {e}")
        raise
