# backend/src/presentation/routes.py
# Роутеры API
#
# Версия: 1.2
# Обновлено: 2026-04-10
# Изменения: исправлены пути к датасетам для Docker (/app/data/)

"""
API роутеры для эндпоинтов сервиса.
"""

import os
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form, Request

from .schemas import (
    SimilarityResponse,
    GraphResponse,
    GraphResponseDetailed,
    GraphNodeDetailed,
    DomainsResponse,
    HealthResponse,
    UploadDataRequest,
    UploadDataResponse,
    WordNetSimilarityResponse,
    WordNetDomainSimilarityResponse,
    HypernymsResponse,
    SynsetInfoSchema,
    BenchmarkRequest,
    BenchmarkResponse,
    BenchmarkDatasetsResponse,
    MethodResultSchema,
    BenchmarkComparisonSchema,
    EnrichmentInfoResponse,
    EnrichedSimilarityResponse,
    BenchmarkMatrixResponse,
    BenchmarkMatrixRow,
    BenchmarkMatrixCell,
    WikipediaSimilarityResponse,
)
from ..application import SimilarityService, CentroidService
from ..application.benchmark_service import BenchmarkService
from ..application.similarity_methods import (
    calculate_sbert_centroid,
    calculate_tfidf_centroid,
    calculate_ensemble_centroid,
    calculate_sbert_similarity,
    calculate_tfidf_similarity,
    calculate_ensemble_similarity,
)
from ..infrastructure.data_loader import (
    DataLoaderError,
    CSVDataLoader,
    JSONDataLoader,
    get_loader,
)
from ..infrastructure.cache_manager import CacheManager
from ..infrastructure.embedding_service import EmbeddingService
from ..infrastructure.enriched_embedding_service import EnrichedEmbeddingService
from ..infrastructure.wordnet_service import (
    WordNetService,
    WordNetNotInitializedError,
    WordNetServiceError,
)
from ..infrastructure.tfidf_service import TfidfService

router = APIRouter()

# === Глобальный кеш для RAG индексов ===
# Хранит загруженные FAISS индексы в памяти для повторного использования
_rag_index_cache: dict[str, tuple] = {}  # cache_key -> (index, metadata)


def _get_cached_rag_index(cache_key: str) -> tuple | None:
    """Получить кешированный RAG индекс.
    
    Args:
        cache_key: Уникальный ключ индекса.
    
    Returns:
        Кортеж (index, metadata) если есть в кеше, иначе None.
    """
    return _rag_index_cache.get(cache_key)


def _cache_rag_index(cache_key: str, index, metadata: dict) -> None:
    """Сохранить RAG индекс в кеш.
    
    Args:
        cache_key: Уникальный ключ индекса.
        index: FAISS индекс.
        metadata: Метаданные индекса.
    """
    _rag_index_cache[cache_key] = (index, metadata)
    logger.info(f"[rag_cache] Индекс '{cache_key}' закеширован, всего в кеше: {len(_rag_index_cache)}")


def _clear_rag_cache() -> None:
    """Очистить весь RAG кеш."""
    global _rag_index_cache
    _rag_index_cache.clear()
    logger.info("[rag_cache] RAG кеш очищен")

# Базовый путь к данным (для Docker: /app/data, для локальной разработки: data)
DATA_BASE = os.getenv("DATA_BASE", "/app/data")

# Пути к датасетам бенчмарков
BENCHMARK_DATASETS = {
    "hj-rg": f"{DATA_BASE}/hj-rg.csv",
    "simlex999": f"{DATA_BASE}/simlex999.csv",
    "simlex999_rus": f"{DATA_BASE}/simlex999_rus_without_dupl.csv",
}


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Проверка здоровья API",
    responses={
        200: {
            "description": "Сервис работает корректно",
            "model": HealthResponse,
        }
    },
)
async def health():
    """Проверка здоровья API.
    
    Простой эндпоинт для проверки доступности сервиса.
    Используется для мониторинга и проверки connectivity.
    
    Returns:
        HealthResponse: Статус сервиса всегда "ok" при успехе.
    
    Example:
        ```bash
        curl http://localhost:8000/api/v1/health
        ```
        
        ```json
        {"status": "ok"}
        ```
    """
    return HealthResponse(status="ok")


@router.get(
    "/rag/status",
    tags=["RAG"],
    summary="Проверка статуса RAG индексов",
    responses={
        200: {
            "description": "Статус RAG индексов",
        },
    },
)
async def get_rag_status():
    """Проверка статуса RAG индексов.
    
    Возвращает информацию о наличии предпосчитанных FAISS индексов
    для RAG-метода. Если индексы не построены, метод RAG
    недоступен для использования.
    
    Returns:
        dict: Статус RAG индексов:
            - built: bool - индексы построены
            - terms_count: int - количество терминов в индексе
            - domains_count: int - количество доменов
            - index_path: str - путь к индексу
            - message: str - инструкция для пользователя
    
    Example:
        ```bash
        curl http://localhost:8000/api/v1/rag/status
        ```
        
        Если индексы построены:
        ```json
        {
            "built": true,
            "terms_count": 500,
            "domains_count": 10,
            "index_path": "data/rag_indices/domains.index"
        }
        ```
        
        Если индексы НЕ построены:
        ```json
        {
            "built": false,
            "message": "RAG индексы не найдены. Запустите: python -m scripts.build_rag_index"
        }
        ```
    """
    import pickle
    
    # Путь к директории с индексами
    rag_dir = Path(f"{DATA_BASE}/rag_indices")
    index_path = rag_dir / "domains.index"
    metadata_path = rag_dir / "domains.meta.pkl"
    
    if not index_path.exists() or not metadata_path.exists():
        return {
            "built": False,
            "message": "RAG индексы не найдены. Запустите: python -m scripts.build_rag_index"
        }
    
    try:
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        return {
            "built": True,
            "terms_count": len(metadata.get("terms", [])),
            "domains_count": len(metadata.get("domains", [])),
            "index_path": str(index_path),
            "model_name": metadata.get("model_name", "unknown"),
        }
    except Exception as e:
        return {
            "built": False,
            "message": f"Ошибка чтения индексов: {e}"
        }


# ==== Cache Management Endpoints ====

@router.post(
    "/cache/warmup",
    tags=["Cache"],
    summary="Warm-up кеша из диска",
    responses={
        200: {
            "description": "Результат warm-up",
        },
    },
)
async def warmup_cache(request: Request, method: str = Query(default=None, description="Метод (sbert, rag, wordnet, bert). Если None - все.")):
    """Warm-up Redis кеша из файлов на диске.
    
    При старте приложения загружает все similarity результаты
    из JSON файлов в Redis для быстрого доступа.
    
    Args:
        request: HTTP request.
        method: Метод для загрузки. Если None — все методы.
    
    Returns:
        dict: Количество загруженных записей.
    
    Example:
        ```bash
        curl -X POST http://localhost:8000/api/v1/cache/warmup
        curl -X POST "http://localhost:8000/api/v1/cache/warmup?method=sbert"
        ```
    """
    cache: CacheManager = request.app.state.cache
    loaded = await cache.warm_up(method=method)
    return {"loaded": loaded, "message": f"Загружено {loaded} записей в Redis кеш"}


@router.get(
    "/cache/stats",
    tags=["Cache"],
    summary="Статистика кеша",
    responses={
        200: {
            "description": "Статистика Redis и Disk кешей",
        },
    },
)
async def get_cache_stats(request: Request):
    """Получение статистики кеша.
    
    Возвращает информацию о:
    - Redis: количество ключей по префиксам
    - Disk: количество JSON файлов по методам
    
    Returns:
        dict: Статистика кешей.
    
    Example:
        ```bash
        curl http://localhost:8000/api/v1/cache/stats
        ```
    """
    cache: CacheManager = request.app.state.cache
    
    # Redis stats
    redis_stats = {
        "embeddings": 0,
        "centroids": 0,
        "similarity": 0,
    }
    
    for prefix, key in [
        ("embeddings", cache.EMB_PREFIX),
        ("centroids", cache.CENTROID_PREFIX),
        ("similarity", cache.SIM_PREFIX),
    ]:
        keys = []
        async for k in cache.client.scan_iter(match=f"{key}*"):
            keys.append(k)
        redis_stats[prefix] = len(keys)
    
    # Disk stats
    disk_stats = cache.disk_cache.get_stats() if cache.disk_cache else {"total": 0, "methods": {}}
    
    return {
        "redis": redis_stats,
        "disk": disk_stats,
    }


@router.post(
    "/cache/clear",
    tags=["Cache"],
    summary="Очистка кеша",
    responses={
        200: {
            "description": "Результат очистки",
        },
    },
)
async def clear_cache(request: Request, target: str = Query(default="all", enum=["all", "redis", "disk"], description="Что очищать")):
    """Очистка кеша.
    
    Args:
        request: HTTP request.
        target: Что очищать (all, redis, disk).
    
    Returns:
        dict: Результат очистки.
    
    Example:
        ```bash
        curl -X POST http://localhost:8000/api/v1/cache/clear?target=redis
        ```
    """
    cache: CacheManager = request.app.state.cache
    
    cleared = {"redis": 0, "disk": 0}
    
    if target in ["all", "redis"]:
        await cache.clear_all()
        cleared["redis"] = -1  # flushdb не возвращает count
    
    if target in ["all", "disk"]:
        if cache.disk_cache:
            cleared["disk"] = cache.disk_cache.clear_all()
    
    return {"cleared": cleared, "message": f"Очищено: Redis={cleared['redis']}, Disk={cleared['disk']}"}


@router.get(
    "/status",
    tags=["System"],
    summary="Получение статуса системы",
    responses={
        200: {
            "description": "Текущий статус системы",
        }
    },
)
async def get_system_status(request: Request):
    """Получение статуса системы.
    
    Возвращает информацию о заблокированных операциях.
    Используется фронтендом для проверки перед запуском тяжёлых операций.
    
    Args:
        request: HTTP request с доступом к состоянию приложения.
    
    Returns:
        dict: Статус системы с информацией о блокировках.
    
    Example:
        ```bash
        curl http://localhost:8000/api/v1/status
        ```
    """
    cache: CacheManager = request.app.state.cache
    status = await cache.get_system_status()
    return status


@router.get(
    "/ping",
    tags=["Health"],
    summary="Проверка связи (ping-pong)",
    responses={
        200: {
            "description": "Ответ на ping",
            "content": {
                "application/json": {
                    "example": {"pong": True}
                }
            },
        }
    },
)
async def ping():
    """Проверка связи (ping-pong).
    
    Простой эндпоинт для проверки connectivity.
    Всегда возвращает `{"pong": true}` при успехе.
    
    Returns:
        dict: Ответ `{"pong": True}`.
    
    Example:
        ```bash
        curl http://localhost:8000/api/v1/ping
        ```
        
        ```json
        {"pong": true}
        ```
    """
    return {"pong": True}


@router.get(
    "/domains",
    response_model=DomainsResponse,
    tags=["Domains"],
    summary="Получение списка доменов",
    responses={
        200: {
            "description": "Список названий доменов",
            "model": DomainsResponse,
        }
    },
)
async def get_domains(request: Request):
    """Получение списка всех доменов.
    
    Возвращает список названий доменов из загруженных данных.
    Если данные ещё не загружены, возвращает пустой список.
    
    Args:
        request: HTTP request с доступом к data_loader.
    
    Returns:
        DomainsResponse: Список названий доменов.
    
    Example:
        ```bash
        curl http://localhost:8000/api/v1/domains
        ```
        
        ```json
        {
            "domains": ["machine_learning", "deep_learning", "statistics"]
        }
        ```
    """
    data_loader = getattr(request.app.state, 'data_loader', None)
    if data_loader is None:
        return DomainsResponse(domains=[])
    return DomainsResponse(domains=data_loader.domain_names)


@router.post(
    "/upload/json",
    response_model=UploadDataResponse,
    tags=["Data"],
    summary="Загрузка данных в JSON формате",
    responses={
        200: {
            "description": "Данные успешно загружены",
            "model": UploadDataResponse,
        },
        400: {
            "description": "Ошибка в формате данных",
        },
    },
)
async def upload_json(request: Request, body: UploadDataRequest):
    """Загрузка данных в JSON формате.
    
    Принимает JSON с описанием доменов и терминов.
    Загруженные данные доступны для всех последующих операций.
    
    Args:
        request: HTTP request.
        body: UploadDataRequest с доменами и терминами.
    
    Returns:
        UploadDataResponse: Результат загрузки с количеством загруженных данных.
    
    Raises:
        HTTPException: При ошибке парсинга или загрузки данных.
    
    Example:
        ```bash
        curl -X POST http://localhost:8000/api/v1/upload/json \
          -H "Content-Type: application/json" \
          -d '{
            "domains": [
              {
                "name": "ML",
                "terms": [
                  {"name": "нейронная сеть", "frequency": 10},
                  {"name": "gradient descent", "frequency": 5}
                ]
              }
            ]
          }'
        ```
        
        ```json
        {
            "success": true,
            "domains_loaded": 1,
            "terms_loaded": 2,
            "message": "Загружено 2 терминов из 1 доменов"
        }
        ```
    """
    try:
        loader = JSONDataLoader()
        terms_count = loader.load_from_request(body)
        request.app.state.data_loader = loader
        
        return UploadDataResponse(
            success=True,
            domains_loaded=len(loader.domain_names),
            terms_loaded=terms_count,
            message=f"Загружено {terms_count} терминов из {len(loader.domain_names)} доменов"
        )
    except DataLoaderError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/upload/file",
    response_model=UploadDataResponse,
    tags=["Data"],
    summary="Загрузка файла с данными",
    responses={
        200: {
            "description": "Файл успешно загружен",
            "model": UploadDataResponse,
        },
        400: {
            "description": "Ошибка загрузки файла",
        },
    },
)
async def upload_file(
    request: Request,
    file: UploadFile = File(..., description="CSV или JSON файл с данными"),
    file_format: str = Form(
        ...,
        description="Формат файла: csv или json",
        examples=["csv"]
    ),
):
    """Загрузка файла с данными.
    
    Загружает CSV или JSON файл с данными о доменах и терминах.
    Поддерживает форматы:
    - CSV: колонки domain, term, frequency
    - JSON: структура с массивом доменов и их терминов
    
    Args:
        request: HTTP request.
        file: Загружаемый файл.
        file_format: Формат файла ("csv" или "json").
    
    Returns:
        UploadDataResponse: Результат загрузки.
    
    Raises:
        HTTPException: При ошибке чтения или парсинга файла.
    
    Example:
        ```bash
        curl -X POST http://localhost:8000/api/v1/upload/file \
          -F "file=@data/terms.csv" \
          -F "file_format=csv"
        ```
    """
    try:
        loader = get_loader(file_format)
        content = await file.read()
        content_str = content.decode("utf-8")
        
        if isinstance(loader, CSVDataLoader):
            terms_count = loader.load_string(content_str)
        else:
            terms_count = loader.load_string(content_str)
        
        request.app.state.data_loader = loader
        
        return UploadDataResponse(
            success=True,
            domains_loaded=len(loader.domain_names),
            terms_loaded=terms_count,
            message=f"Загружено {terms_count} терминов из {len(loader.domain_names)} доменов"
        )
    except DataLoaderError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки: {e}")


@router.get(
    "/domains/{domain1}/similarity/{domain2}",
    response_model=SimilarityResponse,
    tags=["Similarity"],
    summary="Расчёт близости между доменами",
    responses={
        200: {
            "description": "Значение близости между доменами",
            "model": SimilarityResponse,
        },
        400: {
            "description": "Данные не загружены или домены одинаковые",
        },
        404: {
            "description": "Домен не найден",
        },
    },
)
async def get_similarity(
    request: Request,
    domain1: str,
    domain2: str,
    metric: str = Query(
        default="cosine",
        enum=["cosine", "euclidean"],
        description="Метрика близости: cosine или euclidean",
    ),
):
    """Расчёт близости между двумя доменами.
    
    Вычисляет семантическую близость между доменами на основе
    эмбеддингов их терминов. Использует центроидный подход:
    
    1. Получает эмбеддинги всех терминов каждого домена (SBERT)
    2. Вычисляет центроид (средний вектор) для каждого домена
    3. Рассчитывает близость между центроидами
    
    Результат кэшируется в Redis для повторных запросов.
    
    Args:
        request: HTTP request.
        domain1: Название первого домена.
        domain2: Название второго домена.
        metric: Метрика близости:
            - `cosine`: Косинусное сходство (по умолчанию)
            - `euclidean`: Евклидово расстояние (меньше = ближе)
    
    Returns:
        SimilarityResponse: Значение близости между доменами.
    
    Raises:
        HTTPException 400: Если данные не загружены или домены одинаковые.
        HTTPException 404: Если домен не найден в данных.
    
    Example:
        ```bash
        curl "http://localhost:8000/api/v1/domains/ML/DL/similarity?metric=cosine"
        ```
        
        ```json
        {
            "domain1": "ML",
            "domain2": "DL",
            "score": 0.847,
            "metric": "cosine"
        }
        ```
    """
    data_loader = getattr(request.app.state, 'data_loader', None)
    
    if data_loader is None:
        raise HTTPException(
            status_code=400,
            detail="Данные не загружены. Сначала загрузите данные через /upload/json или /upload/file"
        )
    
    if domain1 == domain2:
        raise HTTPException(
            status_code=400, 
            detail="Для сравнения домена с самим собой используйте значение 1.0"
        )
    
    if domain1 not in data_loader._domains:
        raise HTTPException(status_code=404, detail=f"Домен '{domain1}' не найден")
    if domain2 not in data_loader._domains:
        raise HTTPException(status_code=404, detail=f"Домен '{domain2}' не найден")
    
    cache: CacheManager = request.app.state.cache
    cached_score = await cache.get_similarity(domain1, domain2, metric)
    if cached_score is not None:
        return SimilarityResponse(
            domain1=domain1,
            domain2=domain2,
            score=cached_score,
            metric=metric,
        )
    
    # Используем предзагруженный сервис
    embedding_service = getattr(request.app.state, 'embedding_service', None)
    if embedding_service is None:
        embedding_service = EmbeddingService(cache=cache)
    
    centroid_service = CentroidService()
    similarity_service = SimilarityService(metric=metric)
    
    domain1_obj = data_loader._domains[domain1]
    domain2_obj = data_loader._domains[domain2]
    
    terms1 = [term.name for term in domain1_obj.terms]
    terms2 = [term.name for term in domain2_obj.terms]
    
    embeddings1 = embedding_service.get_embeddings_batch(terms1)
    embeddings2 = embedding_service.get_embeddings_batch(terms2)
    
    centroid1 = centroid_service.calculate_centroid(embeddings1)
    centroid2 = centroid_service.calculate_centroid(embeddings2)
    
    score = similarity_service.calculate_similarity(centroid1, centroid2)
    
    await cache.set_similarity(domain1, domain2, score, metric)
    await cache.set_similarity(domain2, domain1, score, metric)
    
    return SimilarityResponse(
        domain1=domain1,
        domain2=domain2,
        score=score,
        metric=metric,
    )


@router.get(
    "/graph",
    response_model=GraphResponse,
    tags=["Graph"],
    summary="Получение графа близости доменов",
    responses={
        200: {
            "description": "Граф с узлами доменов и рёбрами близости",
            "model": GraphResponse,
        }
    },
)
async def get_graph(
    request: Request,
    threshold: float = Query(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Порог близости для включения ребра [0.0, 1.0]",
    ),
    metric: str = Query(
        default="cosine",
        enum=["cosine", "euclidean"],
        description="Метрика близости: cosine или euclidean",
    ),
):
    """Получение графа близости доменов.
    
    Строит граф, где:
    - Узлы — домены из загруженных данных
    - Рёбра — пары доменов с близостью >= threshold
    
    Используется для визуализации связей между предметными областями.
    
    Args:
        request: HTTP request.
        threshold: Минимальное значение близости для отображения ребра.
            По умолчанию 0.5. Рёбра с близостью ниже порога скрываются.
        metric: Метрика для расчёта близости.
    
    Returns:
        GraphResponse: Граф с узлами и рёбрами.
    
    Example:
        ```bash
        curl "http://localhost:8000/api/v1/graph?threshold=0.6"
        ```
        
        ```json
        {
            "nodes": [
                {"id": "ML", "label": "ML", "type": "domain"},
                {"id": "DL", "label": "DL", "type": "domain"}
            ],
            "edges": [
                {"source": "ML", "target": "DL", "weight": 0.72, "type": "similarity"}
            ],
            "threshold": 0.6
        }
        ```
    """
    data_loader = getattr(request.app.state, 'data_loader', None)
    
    if data_loader is None or not data_loader.domain_names:
        return GraphResponse(nodes=[], edges=[], threshold=threshold)
    
    # Используем предзагруженный сервис
    embedding_service = getattr(request.app.state, 'embedding_service', None)
    if embedding_service is None:
        embedding_service = EmbeddingService(cache=cache)
    
    centroid_service = CentroidService()
    similarity_service = SimilarityService(metric=metric)
    
    domains = []
    centroids = {}
    
    for domain_name in data_loader.domain_names:
        domain_obj = data_loader._domains[domain_name]
        terms = [term.name for term in domain_obj.terms]
        embeddings = embedding_service.get_embeddings_batch(terms)
        centroid = centroid_service.calculate_centroid(embeddings)
        
        domains.append(domain_obj)
        centroids[domain_name] = centroid
    
    graph_data = similarity_service.build_graph(domains, centroids, threshold)
    
    return GraphResponse(
        nodes=[{"id": n["id"], "label": n["label"]} for n in graph_data.nodes],
        edges=graph_data.edges,
        threshold=threshold,
    )


# ==== RuWordNet эндпоинты ====

def _get_wordnet_service(request: Request) -> WordNetService:
    """Получение или создание сервиса WordNet.
    
    Вспомогательная функция для инициализации WordNet сервиса.
    Сервис создаётся один раз и кэшируется в состоянии приложения.
    
    Args:
        request: HTTP request.
    
    Returns:
        WordNetService: Инициализированный сервис.
    
    Raises:
        HTTPException: Если RuWordNet не инициализирован.
    """
    wordnet_service: WordNetService = getattr(request.app.state, 'wordnet_service', None)
    
    if wordnet_service is None:
        # Путь к базе данных из переменной окружения
        db_path = os.environ.get("RUWORDNET_DB_PATH")
        wordnet_service = WordNetService(db_path=db_path)
        try:
            wordnet_service.initialize()
            request.app.state.wordnet_service = wordnet_service
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"RuWordNet не инициализирован: {e}"
            )
    
    return wordnet_service


@router.get(
    "/wordnet/similarity/{term1}/{term2}",
    response_model=WordNetSimilarityResponse,
    tags=["WordNet"],
    summary="Расчёт близости терминов через RuWordNet",
    responses={
        200: {
            "description": "Близость терминов с информацией о синсетах",
            "model": WordNetSimilarityResponse,
        },
        500: {
            "description": "RuWordNet не инициализирован или ошибка расчёта",
        },
    },
)
async def get_wordnet_similarity(
    request: Request,
    term1: str,
    term2: str,
    algorithm: str = Query(
        default="lin",
        enum=["lin", "wup", "path"],
        description="Алгоритм: lin, wup или path",
    ),
):
    """Расчёт семантической близости между терминами через RuWordNet.
    
    Вычисляет близость двух терминов на основе их позиции в иерархии
    RuWordNet. Использует три алгоритма:
    
    - **Lin**: Information Content на основе LCS (Lowest Common Subsumer)
    - **Wu-Palmer**: Глубинная метрика с учётом глубины терминов
    - **Path**: Обратная длина кратчайшего пути
    
    Args:
        request: HTTP request.
        term1: Первый термин (на русском языке).
        term2: Второй термин (на русском языке).
        algorithm: Алгоритм расчёта близости.
    
    Returns:
        WordNetSimilarityResponse: Значение близости и информация о синсетах.
    
    Raises:
        HTTPException: При ошибке инициализации RuWordNet.
    
    Example:
        ```bash
        curl "http://localhost:8000/api/v1/wordnet/similarity/кот/собака?algorithm=lin"
        ```
        
        ```json
        {
            "term1": "кот",
            "term2": "собака",
            "similarity": 0.78,
            "algorithm": "lin",
            "synset1": {
                "synset_id": "123-N",
                "title": "кошка",
                "gloss": "домашнее животное...",
                "depth": 8,
                "ic": 0.01
            },
            "synset2": {
                "synset_id": "124-N",
                "title": "собака",
                "gloss": "домашнее животное...",
                "depth": 8,
                "ic": 0.01
            },
            "lcs": {
                "synset_id": "100-N",
                "title": "млекопитающее",
                "gloss": "...",
                "depth": 6,
                "ic": 0.05
            }
        }
        ```
    """
    try:
        wordnet_service = _get_wordnet_service(request)
        result = wordnet_service.get_similarity(term1, term2, algorithm)
        
        return WordNetSimilarityResponse(
            term1=result.term1,
            term2=result.term2,
            similarity=result.similarity,
            algorithm=result.algorithm,
            synset1=SynsetInfoSchema(
                synset_id=result.synset1.synset_id,
                title=result.synset1.title,
                gloss=result.synset1.gloss,
                depth=result.synset1.depth,
                ic=result.synset1.ic,
            ) if result.synset1 else None,
            synset2=SynsetInfoSchema(
                synset_id=result.synset2.synset_id,
                title=result.synset2.title,
                gloss=result.synset2.gloss,
                depth=result.synset2.depth,
                ic=result.synset2.ic,
            ) if result.synset2 else None,
            lcs=SynsetInfoSchema(
                synset_id=result.lcs.synset_id,
                title=result.lcs.title,
                gloss=result.lcs.gloss,
                depth=result.lcs.depth,
                ic=result.lcs.ic,
            ) if result.lcs else None,
        )
    except WordNetNotInitializedError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except WordNetServiceError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/wordnet/domains/{domain1}/{domain2}",
    response_model=WordNetDomainSimilarityResponse,
    tags=["WordNet"],
    summary="Расчёт близости доменов через RuWordNet",
    responses={
        200: {
            "description": "Близость доменов через WordNet",
            "model": WordNetDomainSimilarityResponse,
        },
        400: {
            "description": "Данные не загружены",
        },
        404: {
            "description": "Домен не найден",
        },
    },
)
async def get_wordnet_domain_similarity(
    request: Request,
    domain1: str,
    domain2: str,
    algorithm: str = Query(
        default="lin",
        enum=["lin", "wup", "path"],
        description="Алгоритм: lin, wup или path",
    ),
    aggregation: str = Query(
        default="max",
        enum=["max", "mean", "min"],
        description="Метод агрегации: max, mean или min",
    ),
):
    """Расчёт близости между доменами через RuWordNet.
    
    Вычисляет близость двух доменов путём попарного сравнения
    их терминов через RuWordNet с последующей агрегацией результатов.
    
    Args:
        request: HTTP request.
        domain1: Название первого домена.
        domain2: Название второго домена.
        algorithm: Алгоритм расчёта близости терминов.
        aggregation: Метод агрегации результатов:
            - `max`: Максимальная близость среди пар
            - `mean`: Средняя близость всех пар
            - `min`: Минимальная близость среди пар
    
    Returns:
        WordNetDomainSimilarityResponse: Значение близости и статистика.
    
    Raises:
        HTTPException 400: Если данные не загружены.
        HTTPException 404: Если домен не найден.
    
    Example:
        ```bash
        curl "http://localhost:8000/api/v1/wordnet/domains/ML/DL?algorithm=lin&aggregation=max"
        ```
    """
    data_loader = getattr(request.app.state, 'data_loader', None)
    
    if data_loader is None:
        raise HTTPException(
            status_code=400,
            detail="Данные не загружены. Сначала загрузите данные через /upload/json или /upload/file"
        )
    
    if domain1 not in data_loader._domains:
        raise HTTPException(status_code=404, detail=f"Домен '{domain1}' не найден")
    if domain2 not in data_loader._domains:
        raise HTTPException(status_code=404, detail=f"Домен '{domain2}' не найден")
    
    try:
        wordnet_service = _get_wordnet_service(request)
        
        domain1_obj = data_loader._domains[domain1]
        domain2_obj = data_loader._domains[domain2]
        
        terms1 = [term.name for term in domain1_obj.terms]
        terms2 = [term.name for term in domain2_obj.terms]
        
        similarity = wordnet_service.domain_similarity(
            terms1, terms2, algorithm, aggregation
        )
        
        pairs_count = len(terms1) * len(terms2)
        
        return WordNetDomainSimilarityResponse(
            domain1=domain1,
            domain2=domain2,
            similarity=similarity,
            algorithm=algorithm,
            pairs_count=pairs_count,
            aggregation=aggregation,
        )
    except WordNetNotInitializedError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except WordNetServiceError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/wordnet/hypernyms/{term}",
    response_model=HypernymsResponse,
    tags=["WordNet"],
    summary="Получение иерархии гиперонимов",
    responses={
        200: {
            "description": "Иерархия гиперонимов термина",
            "model": HypernymsResponse,
        },
        500: {
            "description": "Ошибка работы с RuWordNet",
        },
    },
)
async def get_wordnet_hypernyms(request: Request, term: str):
    """Получение иерархии гиперонимов для термина.
    
    Возвращает цепочку обобщений термина от самого частного
    к самому общему понятию в иерархии RuWordNet.
    
    Args:
        request: HTTP request.
        term: Термин для поиска гиперонимов.
    
    Returns:
        HypernymsResponse: Список гиперонимов от частного к общему.
    
    Raises:
        HTTPException: При ошибке работы с RuWordNet.
    
    Example:
        ```bash
        curl "http://localhost:8000/api/v1/wordnet/hypernyms/нейронная%20сеть"
        ```
        
        ```json
        {
            "term": "нейронная сеть",
            "hypernyms": [
                "нейронная сеть",
                "модель машинного обучения",
                "алгоритм",
                "абстракция",
                "сущность"
            ]
        }
        ```
    """
    try:
        wordnet_service = _get_wordnet_service(request)
        hypernyms = wordnet_service.get_hypernyms(term)
        
        return HypernymsResponse(term=term, hypernyms=hypernyms)
    except WordNetNotInitializedError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except WordNetServiceError as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==== Бенчмарк эндпоинты ====

@router.get(
    "/benchmark/datasets",
    response_model=BenchmarkDatasetsResponse,
    tags=["Benchmark"],
    summary="Получение списка датасетов",
    responses={
        200: {
            "description": "Список доступных датасетов",
            "model": BenchmarkDatasetsResponse,
        }
    },
)
async def list_benchmark_datasets():
    """Получение списка доступных датасетов для бенчмаркинга.
    
    Возвращает информацию о доступных бенчмарк-датасетах:
    - hj-rg: Russian semantic similarity benchmark
    - simlex999: English SimLex-999
    - simlex999_rus: Russian translation of SimLex-999
    
    Returns:
        BenchmarkDatasetsResponse: Список датасетов с метаинформацией.
    
    Example:
        ```bash
        curl http://localhost:8000/api/v1/benchmark/datasets
        ```
        
        ```json
        {
            "datasets": [
                {"name": "hj-rg", "path": "/path/to/hj-rg.csv", "size": 500},
                {"name": "simlex999", "path": "/path/to/simlex999.csv", "size": 999}
            ]
        }
        ```
    """
    import pandas as pd
    
    datasets = []
    for name, path in BENCHMARK_DATASETS.items():
        full_path = Path(path)
        if full_path.exists():
            df = pd.read_csv(full_path)
            datasets.append({
                "name": name,
                "path": str(full_path),
                "size": len(df),
            })
    
    return BenchmarkDatasetsResponse(datasets=datasets)


@router.post(
    "/benchmark/run",
    response_model=BenchmarkResponse,
    tags=["Benchmark"],
    summary="Запуск бенчмарка",
    responses={
        200: {
            "description": "Результаты бенчмарка",
            "model": BenchmarkResponse,
        },
        409: {
            "description": "Бенчмарк уже выполняется",
        },
    },
)
async def run_benchmark(request: Request, body: BenchmarkRequest):
    """Запуск бенчмарка на указанном датасете.
    
    Сравнивает различные методы семантической близости на стандартных
    бенчмарках. Методы включают:
    
    - **SBERT (baseline)**: Только эмбеддинги SBERT
    - **SBERT + TF-IDF**: Взвешивание терминов через TF-IDF
    - **SBERT + Z-score**: Нормализация через Z-score
    - **RuWordNet (Lin)**: WordNet similarity (Lin algorithm)
    - **RuWordNet (Wu-Palmer)**: WordNet similarity (Wu-Palmer)
    - **Hybrid (SBERT + RuWordNet)**: Комбинация подходов
    
    Использует Redis mutex для предотвращения параллельных запусков.
    
    Args:
        request: HTTP request.
        body: BenchmarkRequest с названием датасета.
    
    Returns:
        BenchmarkResponse: Результаты всех методов с метриками.
    
    Raises:
        HTTPException 409: Если бенчмарк уже выполняется.
    
    Example:
        ```bash
        curl -X POST http://localhost:8000/api/v1/benchmark/run \
          -H "Content-Type: application/json" \
          -d '{"dataset": "hj-rg"}'
        ```
    """
    cache: CacheManager = request.app.state.cache
    
    # Пытаемся получить блокировку
    if not await cache.acquire_lock("benchmark"):
        return BenchmarkResponse(
            success=False,
            error="Бенчмарк уже выполняется. Подождите завершения текущей операции."
        )
    
    try:
        # Устанавливаем статус
        await cache.set_operation_status("benchmark", "started")
        
        # Определяем путь к датасету
        if body.dataset in BENCHMARK_DATASETS:
            dataset_path = Path(BENCHMARK_DATASETS[body.dataset])
        else:
            dataset_path = Path(body.dataset)
        
        if not dataset_path.exists():
            await cache.set_operation_status("benchmark", "failed")
            return BenchmarkResponse(
                success=False,
                error=f"Датасет не найден: {dataset_path}"
            )
        
        # Создаём сервис бенчмаркинга
        benchmark_service = BenchmarkService()
        
        # Запускаем бенчмарк
        await cache.set_operation_status("benchmark", "processing", 0.1)
        comparison = benchmark_service.run_all(
            dataset_path=str(dataset_path),
            dataset_name=body.dataset
        )
        await cache.set_operation_status("benchmark", "processing", 0.9)
        
        # Конвертируем в схему
        comparison_schema = BenchmarkComparisonSchema(
            dataset_name=comparison.dataset_name,
            dataset_size=comparison.dataset_size,
            execution_time_sec=comparison.execution_time_sec,
            results=[
                MethodResultSchema(
                    method=r.method,
                    spearman=r.spearman,
                    pearson=r.pearson,
                    mse=r.mse,
                    missing=r.missing,
                    predictions_count=r.predictions_count,
                )
                for r in comparison.results
            ]
        )
        
        await cache.set_operation_status("benchmark", "completed", 1.0)

        # Сохраняем результаты в SQLite для персистентного хранения
        from ..infrastructure.results_storage import get_storage
        storage = get_storage()
        storage.save_result(
            dataset=body.dataset,
            results=[
                {
                    "method": r.method,
                    "spearman": r.spearman,
                    "pearson": r.pearson,
                    "mse": r.mse,
                    "missing": r.missing,
                    "predictions_count": r.predictions_count,
                }
                for r in comparison.results
            ],
            execution_time_sec=comparison.execution_time_sec,
        )

        return BenchmarkResponse(
            success=True,
            comparison=comparison_schema,
        )

    except Exception as e:
        await cache.set_operation_status("benchmark", "failed")
        return BenchmarkResponse(
            success=False,
            error=f"Ошибка выполнения бенчмарка: {str(e)}"
        )
    finally:
        # Освобождаем блокировку
        await cache.release_lock("benchmark")


@router.get(
    "/benchmark/result/{dataset}",
    response_model=BenchmarkResponse,
    tags=["Benchmark"],
    summary="Получение результата бенчмарка",
    responses={
        200: {
            "description": "Результаты бенчмарка",
            "model": BenchmarkResponse,
        },
    },
)
async def get_benchmark_result(request: Request, dataset: str):
    """Получение сохранённых результатов бенчмарка для датасета.
    
    Сначала проверяет SQLite хранилище, если результаты есть —
    возвращает их немедленно. Если нет — запускает вычисления.
    
    Args:
        request: HTTP request.
        dataset: Название датасета (hj-rg, simlex999, simlex999_rus).
    
    Returns:
        BenchmarkResponse: Результаты бенчмарка.
    
    Example:
        ```bash
        curl http://localhost:8000/api/v1/benchmark/result/hj-rg
        ```
    """
    # Проверяем SQLite хранилище
    from ..infrastructure.results_storage import get_benchmark_result as get_stored_result
    
    stored = get_stored_result(dataset)
    if stored:
        # Результаты есть в SQLite — возвращаем сразу
        comparison_schema = BenchmarkComparisonSchema(
            dataset_name=stored["dataset_name"],
            dataset_size=len(stored["results"]) if stored.get("results") else 0,
            execution_time_sec=stored.get("execution_time_sec", 0.0),
            results=[
                MethodResultSchema(
                    method=r["method"],
                    spearman=r["spearman"],
                    pearson=r["pearson"],
                    mse=r["mse"],
                    missing=r["missing"],
                    predictions_count=r["predictions_count"],
                )
                for r in (stored.get("results") or [])
            ]
        )
        
        return BenchmarkResponse(
            success=True,
            comparison=comparison_schema,
        )
    
    # Результатов нет — запускаем async задачу
    body = BenchmarkRequest(dataset=dataset)
    return await run_benchmark(request, body)


@router.get(
    "/benchmark/saved/{dataset}",
    tags=["Benchmark"],
    summary="Получение сохранённых результатов",
    responses={
        200: {
            "description": "Сохранённые результаты",
        },
        404: {
            "description": "Результаты не найдены",
        },
    },
)
async def get_saved_benchmark(request: Request, dataset: str):
    """Получение сохранённых результатов без запуска бенчмарка.
    
    Только читает из SQLite, не запускает вычисления.
    Если результаты не найдены — возвращает 404.
    
    Args:
        request: HTTP request.
        dataset: Название датасета.
    
    Returns:
        dict: Сохранённые результаты с метаинформацией.
    
    Raises:
        HTTPException 404: Если результаты не найдены.
    
    Example:
        ```bash
        curl http://localhost:8000/api/v1/benchmark/saved/hj-rg
        ```
    """
    from ..infrastructure.results_storage import get_storage
    
    storage = get_storage()
    stored = storage.get_result(dataset)
    
    if not stored:
        raise HTTPException(
            status_code=404,
            detail=f"Результаты для датасета '{dataset}' не найдены. Запустите бенчмарк."
        )
    
    return {
        "dataset_name": stored["dataset_name"],
        "dataset_size": len(stored.get("results", [])),
        "execution_time_sec": stored.get("execution_time_sec", 0.0),
        "results": stored.get("results", []),
        "created_at": stored.get("created_at"),
        "updated_at": stored.get("updated_at"),
    }


@router.get(
    "/benchmark/matrix",
    response_model=BenchmarkMatrixResponse,
    tags=["Benchmark"],
    summary="Получение интегральной матрицы результатов",
    responses={
        200: {
            "description": "Интегральная матрица результатов",
            "model": BenchmarkMatrixResponse,
        },
        409: {
            "description": "Бенчмарк уже выполняется",
        },
    },
)
async def get_benchmark_matrix(request: Request):
    """Получение интегральной матрицы результатов бенчмарков.
    
    Запускает бенчмаркинг на всех доступных датасетах и возвращает
    матрицу результатов для всех методов.
    
    Матрица включает:
    - Строки: методы (SBERT, TF-IDF, Z-score, RuWordNet, etc.)
    - Столбцы: датасеты (hj-rg, simlex999, simlex999_rus)
    
    Returns:
        BenchmarkMatrixResponse: Интегральная матрица результатов.
    
    Example:
        ```bash
        curl http://localhost:8000/api/v1/benchmark/matrix
        ```
        
        ```json
        {
            "success": true,
            "results": [
                {
                    "method": "SBERT (baseline)",
                    "hj_rg": {"spearman": 0.65, "pearson": 0.68, "missing": 5, "predictions": 495},
                    "simlex999_rus": {"spearman": 0.58, "pearson": 0.61, "missing": 10, "predictions": 989}
                },
                {
                    "method": "RuWordNet (Lin)",
                    "hj_rg": {"spearman": 0.42, "pearson": 0.45, "missing": 100, "predictions": 400}
                }
            ],
            "execution_time_sec": 45.2
        }
        ```
    """
    import time
    from pathlib import Path
    
    start_time = time.time()
    
    cache: CacheManager = request.app.state.cache
    
    # Пытаемся получить блокировку
    if not await cache.acquire_lock("benchmark"):
        return BenchmarkMatrixResponse(
            success=False,
            results=[],
            execution_time_sec=0.0,
            error="Бенчмарк уже выполняется. Подождите завершения текущей операции."
        )
    
    try:
        await cache.set_operation_status("benchmark", "started")
        
        # Датасеты для матрицы (используем константу с DATA_BASE)
        datasets_to_run = {
            "hj-rg": f"{DATA_BASE}/hj-rg.csv",
            "simlex999": f"{DATA_BASE}/simlex999.csv",
            "simlex999_rus": f"{DATA_BASE}/simlex999_rus_without_dupl.csv",
        }
        
        # Результаты по датасетам
        all_results: dict[str, dict[str, MethodResultSchema]] = {}
        
        # Запускаем бенчмарк для каждого датасета
        for dataset_name, dataset_path in datasets_to_run.items():
            full_path = Path(dataset_path)
            
            if not full_path.exists():
                continue
            
            benchmark_service = BenchmarkService()
            comparison = benchmark_service.run_all(
                dataset_path=str(full_path),
                dataset_name=dataset_name
            )
            
            # Сохраняем результаты по методам
            all_results[dataset_name] = {
                r.method: r for r in comparison.results
            }
        
        # Собираем матрицу
        matrix_rows = []
        
        # Русские методы (для hj-rg, simlex999_rus)
        russian_methods = [
            "SBERT (baseline)",
            "SBERT + TF-IDF",
            "SBERT + Z-score",
            "RuWordNet (Lin)",
            "RuWordNet (Wu-Palmer)",
            "Hybrid (SBERT + RuWordNet)",
        ]
        
        # Английские методы (для simlex999)
        english_methods = [
            "SBERT (baseline)",
            "English WordNet (Lin)",
            "English WordNet (Wu-Palmer)",
        ]
        
        # Собираем все уникальные методы из результатов
        all_methods = set()
        for dataset_results in all_results.values():
            all_methods.update(dataset_results.keys())
        
        # Добавляем русские методы
        for method in russian_methods:
            if method not in all_methods:
                continue
            row = BenchmarkMatrixRow(method=method)
            
            # Результаты для hj-rg
            if "hj-rg" in all_results and method in all_results["hj-rg"]:
                result = all_results["hj-rg"][method]
                row.hj_rg = BenchmarkMatrixCell(
                    spearman=result.spearman,
                    pearson=result.pearson,
                    missing=result.missing,
                    predictions=result.predictions_count,
                )
            
            # Результаты для simlex999_rus (Russian)
            if "simlex999_rus" in all_results and method in all_results["simlex999_rus"]:
                result = all_results["simlex999_rus"][method]
                row.simlex999_rus = BenchmarkMatrixCell(
                    spearman=result.spearman,
                    pearson=result.pearson,
                    missing=result.missing,
                    predictions=result.predictions_count,
                )
            
            matrix_rows.append(row)
        
        # Добавляем английские методы
        for method in english_methods:
            if method not in all_methods:
                continue
            row = BenchmarkMatrixRow(method=method)
            
            # Результаты для simlex999 (English)
            if "simlex999" in all_results and method in all_results["simlex999"]:
                result = all_results["simlex999"][method]
                row.simlex999 = BenchmarkMatrixCell(
                    spearman=result.spearman,
                    pearson=result.pearson,
                    missing=result.missing,
                    predictions=result.predictions_count,
                )
            
            matrix_rows.append(row)
        
        await cache.set_operation_status("benchmark", "completed", 1.0)
        
        return BenchmarkMatrixResponse(
            success=True,
            results=matrix_rows,
            execution_time_sec=time.time() - start_time,
        )
        
    except Exception as e:
        await cache.set_operation_status("benchmark", "failed")
        return BenchmarkMatrixResponse(
            success=False,
            results=[],
            execution_time_sec=time.time() - start_time,
            error=f"Ошибка выполнения бенчмарка: {str(e)}"
        )
    finally:
        await cache.release_lock("benchmark")


# ==== Обогащённые эмбеддинги ====

def _get_enriched_service(request: Request) -> EnrichedEmbeddingService:
    """Получение или создание сервиса обогащённых эмбеддингов.
    
    Args:
        request: HTTP request.
    
    Returns:
        EnrichedEmbeddingService: Сервис обогащённых эмбеддингов.
    """
    enriched_service: EnrichedEmbeddingService = getattr(
        request.app.state, 'enriched_service', None
    )
    
    if enriched_service is None:
        embedding_service = EmbeddingService()
        enriched_service = EnrichedEmbeddingService(
            embedding_service=embedding_service
        )
        request.app.state.enriched_service = enriched_service
    
    return enriched_service


@router.get(
    "/enriched/info/{term}",
    response_model=EnrichmentInfoResponse,
    tags=["Enrichment"],
    summary="Информация об обогащении термина",
    responses={
        200: {
            "description": "Информация о гиперонимах термина",
            "model": EnrichmentInfoResponse,
        },
        500: {
            "description": "Ошибка обогащения",
        },
    },
)
async def get_enrichment_info(request: Request, term: str):
    """Получение информации об обогащении термина.
    
    Возвращает список гиперонимов из RuWordNet и флаг,
    указывающий, было ли применено обогащение.
    
    Args:
        request: HTTP request.
        term: Термин для проверки.
    
    Returns:
        EnrichmentInfoResponse: Информация о гиперонимах.
    
    Example:
        ```bash
        curl "http://localhost:8000/api/v1/enriched/info/нейронная%20сеть"
        ```
        
        ```json
        {
            "term": "нейронная сеть",
            "hypernyms": ["сеть", "модель", "алгоритм"],
            "hypernym_count": 3,
            "enriched": true
        }
        ```
    """
    try:
        enriched_service = _get_enriched_service(request)
        info = enriched_service.get_enrichment_info(term)
        
        return EnrichmentInfoResponse(**info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/enriched/similarity/{domain1}/{domain2}",
    response_model=EnrichedSimilarityResponse,
    tags=["Enrichment"],
    summary="Расчёт близости с обогащёнными эмбеддингами",
    responses={
        200: {
            "description": "Близость с учётом гиперонимов",
            "model": EnrichedSimilarityResponse,
        },
        400: {
            "description": "Данные не загружены",
        },
        404: {
            "description": "Домен не найден",
        },
    },
)
async def get_enriched_similarity(
    request: Request,
    domain1: str,
    domain2: str,
    alpha: float = Query(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Вес оригинального термина (0.0-1.0)",
    ),
    metric: str = Query(
        default="cosine",
        enum=["cosine", "euclidean"],
        description="Метрика близости",
    ),
):
    """Расчёт близости между доменами с обогащёнными эмбеддингами.
    
    Использует гиперонимы из RuWordNet для улучшения
    семантического представления терминов.
    
    Формула расчёта обогащённого эмбеддинга:
    
        emb_final = alpha * emb(term) + (1 - alpha) * mean(emb(hypernyms))
    
    Где:
    - alpha: вес оригинального термина (по умолчанию 0.7)
    - emb(term): эмбеддинг оригинального термина
    - emb(hypernyms): эмбеддинги гиперонимов из RuWordNet
    
    Args:
        request: HTTP request.
        domain1: Первый домен.
        domain2: Второй домен.
        alpha: Вес оригинального термина. При alpha=1.0 используется
            только оригинальный термин. При alpha=0.0 — только гиперонимы.
        metric: Метрика близости.
    
    Returns:
        EnrichedSimilarityResponse: Значение близости и статистика обогащения.
    
    Example:
        ```bash
        curl "http://localhost:8000/api/v1/enriched/similarity/ML/DL?alpha=0.7"
        ```
    """
    data_loader = getattr(request.app.state, 'data_loader', None)
    
    if data_loader is None:
        raise HTTPException(
            status_code=400,
            detail="Данные не загружены. Сначала загрузите данные через /upload/json или /upload/file"
        )
    
    if domain1 not in data_loader._domains:
        raise HTTPException(status_code=404, detail=f"Домен '{domain1}' не найден")
    if domain2 not in data_loader._domains:
        raise HTTPException(status_code=404, detail=f"Домен '{domain2}' не найден")
    
    try:
        enriched_service = _get_enriched_service(request)
        similarity_service = SimilarityService(metric=metric)
        
        domain1_obj = data_loader._domains[domain1]
        domain2_obj = data_loader._domains[domain2]
        
        terms1 = [term.name for term in domain1_obj.terms]
        terms2 = [term.name for term in domain2_obj.terms]
        
        # Получаем обогащённые эмбеддинги для всех терминов
        embeddings1 = []
        embeddings2 = []
        enriched_count = 0
        
        for term in terms1:
            emb = enriched_service.get_enriched_embedding(term, alpha)
            embeddings1.append(emb)
            info = enriched_service.get_enrichment_info(term)
            if info["enriched"]:
                enriched_count += 1
        
        for term in terms2:
            emb = enriched_service.get_enriched_embedding(term, alpha)
            embeddings2.append(emb)
            info = enriched_service.get_enrichment_info(term)
            if info["enriched"]:
                enriched_count += 1
        
        # Центроиды
        import numpy as np
        centroid1 = np.mean(embeddings1, axis=0)
        centroid2 = np.mean(embeddings2, axis=0)
        
        # Нормализуем центроиды
        norm1 = np.linalg.norm(centroid1)
        norm2 = np.linalg.norm(centroid2)
        if norm1 > 0:
            centroid1 = centroid1 / norm1
        if norm2 > 0:
            centroid2 = centroid2 / norm2
        
        score = similarity_service.calculate_similarity(centroid1, centroid2)
        
        return EnrichedSimilarityResponse(
            domain1=domain1,
            domain2=domain2,
            score=score,
            metric=metric,
            alpha=alpha,
            terms_enriched=enriched_count,
            terms_total=len(terms1) + len(terms2),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/enriched/graph",
    response_model=GraphResponse,
    tags=["Enrichment"],
    summary="Граф с обогащёнными эмбеддингами",
    responses={
        200: {
            "description": "Граф с обогащёнными эмбеддингами",
            "model": GraphResponse,
        }
    },
)
async def get_enriched_graph(
    request: Request,
    threshold: float = Query(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Порог близости",
    ),
    alpha: float = Query(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Вес оригинального термина",
    ),
    metric: str = Query(
        default="cosine",
        enum=["cosine", "euclidean"],
        description="Метрика близости",
    ),
):
    """Получение графа близости доменов с обогащёнными эмбеддингами.
    
    Строит граф связей между доменами, используя эмбеддинги
    с гиперонимами из RuWordNet.
    
    Args:
        request: HTTP request.
        threshold: Порог близости для отображения рёбер.
        alpha: Вес оригинального термина.
        metric: Метрика близости.
    
    Returns:
        GraphResponse: Граф с узлами доменов и рёбрами близости.
    
    Example:
        ```bash
        curl "http://localhost:8000/api/v1/enriched/graph?threshold=0.6&alpha=0.7"
        ```
    """
    data_loader = getattr(request.app.state, 'data_loader', None)
    
    if data_loader is None or not data_loader.domain_names:
        return GraphResponse(nodes=[], edges=[], threshold=threshold)
    
    import numpy as np
    enriched_service = _get_enriched_service(request)
    similarity_service = SimilarityService(metric=metric)
    
    centroids = {}
    
    for domain_name in data_loader.domain_names:
        domain_obj = data_loader._domains[domain_name]
        terms = [term.name for term in domain_obj.terms]
        
        embeddings = []
        for term in terms:
            emb = enriched_service.get_enriched_embedding(term, alpha)
            embeddings.append(emb)
        
        centroid = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        
        centroids[domain_name] = centroid
    
    # Строим матрицу близости
    domains_list = list(data_loader.domain_names)
    edges = []
    
    for i, d1 in enumerate(domains_list):
        for j, d2 in enumerate(domains_list):
            if i < j:
                score = similarity_service.calculate_similarity(
                    centroids[d1], centroids[d2]
                )
                if score >= threshold:
                    edges.append({
                        "source": d1,
                        "target": d2,
                        "weight": float(score)
                    })
    
    return GraphResponse(
        nodes=[{"id": d, "label": d} for d in domains_list],
        edges=edges,
        threshold=threshold,
    )


@router.get(
    "/graph/detailed",
    response_model=GraphResponseDetailed,
    tags=["Graph"],
    summary="Детальный граф с терминами",
    responses={
        200: {
            "description": "Граф с терминами-центроидами",
            "model": GraphResponseDetailed,
        },
        409: {
            "description": "Генерация графа уже выполняется",
        },
    },
)
async def get_graph_detailed(
        request: Request,
        threshold: float = Query(
            default=0.5,
            ge=0.0,
            le=1.0,
            description="Порог близости",
        ),
        metric: str = Query(
            default="cosine",
            enum=["cosine", "euclidean"],
            description="Метрика близости",
        ),
        show_terms: bool = Query(
            default=True,
            description="Показывать термины-центроиды",
        ),
        method: str = Query(
            default="sbert",
            enum=["sbert", "rag", "wordnet", "bert", "tfidf", "ensemble"],
            description="Метод расчёта близости: sbert (центроиды), rag (RAG-Centroids), wordnet, bert, tfidf (char-ngram), ensemble (SBERT+TF-IDF)",
        ),
        force_recalculate: bool = Query(
            default=False,
            description="Принудительно пересчитать результаты, игнорируя кеш",
        ),
        sbert_weight: float = Query(
            default=None,
            description="Вес SBERT для ensemble метода (если не передан — из weights.json)",
        ),
        tfidf_weight: float = Query(
            default=None,
            description="Вес TF-IDF для ensemble метода (если не передан — из weights.json)",
        ),
    ):
    """Получение детального графа с терминами-центроидами.

    Возвращает граф, где:
    - Домены отображаются как большие узлы
    - Термины отображаются как маленькие узлы, связанные с родительским доменом
    - Стрелки от терминов к доменам показывают принадлежность
    - Рёбра между доменами показывают близость (similarity)

    Поддерживает различные методы расчёта близости:
    - **sbert**: SBERT эмбеддинги + центроиды (по умолчанию)
    - **rag**: RAG-Centroids подход (kNN по корпусу)
    - **wordnet**: Лексическая близость через RuWordNet
    - **bert**: BERT эмбеддинги (bert-base-multilingual)
    - **tfidf**: TF-IDF char-ngram центроиды
    - **ensemble**: Комбинация SBERT + TF-IDF центроидов

    Args:
        request: HTTP request.
        threshold: Минимальная близость для отображения ребра.
        metric: Метрика расчёта близости.
        show_terms: Включать ли узлы терминов.
        method: Метод расчёта близости доменов.

    Returns:
        GraphResponseDetailed: Детальный граф с терминами.

    Example:
        ```bash
        curl "http://localhost:8000/api/v1/graph/detailed?threshold=0.6&method=sbert"
        curl "http://localhost:8000/api/v1/graph/detailed?threshold=0.6&method=rag"
        curl "http://localhost:8000/api/v1/graph/detailed?threshold=0.6&method=tfidf"
        curl "http://localhost:8000/api/v1/graph/detailed?threshold=0.6&method=ensemble"
        ```
    """
    cache: CacheManager = request.app.state.cache
    
    # Пытаемся получить блокировку
    if not await cache.acquire_lock("graph"):
        raise HTTPException(
            status_code=409,
            detail="Генерация графа уже выполняется. Подождите завершения текущей операции."
        )
    
    try:
        # Устанавливаем статус
        await cache.set_operation_status("graph", "started")
        
        data_loader = getattr(request.app.state, 'data_loader', None)
        
        if data_loader is None or not data_loader.domain_names:
            return GraphResponseDetailed(nodes=[], edges=[], threshold=threshold)
        
        import numpy as np
        
        # Используем предзагруженный сервис
        embedding_service = getattr(request.app.state, 'embedding_service', None)
        if embedding_service is None:
            embedding_service = EmbeddingService(cache=cache)
        
        centroid_service = CentroidService()
        similarity_service = SimilarityService(metric=metric)
        
        await cache.set_operation_status("graph", "processing", 0.3)
        
        # Собираем все данные
        nodes = []
        edges = []
        domains_list = list(data_loader.domain_names)
        centroids = {}
        
        # Создаём узлы доменов
        for domain_name in domains_list:
            nodes.append(GraphNodeDetailed(
                id=domain_name,
                label=domain_name,
                type="domain",
                parent=None
            ))
        
        # RAG: вычисляем центроиды ОДИН раз ДО внешнего цикла (исправление дублирования)
        if method == "rag":
            try:
                from ..infrastructure.retrieval_service import RetrievalService
                
                # Параметры RAG (как в benchmark)
                k_rag = 5
                alpha_rag = 0.5  # Унифицировано: alpha=0.5 для баланса оригинального эмбеддинга и RAG-контекста (configs.py)
                
                # Кеш для RetrievalService (один раз на все домены)
                rag_cache_key = "rag_graph_retrieval_service"
                if not hasattr(request.app.state, '_rag_retrieval_cache'):
                    request.app.state._rag_retrieval_cache = {}
                
                # Получаем все термины для построения индекса
                all_terms_for_index = []
                for domain_name in domains_list:
                    domain_obj = data_loader._domains[domain_name]
                    all_terms_for_index.extend([term.name for term in domain_obj.terms])
                
                all_terms_for_index = list(set(all_terms_for_index))  # Убираем дубликаты
                
                logger.info(f"[RAG] Используем RetrievalService с {len(all_terms_for_index)} терминами")
                
                if rag_cache_key not in request.app.state._rag_retrieval_cache:
                    # Создаём RetrievalService и строим индекс
                    retrieval_service = RetrievalService(embedding_service)
                    retrieval_service.build_index(all_terms_for_index, k=k_rag)
                    request.app.state._rag_retrieval_cache[rag_cache_key] = retrieval_service
                    logger.info(f"[RAG] RetrievalService создан и закеширован")
                else:
                    retrieval_service = request.app.state._rag_retrieval_cache[rag_cache_key]
                    logger.info(f"[RAG] Используем кешированный RetrievalService")
                
                # Batch: получаем эмбеддинги всех терминов одним вызовом
                all_terms = list(set(all_terms_for_index))
                term_embs = embedding_service.get_embeddings_batch(all_terms)
                term_to_idx = {term: i for i, term in enumerate(all_terms)}

                # Batch retrieval для всех терминов сразу (один FAISS search!)
                all_retrievals = retrieval_service.get_retrieved_context_batch(all_terms, k_rag)

                # Batch: получаем эмбеддинги всех retrieved соседей одним вызовом
                all_retrieved_terms = set()
                for retrieved_list in all_retrievals.values():
                    all_retrieved_terms.update(retrieved_list)
                
                if all_retrieved_terms:
                    retrieved_embs = embedding_service.get_embeddings_batch(list(all_retrieved_terms))
                    retrieved_term_to_idx = {term: i for i, term in enumerate(all_retrieved_terms)}
                else:
                    retrieved_embs = np.array([])
                    retrieved_term_to_idx = {}

                # Вычисляем RAG центроид для каждого домена
                for domain_name in domains_list:
                    domain_obj = data_loader._domains[domain_name]
                    domain_terms = [term.name for term in domain_obj.terms]

                    if not domain_terms:
                        continue

                    rag_embs = []
                    for term in domain_terms:
                        idx = term_to_idx[term]
                        emb = term_embs[idx]

                        retrieved = all_retrievals.get(term, [])
                        if retrieved:
                            # Alpha blending с центроидом retrieved соседей
                            retrieved_centroid = np.mean([
                                retrieved_embs[retrieved_term_to_idx[t]] 
                                for t in retrieved if t in retrieved_term_to_idx
                            ], axis=0)
                            rag_emb = alpha_rag * emb + (1 - alpha_rag) * retrieved_centroid
                        else:
                            rag_emb = emb

                        rag_embs.append(rag_emb)

                    if rag_embs:
                        centroid = np.mean(rag_embs, axis=0)
                        norm = np.linalg.norm(centroid)
                        if norm > 0:
                            centroid = centroid / norm
                    else:
                        embeddings = embedding_service.get_embeddings_batch(domain_terms)
                        centroid = centroid_service.calculate_centroid(embeddings)

                    centroids[domain_name] = centroid
                    
                logger.info(f"[RAG] RAG центроиды вычислены для {len(centroids)} доменов")
                
            except Exception as e:
                # Fallback при любой ошибке RAG
                logger.warning(f"[graph] RAG ошибка: {e}, используем SBERT fallback")
                # Fallback: вычисляем SBERT центроиды для всех доменов
                for domain_name in domains_list:
                    domain_obj = data_loader._domains[domain_name]
                    terms = [term.name for term in domain_obj.terms]
                    embeddings = embedding_service.get_embeddings_batch(terms)
                    centroids[domain_name] = centroid_service.calculate_centroid(embeddings)
        
        # Цикл по доменам для non-RAG методов (SBERT, WordNet, BERT)
        for domain_name in domains_list:
            # Пропускаем RAG — центроиды уже вычислены выше
            if method == "rag":
                continue
            
            domain_obj = data_loader._domains[domain_name]
            terms = [term.name for term in domain_obj.terms]
            
            try:
                if method == "sbert":
                    embeddings = embedding_service.get_embeddings_batch(terms)
                    centroid = centroid_service.calculate_centroid(embeddings)
                elif method == "wordnet":
                    # WordNet: центроид на основе гиперонимов
                    try:
                        wordnet_service = _get_wordnet_service(request)
                        centroid = wordnet_service.get_term_centroid(terms)
                    except HTTPException:
                        # Fallback если WordNet не инициализирован
                        logger.warning(f"[graph] WordNet не доступен для домена {domain_name}, используем SBERT fallback")
                        embeddings = embedding_service.get_embeddings_batch(terms)
                        centroid = centroid_service.calculate_centroid(embeddings)
                elif method == "bert":
                    # BERT: используем BERT эмбеддинги (singleton)
                    try:
                        from ..infrastructure.bert_embedding_service import get_bert_embedding_service
                        bert_service = get_bert_embedding_service()
                        embeddings = bert_service.get_embeddings_batch(terms)
                        if embeddings and isinstance(embeddings[0], list):
                            embeddings = [np.array(e) for e in embeddings]
                        centroid = centroid_service.calculate_centroid(embeddings)
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                            # CUDA OOM - fallback на SBERT
                            logger.warning(f"[graph] BERT OOM для {domain_name}, используем SBERT fallback")
                            embeddings = embedding_service.get_embeddings_batch(terms)
                            centroid = centroid_service.calculate_centroid(embeddings)
                        else:
                            raise
                elif method == "tfidf":
                    # TF-IDF: используем char-ngram центроиды
                    logger.info(f"[graph/tfidf] Вычисляю TF-IDF центроид для домена '{domain_name}' ({len(terms)} терминов)")
                    tfidf_service = TfidfService()
                    centroid = calculate_tfidf_centroid(terms, tfidf_service)
                    logger.info(f"[graph/tfidf] TF-IDF центроид вычислен для '{domain_name}'")
                elif method == "ensemble":
                    # Ensemble: используем только SBERT центроиды для similarity
                    # TF-IDF будет вычислен отдельно в блоке расчёта similarity
                    logger.info(f"[graph/ensemble] Вычисляю SBERT центроид для домена '{domain_name}' ({len(terms)} терминов)")
                    embeddings = embedding_service.get_embeddings_batch(terms)
                    centroid = centroid_service.calculate_centroid(embeddings)
                    logger.info(f"[graph/ensemble] SBERT центроид вычислен для '{domain_name}'")
                else:
                    # По умолчанию SBERT
                    embeddings = embedding_service.get_embeddings_batch(terms)
                    centroid = centroid_service.calculate_centroid(embeddings)
            except Exception as e:
                # Fallback на SBERT при любой ошибке
                logger.warning(f"[graph] Ошибка при вычислении центроида для {domain_name} методом {method}: {e}")
                embeddings = embedding_service.get_embeddings_batch(terms)
                centroid = centroid_service.calculate_centroid(embeddings)
            
            centroids[domain_name] = centroid
            
        # Создаём узлы терминов
        if show_terms:
            for domain_name in domains_list:
                domain_obj = data_loader._domains[domain_name]
                terms = [term.name for term in domain_obj.terms]
                
                # Узлы для каждого термина
                for i, term_name in enumerate(terms):
                    term_id = f"{domain_name}_term_{i}"
                    nodes.append(GraphNodeDetailed(
                        id=term_id,
                        label=term_name,
                        type="term",
                        parent=domain_name
                    ))
                    
                    # Ребро принадлежности (от термина к домену)
                    edges.append({
                        "source": term_id,
                        "target": domain_name,
                        "weight": 1.0,
                        "type": "belongs_to"
                    })
        else:
            # Только центроиды без терминов
            for domain_name in domains_list:
                domain_obj = data_loader._domains[domain_name]
                terms = [term.name for term in domain_obj.terms]
                embeddings = embedding_service.get_embeddings_batch(terms)
                centroid = centroid_service.calculate_centroid(embeddings)
                centroids[domain_name] = centroid
        
        # Вычисляем близость между доменами (с кешированием)
        logger.info(f"[graph] Начинаю расчёт близости для {len(domains_list)} доменов методом '{method}'")
        
        for i, d1 in enumerate(domains_list):
            for j, d2 in enumerate(domains_list):
                if i < j:
                    # Проверяем кеш (если не force_recalculate)
                    cached_score = None
                    if not force_recalculate:
                        cached_score = await cache.get_similarity_by_method(
                            d1, d2, method, metric
                        )
                    
                    if cached_score is not None:
                        score = cached_score
                        logger.debug(f"[graph] Использую кеш: {d1}-{d2} = {score:.4f}")
                    else:
                        # Вычисляем в зависимости от метода
                        domain1_obj = data_loader._domains[d1]
                        domain2_obj = data_loader._domains[d2]
                        terms1 = [term.name for term in domain1_obj.terms]
                        terms2 = [term.name for term in domain2_obj.terms]
                        
                        logger.debug(f"[graph/{method}] Расчёт: {d1} ({len(terms1)} терминов) vs {d2} ({len(terms2)} терминов)")
                        
                        if method == "sbert":
                            score = similarity_service.calculate_similarity(
                                centroids[d1], centroids[d2]
                            )
                        elif method == "tfidf":
                            # TF-IDF: создаём единый сервис для всех пар (это уже делается в calculate_tfidf_similarity)
                            tfidf_service = TfidfService()
                            score = calculate_tfidf_similarity(terms1, terms2, tfidf_service)
                        elif method == "ensemble":
                            # Ensemble: используем единый TF-IDF сервис для всех пар
                            # Если его ещё нет - создаём и обучаем на всех терминах
                            if not hasattr(request.app.state, '_ensemble_tfidf_service'):
                                # Собираем все термины всех доменов
                                all_domain_terms = []
                                for dn in domains_list:
                                    domain_obj = data_loader._domains[dn]
                                    all_domain_terms.extend([t.name for t in domain_obj.terms])
                                # Создаём и обучаем TF-IDF на всех терминах
                                request.app.state._ensemble_tfidf_service = TfidfService()
                                request.app.state._ensemble_tfidf_service.fit_terms(all_domain_terms)
                                logger.info(f"[graph/ensemble] TF-IDF обучен на {len(all_domain_terms)} терминах")
                            
                            # Используем веса из параметров или дефолтные
                            ensemble_weights = {
                                "sbert": sbert_weight if sbert_weight is not None else 0.7,
                                "tfidf": tfidf_weight if tfidf_weight is not None else 0.3,
                            }
                            logger.info(f"[graph/ensemble] Используем веса: sbert={ensemble_weights['sbert']}, tfidf={ensemble_weights['tfidf']}")
                            ensemble_result = calculate_ensemble_similarity(
                                (terms1, centroids[d1]),
                                (terms2, centroids[d2]),
                                embedding_service,
                                request.app.state._ensemble_tfidf_service,
                                weights=ensemble_weights
                            )
                            score = ensemble_result["similarity"]
                        else:
                            # Для остальных методов (wordnet, bert, rag) - SBERT similarity
                            score = similarity_service.calculate_similarity(
                                centroids[d1], centroids[d2]
                            )
                        
                        logger.debug(f"[graph/{method}] Результат: {d1}-{d2} = {score:.4f}")
                        
                        # Сохраняем в кеш (Redis + Disk)
                        await cache.set_similarity_by_method(
                            d1, d2, score, method, metric
                        )
                    
                    if score >= threshold:
                        edges.append({
                            "source": d1,
                            "target": d2,
                            "weight": float(score),
                            "type": "similarity"
                        })
        
        await cache.set_operation_status("graph", "completed", 1.0)
        
        return GraphResponseDetailed(
            nodes=nodes,
            edges=edges,
            threshold=threshold,
        )
    except HTTPException:
        raise
    except Exception as e:
        await cache.set_operation_status("graph", "failed")
        raise HTTPException(status_code=500, detail=f"Ошибка генерации графа: {str(e)}")
    finally:
        # Освобождаем блокировку
        await cache.release_lock("graph")


# === Эндпоинты для асинхронных задач (Celery) ===

@router.post(
    "/tasks/benchmark",
    status_code=202,
    tags=["Tasks"],
    summary="Создание асинхронной задачи бенчмарка",
    responses={
        202: {
            "description": "Задача создана, используйте /tasks/{task_id}/status для проверки",
        },
    },
)
async def create_benchmark_task(
    body: BenchmarkRequest,
    request: Request,
) -> dict:
    """Создание асинхронной задачи бенчмарка.
    
    Запускает бенчмарк в фоновом режиме через Celery.
    Возвращает task_id для отслеживания статуса.
    
    Args:
        body: BenchmarkRequest с названием датасета.
        request: HTTP request.
    
    Returns:
        dict: ID задачи и начальный статус.
    
    Example:
        ```bash
        curl -X POST http://localhost:8000/api/v1/tasks/benchmark \
          -H "Content-Type: application/json" \
          -d '{"dataset": "hj-rg"}'
        ```
        
        ```json
        {
            "task_id": "abc123...",
            "status": "PENDING",
            "message": "Benchmark task created. Use GET /tasks/{task_id}/status to check progress."
        }
        ```
    """
    from ..application.tasks import celery_app, run_benchmark
    
    # Инициализируем Celery с параметрами Redis
    redis_host = os.getenv("REDIS_HOST", "redis")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_password = os.getenv("REDIS_PASSWORD", "changeme")
    
    celery_app.conf.broker_url = f"redis://:{redis_password}@{redis_host}:{redis_port}/0"
    celery_app.conf.result_backend = f"redis://:{redis_password}@{redis_host}:{redis_port}/1"
    
    # Запускаем задачу
    task = run_benchmark.delay(body.dataset, body.methods)
    
    return {
        "task_id": task.id,
        "status": "PENDING",
        "message": "Benchmark task created. Use GET /tasks/{task_id}/status to check progress.",
    }


@router.get(
    "/tasks/{task_id}/status",
    tags=["Tasks"],
    summary="Получение статуса задачи",
    responses={
        200: {
            "description": "Текущий статус задачи",
        },
    },
)
async def get_task_status(task_id: str) -> dict:
    """Получение статуса асинхронной задачи.
    
    Возвращает текущее состояние задачи Celery:
    - PENDING: Задача ожидает выполнения
    - PROGRESS: Задача выполняется
    - SUCCESS: Задача завершена успешно
    - FAILURE: Задача завершена с ошибкой
    
    Args:
        task_id: ID задачи из Celery.
    
    Returns:
        dict: Статус задачи с прогрессом и результатом.
    
    Example:
        ```bash
        curl http://localhost:8000/api/v1/tasks/abc123.../status
        ```
    """
    from ..application.tasks import celery_app
    
    # Инициализируем Celery
    redis_host = os.getenv("REDIS_HOST", "redis")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_password = os.getenv("REDIS_PASSWORD", "changeme")
    celery_app.conf.broker_url = f"redis://:{redis_password}@{redis_host}:{redis_port}/0"
    celery_app.conf.result_backend = f"redis://:{redis_password}@{redis_host}:{redis_port}/1"
    
    task = celery_app.AsyncResult(task_id)
    
    response = {
        "task_id": task_id,
        "status": task.state,
    }
    
    if task.state == "PENDING":
        response["message"] = "Task is waiting to be processed"
    elif task.state == "PROGRESS":
        response["progress"] = task.info.get("progress", 0)
        response["status_message"] = task.info.get("status", "")
    elif task.state == "SUCCESS":
        response["result"] = task.result
        response["message"] = "Task completed successfully"
    elif task.state == "FAILURE":
        response["error"] = str(task.info) if task.info else "Unknown error"
        response["message"] = "Task failed"
    else:
        response["message"] = f"Task in state: {task.state}"
    
    return response


@router.get(
    "/tasks/{task_id}/result",
    tags=["Tasks"],
    summary="Получение результата задачи",
    responses={
        200: {
            "description": "Результат задачи",
        },
    },
)
async def get_task_result(task_id: str) -> dict:
    """Получение результата завершённой задачи.
    
    Если задача завершена успешно — возвращает результат.
    Если задача ещё выполняется — возвращает статус PROGRESS.
    Результаты бенчмарков автоматически сохраняются в SQLite.
    
    Args:
        task_id: ID задачи из Celery.
    
    Returns:
        dict: Результат задачи или её статус.
    
    Example:
        ```bash
        curl http://localhost:8000/api/v1/tasks/abc123.../result
        ```
    """
    from ..application.tasks import celery_app
    from ..infrastructure.results_storage import get_storage
    
    # Инициализируем Celery
    redis_host = os.getenv("REDIS_HOST", "redis")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_password = os.getenv("REDIS_PASSWORD", "changeme")
    celery_app.conf.broker_url = f"redis://:{redis_password}@{redis_host}:{redis_port}/0"
    celery_app.conf.result_backend = f"redis://:{redis_password}@{redis_host}:{redis_port}/1"
    
    task = celery_app.AsyncResult(task_id)
    
    if task.state == "SUCCESS":
        result = task.result
        
        # Если это результат бенчмарка — пробуем сохранить в SQLite
        if isinstance(result, dict) and result.get("success") and result.get("dataset_name"):
            storage = get_storage()
            storage.save_result(
                dataset=result["dataset_name"],
                results=result.get("results", []),
                execution_time_sec=result.get("execution_time_sec", 0.0),
            )
        
        return {
            "task_id": task_id,
            "status": "SUCCESS",
            "result": result,
        }
    
    elif task.state == "PROGRESS":
        return {
            "task_id": task_id,
            "status": "PROGRESS",
            "progress": task.info.get("progress", 0) if task.info else 0,
            "status_message": task.info.get("status", "") if task.info else "",
        }
    
    elif task.state == "FAILURE":
        return {
            "task_id": task_id,
            "status": "FAILURE",
            "error": str(task.info) if task.info else "Unknown error",
        }
    
    else:  # PENDING or RETRY
        return {
            "task_id": task_id,
            "status": task.state,
            "message": f"Task in state: {task.state}",
        }


@router.post(
    "/tasks/graph",
    status_code=202,
    tags=["Tasks"],
    summary="Создание асинхронной задачи построения графа",
    responses={
        202: {
            "description": "Задача создана",
        },
    },
)
async def create_graph_task(
    request: Request,
    threshold: float = Query(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Порог близости",
    ),
    show_terms: bool = Query(
        default=False,
        description="Показывать термины",
    ),
) -> dict:
    """Создание асинхронной задачи построения графа.
    
    Запускает построение детального графа в фоновом режиме.
    Используйте /tasks/{task_id}/status для проверки прогресса.
    
    Args:
        request: HTTP request.
        threshold: Порог близости для рёбер.
        show_terms: Включать ли узлы терминов.
    
    Returns:
        dict: ID задачи и начальный статус.
    
    Example:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/tasks/graph?threshold=0.6&show_terms=true"
        ```
    """
    from ..application.tasks import celery_app, build_graph
    
    # Инициализируем Celery
    redis_host = os.getenv("REDIS_HOST", "redis")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_password = os.getenv("REDIS_PASSWORD", "changeme")
    celery_app.conf.broker_url = f"redis://:{redis_password}@{redis_host}:{redis_port}/0"
    celery_app.conf.result_backend = f"redis://:{redis_password}@{redis_host}:{redis_port}/1"
    
    # Запускаем задачу
    task = build_graph.delay(threshold=threshold, show_terms=show_terms)
    
    return {
        "task_id": task.id,
        "status": "PENDING",
        "message": "Graph build task created. Use GET /tasks/{task_id}/status to check progress.",
    }




@router.get(
    "/similarity/ensemble/{domain1}/{domain2}",
    tags=["Similarity"],
    summary="Ансамблевая близость доменов (SBERT + TF-IDF)",
    responses={
        200: {
            "description": "Ансамблевая близость с индивидуальными оценками",
        },
        400: {
            "description": "Данные не загружены",
        },
        404: {
            "description": "Домен не найден",
        },
    },
)
async def get_ensemble_similarity(
    request: Request,
    domain1: str,
    domain2: str,
    sbert_weight: float = Query(default=None, description="Вес SBERT (если не передан — из weights.json)"),
    tfidf_weight: float = Query(default=None, description="Вес TF-IDF (если не передан — из weights.json)"),
    language: str = Query(default="ru", description="Язык: ru или en"),
) -> dict:
    """Расчёт ансамблевой близости между доменами (SBERT + TF-IDF).
    
    Вычисляет близость двух доменов комбинацией SBERT и TF-IDF методов.
    Если веса не переданы, загружаются из weights.json.
    
    Args:
        request: HTTP request.
        domain1: Первый домен.
        domain2: Второй домен.
        sbert_weight: Веса SBERT (0.0-1.0).
        tfidf_weight: Веса TF-IDF (0.0-1.0).
        language: Язык для загрузки дефолтных весов (ru/en).
    
    Returns:
        dict: {
            "similarity": ensemble_score,
            "sbert_score": float,
            "tfidf_score": float,
            "weights": {"sbert": float, "tfidf": float}
        }
    
    Example:
        ```bash
        curl "http://localhost:8000/api/v1/similarity/ensemble/ML/DL"
        curl "http://localhost:8000/api/v1/similarity/ensemble/ML/DL?sbert_weight=0.7&tfidf_weight=0.3"
        ```
    """
    data_loader = getattr(request.app.state, 'data_loader', None)
    
    if data_loader is None:
        raise HTTPException(
            status_code=400,
            detail="Данные не загружены. Сначала загрузите данные через /upload/json или /upload/file"
        )
    
    if domain1 not in data_loader._domains:
        raise HTTPException(status_code=404, detail=f"Домен '{domain1}' не найден")
    if domain2 not in data_loader._domains:
        raise HTTPException(status_code=404, detail=f"Домен '{domain2}' не найден")
    
    import numpy as np
    
    # Загрузка весов из weights.json
    weights = {"sbert": 0.8334, "tfidf": 0.1666}
    if sbert_weight is not None and tfidf_weight is not None:
        weights = {"sbert": sbert_weight, "tfidf": tfidf_weight}
    else:
        # Пытаемся загрузить из weights.json
        try:
            weights_path = Path(__file__).parent.parent.parent.parent / "weights.json"
            if weights_path.exists():
                with open(weights_path, 'r', encoding='utf-8') as f:
                    weights_data = json.load(f)
                lang_key = language if language in weights_data.get("languages", {}) else "ru"
                if lang_key in weights_data.get("languages", {}):
                    lang_weights = weights_data["languages"][lang_key]
                    weights = {
                        "sbert": lang_weights.get("sbert_weight", 0.8334),
                        "tfidf": lang_weights.get("tfidf_weight", 0.1666),
                    }
        except Exception:
            pass  # Используем дефолтные веса
    
    # Нормализуем веса (сумма = 1)
    total_w = weights["sbert"] + weights["tfidf"]
    if total_w > 0:
        weights["sbert"] /= total_w
        weights["tfidf"] /= total_w
    
    # Получаем сервисы
    embedding_service = getattr(request.app.state, 'embedding_service', None)
    if embedding_service is None:
        cache: CacheManager = request.app.state.cache
        embedding_service = EmbeddingService(cache=cache)
    
    tfidf_service = TfidfService()
    
    domain1_obj = data_loader._domains[domain1]
    domain2_obj = data_loader._domains[domain2]
    
    terms1 = [term.name for term in domain1_obj.terms]
    terms2 = [term.name for term in domain2_obj.terms]
    
    # --- SBERT similarity ---
    emb1_1 = embedding_service.get_embeddings_batch(terms1)
    emb2_1 = embedding_service.get_embeddings_batch(terms2)
    
    centroid1 = np.mean(emb1_1, axis=0)
    centroid2 = np.mean(emb2_1, axis=0)
    
    # Нормализация центроидов
    norm1 = np.linalg.norm(centroid1)
    norm2 = np.linalg.norm(centroid2)
    if norm1 > 0:
        centroid1 = centroid1 / norm1
    if norm2 > 0:
        centroid2 = centroid2 / norm2
    
    sbert_score = float(np.dot(centroid1, centroid2))
    
    # --- TF-IDF similarity ---
    tfidf_service.fit_terms(terms1 + terms2)
    
    # Центроиды TF-IDF
    tfidf_centroid1 = np.zeros(len(tfidf_service._ngram_vocab))
    tfidf_centroid2 = np.zeros(len(tfidf_service._ngram_vocab))
    
    for term in terms1:
        vec = tfidf_service.get_vector(term)
        if vec is not None:
            tfidf_centroid1 += vec
    if len(terms1) > 0:
        tfidf_centroid1 /= len(terms1)
    
    for term in terms2:
        vec = tfidf_service.get_vector(term)
        if vec is not None:
            tfidf_centroid2 += vec
    if len(terms2) > 0:
        tfidf_centroid2 /= len(terms2)
    
    # Cosine similarity для TF-IDF
    norm_c1 = np.linalg.norm(tfidf_centroid1)
    norm_c2 = np.linalg.norm(tfidf_centroid2)
    if norm_c1 > 0 and norm_c2 > 0:
        tfidf_score = float(np.dot(tfidf_centroid1, tfidf_centroid2) / (norm_c1 * norm_c2))
    else:
        tfidf_score = 0.0
    
    # --- Ensemble (простое взвешенное среднее) ---
    ensemble_score = (
        weights["sbert"] * sbert_score +
        weights["tfidf"] * tfidf_score
    )
    return {
        "similarity": float(ensemble_score),
        "sbert_score": round(sbert_score, 4),
        "tfidf_score": round(tfidf_score, 4),
        "weights": {
            "sbert": round(weights["sbert"], 4),
            "tfidf": round(weights["tfidf"], 4),
        }
    }

# ==== Wikipedia Similarity эндпоинт ====

@router.get(
    "/similarity/wikipedia",
    response_model=WikipediaSimilarityResponse,
    tags=["Similarity"],
    summary="Wikipedia domain similarity",
    responses={
        200: {
            "description": "Матрица близости доменов из Wikipedia zero-shot классификации",
            "model": WikipediaSimilarityResponse,
        }
    },
)
async def get_wikipedia_similarity():
    """Wikipedia domain similarity из zero-shot классификации.
    
    Возвращает матрицу семантической близости между доменами,
    вычисленную на основе zero-shot классификации Wikipedia статей.
    Данные загружаются из предрассчитанного CSV файла.
    
    Returns:
        WikipediaSimilarityResponse: Домены, матрица близости и топ-5 пар.
    
    Example:
        ```bash
        curl http://localhost:8000/api/v1/similarity/wikipedia
        ```
        
        ```json
        {
            "domains": ["ML", "DL", "Statistics"],
            "matrix": [[1.0, 0.85, 0.6], [0.85, 1.0, 0.65], [0.6, 0.65, 1.0]],
            "top_pairs": [
                {"d1": "ML", "d2": "DL", "score": 0.85}
            ]
        }
        ```
    """
    import pandas as pd
    
    # Путь к файлу similarity
    wiki_sim_path = f"{DATA_BASE}/wikipedia_similarity.csv"
    
    df = pd.read_csv(wiki_sim_path, index_col=0)
    domains = list(df.columns)
    matrix = df.values.tolist()
    
    # Top 5 пар
    pairs = []
    for i, d1 in enumerate(domains):
        for j, d2 in enumerate(domains):
            if i < j:
                pairs.append({"d1": d1, "d2": d2, "score": float(df.loc[d1, d2])})
    pairs.sort(key=lambda x: x["score"], reverse=True)
    
    return WikipediaSimilarityResponse(
        domains=domains,
        matrix=matrix,
        top_pairs=pairs[:5]
    )
