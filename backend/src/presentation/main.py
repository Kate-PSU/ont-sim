# backend/src/presentation/main.py
# Главный файл FastAPI приложения
#
# Версия: 1.5
# Обновлено: 2026-04-10
# Изменения: добавлен settings.py для конфигурации из ENV

"""
Главный модуль FastAPI приложения.
Содержит эндпоинты API для работы с сервисом.
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .routes import router
from .routes_state import router as state_router
from ..infrastructure import CacheManager, CSVDataLoader, EmbeddingService
from ..infrastructure.settings import get_settings

# Настройка логирования - вывод в stderr для Docker
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)

logger.info("[main] === МОДУЛЬ ЗАГРУЖЕН ===")

# Глобальный загрузчик данных
data_loader = CSVDataLoader()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения.
    
    При старте:
    - Инициализация подключения к Redis
    - Загрузка данных
    - Предварительная загрузка модели SBERT
    
    При остановке:
    - Закрытие соединений
    """
    # Получаем настройки
    settings = get_settings()
    
    # Страт приложения
    redis_password = os.getenv("REDIS_PASSWORD", None)
    data_base = os.getenv("DATA_BASE", "/app/data")
    disk_cache_dir = os.getenv("DISK_CACHE_DIR", f"{data_base}/similarity")
    cache = CacheManager(
        host=settings.REDIS_HOST, 
        port=settings.REDIS_PORT, 
        password=redis_password,
        default_ttl=settings.CACHE_TTL,
        disk_cache_dir=disk_cache_dir,
    )
    await cache.connect()
    app.state.cache = cache
    
    # Warm-up: загружаем данные из disk cache в Redis
    logger.info("[startup] Загрузка similarity из disk cache в Redis...")
    try:
        loaded = await cache.warm_up()
        logger.info(f"[startup] Загружено {loaded} записей similarity из disk cache")
    except Exception as e:
        logger.warning(f"[startup] Warm-up disk cache не выполнен: {e}")
    
    # Загрузка данных из CSV
    try:
        terms_count = data_loader.load_file(settings.DATASET_PATH)
        app.state.data_loader = data_loader
        logger.info(f"Загружено {terms_count} терминов из {settings.DATASET_PATH}")
    except Exception as e:
        logger.warning(f"Не удалось загрузить данные: {e}")
        app.state.data_loader = data_loader
    
    # Предварительная загрузка модели SBERT
    logger.info("[startup] === НАЧАЛО ЗАПУСКА ПРИЛОЖЕНИЯ ===")
    logger.info(f"[startup] Redis: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
    logger.info(f"[startup] Датасет: {settings.DATASET_PATH}")
    logger.info(f"[startup] Модель эмбеддингов: {settings.EMBEDDING_MODEL}")
    
    logger.info("[embedding] Начало загрузки модели SBERT...")
    start_time = time.time()
    try:
        embedding_service = EmbeddingService(
            model_name=settings.EMBEDDING_MODEL,
            cache=cache
        )
        embedding_service.preload()
        app.state.embedding_service = embedding_service
        duration = time.time() - start_time
        logger.info(f"[embedding] Модель SBERT готова (загрузка: {duration:.1f}с)")
    except Exception as e:
        logger.error(f"[embedding] Ошибка загрузки модели: {e}")
        # Не падаем, продолжим без модели
        app.state.embedding_service = None
    
    logger.info("[startup] === ПРИЛОЖЕНИЕ ГОТОВО ===")
    
    yield
    
    # Остановка приложения
    logger.info("Остановка приложения...")
    await cache.disconnect()


# Создание приложения
app = FastAPI(
    title="Сервис оценки близости предметных областей",
    description="API для расчёта семантической близости между доменами",
    version="1.2.0",
    lifespan=lifespan,
)

# Middleware для логирования запросов
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Логирование всех HTTP запросов с временем выполнения."""
    start_time = time.time()
    
    # Определяем тип операции для логирования
    path = request.url.path
    operation_type = "unknown"
    if "/benchmark" in path:
        operation_type = "benchmark"
    elif "/graph" in path:
        operation_type = "graph"
    elif "/upload" in path:
        operation_type = "upload"
    elif "/state" in path:
        operation_type = "state"
    
    logger.info(f"[{operation_type}] --> {request.method} {path}")
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    status_code = response.status_code
    
    # Логируем завершение
    log_level = "info" if status_code < 400 else "warning"
    log_func = getattr(logger, log_level)
    
    log_func(
        f"[{operation_type}] <-- {request.method} {path} "
        f"status={status_code} duration={duration:.2f}s"
    )
    
    return response


# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене ограничить
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключение роутеров
app.include_router(router, prefix="/api/v1")
app.include_router(state_router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса."""
    return {"status": "healthy", "version": "1.2.0"}


@app.get("/")
async def root():
    """Корневой эндпоинт."""
    return {
        "service": "Сервис оценки близости",
        "version": "1.2.0",
        "docs": "/docs",
        "state": "/api/v1/state",
    }
