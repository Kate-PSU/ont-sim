# backend/src/presentation/routes_state.py
# Эндпоинты для единой точки входа и SSE
#
# Версия: 1.2
# Обновлено: 2026-04-10
# Изменения: исправлены пути к датасетам для Docker (/app/data/)

"""
Эндпоинты:
- GET /api/v1/state — состояние системы
- GET /api/v1/tasks/stream — SSE для real-time прогресса
- POST /api/v1/benchmark/run-safe — безопасный запуск бенчмарка
- POST /api/v1/embedding/switch-model — переключение модели эмбеддингов
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from .schemas import MethodResultSchema
from .schemas import (
    DatasetStatus,
    ActiveTask,
    SystemStateResponse,
    BenchmarkRunRequest,
    BenchmarkRunResponse,
)
from ..infrastructure.cache_manager import CacheManager
from ..infrastructure.results_storage import get_storage
from ..infrastructure.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

router = APIRouter()

# Хранилище активных задач для SSE
active_tasks: dict[str, dict] = {}

# Базовый путь к данным (для Docker: /app/data, для локальной разработки: data)
DATA_BASE = os.getenv("DATA_BASE", "/app/data")


# ==== Утилиты ====

def get_all_dataset_names() -> list[str]:
    """Получение списка всех доступных датасетов."""
    return ["hj-rg", "simlex999", "simlex999_rus"]


def get_russian_datasets() -> list[str]:
    """Получение русских датасетов."""
    return ["hj-rg", "simlex999_rus"]


def get_english_datasets() -> list[str]:
    """Получение английских датасетов."""
    return ["simlex999"]


def get_methods_for_dataset(dataset: str) -> list[str]:
    """Получение методов для датасета."""
    # Русские методы для hj-rg, simlex999_rus
    russian_methods = [
        "sbert",
        "sbert_tfidf",
        "sbert_zscore",
        "wordnet_lin",
        "wordnet_wup",
        "hybrid",
        "bertopic",
        "doc2vec",
        "lda",
    ]
    # Английские методы для simlex999
    english_methods = [
        "sbert",
        "wordnet_lin",
        "wordnet_wup",
    ]
    
    if dataset in get_russian_datasets():
        return russian_methods
    return english_methods


def method_id_to_display(method_id: str) -> str:
    """Конвертация ID метода в отображаемое имя."""
    mapping = {
        "sbert": "SBERT (baseline)",
        "sbert_tfidf": "SBERT + TF-IDF",
        "sbert_zscore": "SBERT + Z-score",
        "wordnet_lin": "RuWordNet (Lin)",
        "wordnet_wup": "RuWordNet (Wu-Palmer)",
        "wordnet_lin_en": "English WordNet (Lin)",
        "wordnet_wup_en": "English WordNet (Wu-Palmer)",
        "hybrid": "Hybrid (SBERT + RuWordNet)",
        "bertopic": "BERTopic",
        "doc2vec": "Doc2Vec",
        "lda": "LDA",
    }
    return mapping.get(method_id, method_id)


# ==== Эндпоинты ====

@router.get("/state", response_model=SystemStateResponse)
async def get_system_state(request: Request):
    """Получение состояния системы - единая точка входа.
    
    Возвращает:
    - Статус системы (ready/busy/error)
    - Информацию о загруженных данных
    - Статусы всех датасетов с результатами
    - Список активных задач
    """
    cache: CacheManager = request.app.state.cache
    
    # Проверяем загруженность доменов
    data_loader = getattr(request.app.state, 'data_loader', None)
    domains_loaded = data_loader is not None and len(data_loader.domain_names) > 0
    domains_count = len(data_loader.domain_names) if domains_loaded else 0
    
    # Проверяем заблокированные операции
    locked_ops = []
    for op in ["benchmark", "graph", "upload"]:
        if await cache.is_locked(op):
            ttl = await cache.get_lock_ttl(op)
            locked_ops.append({"operation": op, "ttl": ttl})
    
    system_busy = len(locked_ops) > 0
    busy_reason = locked_ops[0]["operation"] if locked_ops else None
    
    # Получаем статусы датасетов из хранилища
    storage = get_storage()
    datasets_status: dict[str, DatasetStatus] = {}
    
    for dataset_name in get_all_dataset_names():
        stored = storage.get_result(dataset_name)
        
        if stored:
            # Преобразуем результаты в словарь
            results_dict = {}
            for r in stored.get("results", []):
                method_id = r.get("method", "")
                results_dict[method_id] = MethodResultSchema(
                    method=method_id_to_display(method_id),
                    spearman=r.get("spearman", 0.0),
                    pearson=r.get("pearson", 0.0),
                    mse=r.get("mse", 0.0),
                    missing=r.get("missing", 0),
                    predictions_count=r.get("predictions_count", 0),
                )
            
            datasets_status[dataset_name] = DatasetStatus(
                status="completed",
                progress=1.0,
                results=results_dict,
                saved_at=stored.get("updated_at") or stored.get("created_at"),
            )
        else:
            # Проверяем, выполняется ли сейчас
            task_id = f"{dataset_name}_current"
            if task_id in active_tasks:
                task = active_tasks[task_id]
                datasets_status[dataset_name] = DatasetStatus(
                    status="running",
                    progress=task.get("progress", 0.0),
                )
            else:
                datasets_status[dataset_name] = DatasetStatus(
                    status="pending",
                    progress=0.0,
                )
    
    # Формируем список активных задач
    active_tasks_list = []
    for task_id, task in active_tasks.items():
        if task.get("status") == "running":
            active_tasks_list.append(ActiveTask(
                id=task_id,
                dataset=task.get("dataset", ""),
                method=task.get("method", ""),
                progress=task.get("progress", 0.0),
                status=task.get("status", "running"),
            ))
    
    return SystemStateResponse(
        system_status="error" if system_busy and busy_reason else "ready" if not system_busy else "busy",
        busy_reason=busy_reason,
        domains_loaded=domains_loaded,
        domains_count=domains_count,
        datasets=datasets_status,
        active_tasks=active_tasks_list,
    )


@router.post("/embedding/switch-model")
async def switch_embedding_model(request: Request, model: str):
    """Переключение модели эмбеддингов.
    
    Используется для переключения между русской и английской моделями:
    - "ru": ai-forever/sbert_large_nlu_ru (по умолчанию)
    - "en": sentence-transformers/all-mpnet-base-v2 (английская)
    
    После переключения на английскую модель рекомендуется вызвать
    reload_default() для возврата к русской модели.
    
    Args:
        request: HTTP request.
        model: Идентификатор модели ("ru" или "en").
    
    Returns:
        dict: Результат переключения.
    """
    # Получаем embedding_service из состояния приложения
    emb_service: EmbeddingService = getattr(request.app.state, 'embedding_service', None)
    
    if emb_service is None:
        raise HTTPException(
            status_code=500,
            detail="Embedding service не инициализирован"
        )
    
    # Определяем модель по идентификатору
    if model == "en":
        model_name = EmbeddingService.ENGLISH_MODEL
    elif model == "ru":
        model_name = EmbeddingService.DEFAULT_MODEL
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Неизвестный идентификатор модели: {model}. Используйте 'ru' или 'en'"
        )
    
    # Выполняем переключение
    success = emb_service.switch_model(model_name)
    
    if success:
        logger.info(f"[embedding] Модель переключена на: {model_name}")
        return {
            "success": True,
            "model": model,
            "model_name": model_name,
            "message": f"Модель переключена на {model_name}"
        }
    else:
        raise HTTPException(
            status_code=500,
            detail=f"Не удалось переключить модель на {model_name}"
        )


@router.post("/embedding/reload-default")
async def reload_default_model(request: Request):
    """Перезагрузка модели по умолчанию (RU).
    
    Используется для возврата к русской модели после выполнения
    английских бенчмарков. Вызывается в фоне после завершения.
    
    Args:
        request: HTTP request.
    
    Returns:
        dict: Результат перезагрузки.
    """
    # Получаем embedding_service из состояния приложения
    emb_service: EmbeddingService = getattr(request.app.state, 'embedding_service', None)
    
    if emb_service is None:
        raise HTTPException(
            status_code=500,
            detail="Embedding service не инициализирован"
        )
    
    # Выполняем перезагрузку
    success = emb_service.reload_default()
    
    if success:
        logger.info(f"[embedding] Модель возвращена к умолчанию: {EmbeddingService.DEFAULT_MODEL}")
        return {
            "success": True,
            "model_name": EmbeddingService.DEFAULT_MODEL,
            "message": f"Модель возвращена к {EmbeddingService.DEFAULT_MODEL}"
        }
    else:
        raise HTTPException(
            status_code=500,
            detail="Не удалось перезагрузить модель"
        )


@router.post("/benchmark/run-safe", response_model=BenchmarkRunResponse)
async def run_benchmark_safe(request: Request, body: BenchmarkRunRequest):
    """Безопасный запуск бенчмарка.
    
    Проверяет:
    1. Не выполняется ли уже задача
    2. Есть ли уже результаты в кэше (возвращаем из кэша вместо пересчёта)
    
    Если система занята — возвращает информацию о текущей задаче.
    Если результаты уже есть — возвращаем их без пересчёта.
    """
    cache: CacheManager = request.app.state.cache
    
    # Проверяем блокировку
    if await cache.is_locked("benchmark"):
        return BenchmarkRunResponse(
            success=False,
            error="system_busy",
            active_task=ActiveTask(
                id="current",
                dataset="benchmark",
                method="unknown",
                progress=0.0,
                status="running",
            ),
        )
    
    # Проверяем кэш — если результаты уже есть, возвращаем их
    storage = get_storage()
    cached_result = storage.get_result(body.dataset)
    
    if cached_result and not body.force_recalculate:
        # Результаты уже есть — возвращаем успех без пересчёта
        logger.info(f"[run-safe] Результаты для {body.dataset} найдены в кэше")
        return BenchmarkRunResponse(
            success=True,
            task_id=None,  # Нет задачи — результаты из кэша
            error=None,
            active_task=None,
        )
    
    # Результатов нет или требуется пересчёт — запускаем задачу
    task_id = f"{body.dataset}_{body.method}_{int(time.time())}"
    
    # Запускаем в фоне
    asyncio.create_task(
        _run_benchmark_task(request.app.state, body.dataset, body.method, task_id)
    )
    
    return BenchmarkRunResponse(
        success=True,
        task_id=task_id,
    )


async def _run_benchmark_task(app_state, dataset: str, method: str, task_id: str):
    """Фоновая задача выполнения бенчмарка."""
    global active_tasks
    
    cache: CacheManager = app_state.cache
    
    # Регистрируем задачу
    active_tasks[task_id] = {
        "dataset": dataset,
        "method": method,
        "progress": 0.0,
        "status": "running",
    }
    
    try:
        # Получаем блокировку
        if not await cache.acquire_lock("benchmark"):
            active_tasks[task_id]["status"] = "failed"
            active_tasks[task_id]["error"] = "Не удалось получить блокировку"
            return
        
        # Определяем путь к датасету
        datasets_paths = {
            "hj-rg": f"{DATA_BASE}/hj-rg.csv",
            "simlex999": f"{DATA_BASE}/simlex999.csv",
            "simlex999_rus": f"{DATA_BASE}/simlex999_rus_without_dupl.csv",
        }
        
        dataset_path = Path(datasets_paths.get(dataset, ""))
        
        if not dataset_path.exists():
            active_tasks[task_id]["status"] = "failed"
            active_tasks[task_id]["error"] = f"Датасет не найден: {dataset_path}"
            return
        
        # Обновляем прогресс
        active_tasks[task_id]["progress"] = 0.1
        
        # Запускаем бенчмарк
        from ..application.benchmark_service import BenchmarkService
        
        benchmark_service = BenchmarkService()
        comparison = benchmark_service.run_all(
            dataset_path=str(dataset_path),
            dataset_name=dataset,
        )
        
        active_tasks[task_id]["progress"] = 0.9
        
        # Сохраняем результаты
        storage = get_storage()
        storage.save_result(
            dataset=dataset,
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
        
        active_tasks[task_id]["progress"] = 1.0
        active_tasks[task_id]["status"] = "completed"
        
    except Exception as e:
        active_tasks[task_id]["status"] = "failed"
        active_tasks[task_id]["error"] = str(e)
    
    finally:
        # Освобождаем блокировку
        await cache.release_lock("benchmark")
        
        # Удаляем задачу через некоторое время
        await asyncio.sleep(10)
        if task_id in active_tasks:
            del active_tasks[task_id]


@router.get("/tasks/stream")
async def tasks_stream(request: Request):
    """SSE поток для real-time обновлений задач.
    
    Возвращает события:
    - progress: обновление прогресса
    - completed: задача завершена
    - failed: задача не удалась
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        while True:
            # Проверяем активные задачи
            for task_id, task in list(active_tasks.items()):
                status = task.get("status", "running")
                
                if status == "running":
                    event_data = {
                        "event": "progress",
                        "task_id": task_id,
                        "dataset": task.get("dataset", ""),
                        "method": task.get("method", ""),
                        "progress": task.get("progress", 0.0),
                        "status": "processing",
                    }
                elif status == "completed":
                    event_data = {
                        "event": "completed",
                        "task_id": task_id,
                        "dataset": task.get("dataset", ""),
                        "method": task.get("method", ""),
                        "status": "completed",
                    }
                elif status == "failed":
                    event_data = {
                        "event": "failed",
                        "task_id": task_id,
                        "dataset": task.get("dataset", ""),
                        "method": task.get("method", ""),
                        "error": task.get("error", "Unknown error"),
                    }
                else:
                    continue
                
                yield f"data: {json.dumps(event_data)}\n\n"
            
            # Проверяем системный статус
            cache: CacheManager = request.app.state.cache
            system_status = await cache.get_system_status()
            
            if system_status.get("busy"):
                locked = system_status.get("locked_operations", [])
                if locked:
                    yield f"data: {json.dumps({'event': 'busy', 'operations': locked})}\n\n"
            
            await asyncio.sleep(1)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Получение статуса конкретной задачи."""
    if task_id in active_tasks:
        task = active_tasks[task_id]
        return {
            "task_id": task_id,
            "status": task.get("status", "unknown"),
            "dataset": task.get("dataset", ""),
            "method": task.get("method", ""),
            "progress": task.get("progress", 0.0),
            "error": task.get("error"),
        }
    
    # Проверяем в хранилище
    storage = get_storage()
    for dataset in get_all_dataset_names():
        stored = storage.get_result(dataset)
        if stored:
            return {
                "task_id": task_id,
                "status": "completed",
                "dataset": dataset,
                "results": stored.get("results", []),
            }
    
    raise HTTPException(status_code=404, detail=f"Задача {task_id} не найдена")
