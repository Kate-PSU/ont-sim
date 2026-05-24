# backend/src/application/tasks.py
# Celery задачи для асинхронных операций
#
# Версия: 1.0
# Обновлено: 2026-04-09

"""
Модуль асинхронных задач Celery.

Тяжёлые операции (бенчмарк, построение графа) выполняются
в фоновом режиме, чтобы не блокировать API.
"""

import json
import logging
from typing import Any

from celery import Celery
from celery.exceptions import SoftTimeLimitExceeded

logger = logging.getLogger(__name__)

# Конфигурация Celery
celery_app = Celery(
    "diplom",
    broker=f"redis://{__name__}",  # Заменяется при инициализации
    backend=f"redis://{__name__}",  # Заменяется при инициализации
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Europe/Moscow",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,  # 10 минут max
    task_soft_time_limit=540,  # 9 минут soft limit
    result_expires=3600,  # Результаты хранятся 1 час
)


@celery_app.task(bind=True, name="benchmark.run")
def run_benchmark(self, dataset: str, methods: list[str] | None = None) -> dict[str, Any]:
    """Запуск бенчмарка в фоновом режиме.
    
    Args:
        dataset: Название датасета (hj, hj-rg, hj-mc, simlex).
        methods: Список методов для тестирования.
    
    Returns:
        Результаты бенчмарка.
    """
    from .benchmark_service import BenchmarkService
    
    logger.info(f"[Benchmark] Starting task {self.request.id} for dataset={dataset}")
    self.update_state(state="PROGRESS", meta={"progress": 0, "status": "Loading dataset"})
    
    try:
        service = BenchmarkService()
        
        # Выбираем путь к датасету
        import os
        from pathlib import Path
        
        data_dir = Path("/data")
        dataset_map = {
            "hj": data_dir / "hj.csv",
            "hj-rg": data_dir / "hj-rg.csv",
            "hj-mc": data_dir / "hj-mc.csv",
            "simlex": data_dir / "simlex.txt",
        }
        
        dataset_path = dataset_map.get(dataset)
        if dataset_path is None or not dataset_path.exists():
            return {
                "success": False,
                "error": f"Dataset not found: {dataset}",
            }
        
        self.update_state(state="PROGRESS", meta={"progress": 20, "status": "Running experiments"})
        
        # Запускаем бенчмарк
        comparison = service.run_all(dataset_path, dataset)
        
        self.update_state(state="PROGRESS", meta={"progress": 90, "status": "Formatting results"})
        
        # Форматируем результаты
        results_list = [
            {
                "method": r.method,
                "spearman": r.spearman,
                "pearson": r.pearson,
                "mse": r.mse,
                "missing": r.missing,
                "predictions_count": r.predictions_count,
            }
            for r in comparison.results
        ]
        
        # Сохраняем в SQLite для персистентного хранения
        from ..infrastructure.results_storage import save_benchmark_results
        save_benchmark_results(
            dataset=dataset,
            results=results_list,
            execution_time_sec=comparison.execution_time_sec,
        )
        
        self.update_state(state="PROGRESS", meta={"progress": 100, "status": "Completed"})
        
        results = {
            "success": True,
            "dataset_name": comparison.dataset_name,
            "dataset_size": comparison.dataset_size,
            "execution_time_sec": comparison.execution_time_sec,
            "results": results_list,
        }
        
        logger.info(f"[Benchmark] Completed task {self.request.id}")
        return results
        
    except SoftTimeLimitExceeded:
        logger.error(f"[Benchmark] Task {self.request.id} timed out")
        return {
            "success": False,
            "error": "Task timed out after 10 minutes",
        }
    except Exception as e:
        logger.error(f"[Benchmark] Task {self.request.id} failed: {e}")
        return {
            "success": False,
            "error": str(e),
        }


@celery_app.task(bind=True, name="graph.build")
def build_graph(
    self,
    threshold: float = 0.5,
    show_terms: bool = False,
) -> dict[str, Any]:
    """Построение графа близости в фоновом режиме.
    
    Args:
        threshold: Порог близости для рёбер.
        show_terms: Включать ли термины в граф.
    
    Returns:
        Данные графа (nodes, edges).
    """
    from ..infrastructure.data_loader import DataLoader
    
    logger.info(f"[Graph] Starting task {self.request.id}")
    
    try:
        # Загружаем данные
        loader = DataLoader("/data/terms.csv")
        data = loader.load()
        
        self.update_state(state="PROGRESS", meta={"progress": 30, "status": "Computing centroids"})
        
        # Вычисляем центроиды
        from .centroid_service import CentroidService
        from ..infrastructure.embedding_service import EmbeddingService
        
        embedding_service = EmbeddingService()
        centroid_service = CentroidService(embedding_service)
        centroids = centroid_service.compute_centroids(data.domains)
        
        self.update_state(state="PROGRESS", meta={"progress": 60, "status": "Building graph"})
        
        # Вычисляем близости
        from .similarity_service import SimilarityService
        similarity_service = SimilarityService(centroid_service)
        
        nodes = []
        edges = []
        
        # Добавляем узлы доменов
        for domain in data.domains:
            nodes.append({
                "id": domain.name,
                "label": domain.name,
                "type": "domain",
            })
            
            # Если нужно, добавляем термины
            if show_terms:
                for term in domain.terms:
                    nodes.append({
                        "id": f"{domain.name}:{term.name}",
                        "label": term.name,
                        "type": "term",
                        "parent": domain.name,
                    })
        
        self.update_state(state="PROGRESS", meta={"progress": 80, "status": "Computing edges"})
        
        # Добавляем рёбра
        domain_names = [d.name for d in data.domains]
        for i, d1 in enumerate(domain_names):
            for j, d2 in enumerate(domain_names):
                if i < j:
                    sim = similarity_service.compute_similarity(d1, d2, "cosine")
                    if sim >= threshold:
                        edges.append({
                            "source": d1,
                            "target": d2,
                            "weight": sim,
                            "type": "similarity",
                        })
        
        self.update_state(state="PROGRESS", meta={"progress": 100, "status": "Completed"})
        
        return {
            "success": True,
            "nodes": nodes,
            "edges": edges,
            "threshold": threshold,
            "node_count": len(nodes),
            "edge_count": len(edges),
        }
        
    except SoftTimeLimitExceeded:
        logger.error(f"[Graph] Task {self.request.id} timed out")
        return {
            "success": False,
            "error": "Task timed out",
        }
    except Exception as e:
        logger.error(f"[Graph] Task {self.request.id} failed: {e}")
        return {
            "success": False,
            "error": str(e),
        }


def init_celery(redis_host: str = "redis", redis_port: int = 6379) -> None:
    """Инициализация Celery с параметрами Redis.
    
    Args:
        redis_host: Хост Redis.
        redis_port: Порт Redis.
    """
    broker_url = f"redis://{redis_host}:{redis_port}/0"
    celery_app.conf.broker_url = broker_url
    celery_app.conf.result_backend = f"redis://{redis_host}:{redis_port}/1"
    logger.info(f"[Celery] Initialized with broker={broker_url}")
