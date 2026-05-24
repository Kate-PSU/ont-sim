# backend/src/infrastructure/disk_cache.py
# Persistent cache manager for similarity results
#
# Версия: 1.0
# Обновлено: 2026-04-18

"""
Persistent cache manager for storing similarity results on disk.

Хранит результаты similarity в JSON файлах для сохранения между сессиями.
При старте приложения данные загружаются в Redis для быстрого доступа.

Структура директории:
    data/similarity/
    ├── sbert/
    │   ├── ml-dl_cosine.json
    │   └── cs-economics_cosine.json
    ├── rag/
    ├── wordnet/
    └── bert/
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Базовый путь к данным
DATA_BASE = os.getenv("DATA_BASE", "/app/data")
SIMILARITY_DIR = f"{DATA_BASE}/similarity"


class DiskCacheManager:
    """Менеджер persistent кеша на диске.
    
    Сохраняет результаты similarity в JSON файлы.
    Каждый метод (sbert, rag, wordnet, bert) имеет отдельную директорию.
    
    Attributes:
        base_dir: Базовая директория для хранения файлов.
    """
    
    def __init__(
        self,
        base_dir: Optional[str] = None,
    ) -> None:
        """Инициализация DiskCacheManager.
        
        Args:
            base_dir: Базовая директория. По умолчанию DATA_BASE/similarity.
        """
        self.base_dir = Path(base_dir or SIMILARITY_DIR)
        self._ensure_dir_exists()
    
    def _ensure_dir_exists(self) -> None:
        """Создать базовую директорию и поддиректории для методов."""
        # Создаём базовую директорию
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Создаём поддиректории для каждого метода
        methods = ["sbert", "rag", "wordnet", "bert", "tfidf", "ensemble"]
        for method in methods:
            method_dir = self.base_dir / method
            method_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(
        self,
        domain1: str,
        domain2: str,
        method: str = "sbert",
        metric: str = "cosine",
    ) -> Path:
        """Получить путь к JSON файлу кеша.
        
        Args:
            domain1: Первый домен (lower).
            domain2: Второй домен (lower).
            method: Метод расчёта (sbert, rag, wordnet, bert).
            metric: Метрика близости (cosine, euclidean).
        
        Returns:
            Path: Путь к JSON файлу.
        """
        # Нормализуем порядок доменов (alphabetical) для консистентности
        d1, d2 = sorted([domain1.lower(), domain2.lower()])
        
        method_dir = self.base_dir / method.lower()
        filename = f"{d1}-{d2}_{metric}.json"
        return method_dir / filename
    
    def save_similarity(
        self,
        domain1: str,
        domain2: str,
        score: float,
        method: str = "sbert",
        metric: str = "cosine",
    ) -> None:
        """Сохранить результат similarity в JSON файл.
        
        Args:
            domain1: Первый домен.
            domain2: Второй домен.
            score: Значение близости.
            method: Метод расчёта (sbert, rag, wordnet, bert, tfidf, ensemble).
            metric: Метрика близости.
        """
        path = self._get_cache_path(domain1, domain2, method, metric)
        
        # Создаём директорию для кеша если её нет
        cache_dir = os.path.dirname(path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        
        data = {
            "domain1": domain1,
            "domain2": domain2,
            "method": method,
            "metric": metric,
            "score": score,
            "timestamp": datetime.now().isoformat(),
        }
        
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug(f"Сохранено в disk cache: {path}")
        except Exception as e:
            logger.error(f"Ошибка сохранения в disk cache: {e}")
            raise
    
    def load_similarity(
        self,
        domain1: str,
        domain2: str,
        method: str = "sbert",
        metric: str = "cosine",
    ) -> Optional[float]:
        """Загрузить результат similarity из JSON файла.
        
        Args:
            domain1: Первый домен.
            domain2: Второй домен.
            method: Метод расчёта.
            metric: Метрика близости.
        
        Returns:
            Optional[float]: Значение близости или None если не найдено.
        """
        path = self._get_cache_path(domain1, domain2, method, metric)
        
        if not path.exists():
            logger.debug(f"Файл не найден в disk cache: {path}")
            return None
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            score = data.get("score")
            logger.debug(f"Загружено из disk cache: {path} = {score}")
            return float(score) if score is not None else None
        except Exception as e:
            logger.error(f"Ошибка чтения из disk cache: {e}")
            return None
    
    def load_all_similarities(
        self,
        method: Optional[str] = None,
    ) -> list[dict]:
        """Загрузить все результаты similarity из директории.
        
        Args:
            method: Метод для загрузки. Если None — все методы.
        
        Returns:
            list[dict]: Список всех записей similarity.
        """
        results = []
        methods = [method.lower()] if method else ["sbert", "rag", "wordnet", "bert", "tfidf", "ensemble"]
        
        for m in methods:
            method_dir = self.base_dir / m
            if not method_dir.exists():
                continue
            
            for json_file in method_dir.glob("*.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        results.append(data)
                except Exception as e:
                    logger.warning(f"Пропущен файл {json_file}: {e}")
                    continue
        
        logger.info(f"Загружено {len(results)} записей из disk cache")
        return results
    
    def get_cache_count(self, method: Optional[str] = None) -> int:
        """Получить количество записей в кеше.
        
        Args:
            method: Метод для подсчёта. Если None — все методы.
        
        Returns:
            int: Количество JSON файлов.
        """
        methods = [method.lower()] if method else ["sbert", "rag", "wordnet", "bert", "tfidf", "ensemble"]
        total = 0
        
        for m in methods:
            method_dir = self.base_dir / m
            if method_dir.exists():
                total += len(list(method_dir.glob("*.json")))
        
        return total
    
    def get_stats(self) -> dict:
        """Получить статистику disk cache.
        
        Returns:
            dict: Статистика по методам.
        """
        stats = {
            "total": 0,
            "methods": {},
        }
        
        methods = ["sbert", "rag", "wordnet", "bert", "tfidf", "ensemble"]
        for m in methods:
            method_dir = self.base_dir / m
            count = len(list(method_dir.glob("*.json"))) if method_dir.exists() else 0
            stats["methods"][m] = count
            stats["total"] += count
        
        return stats
    
    def clear_method(self, method: str) -> int:
        """Очистить кеш для конкретного метода.
        
        Args:
            method: Метод для очистки.
        
        Returns:
            int: Количество удалённых файлов.
        """
        method_dir = self.base_dir / method.lower()
        if not method_dir.exists():
            return 0
        
        count = 0
        for json_file in method_dir.glob("*.json"):
            try:
                json_file.unlink()
                count += 1
            except Exception as e:
                logger.warning(f"Не удалось удалить {json_file}: {e}")
        
        logger.info(f"Очищено {count} файлов для метода {method}")
        return count
    
    def clear_all(self) -> int:
        """Очистить весь disk cache.
        
        Returns:
            int: Количество удалённых файлов.
        """
        total = 0
        methods = ["sbert", "rag", "wordnet", "bert", "tfidf", "ensemble"]
        
        for m in methods:
            total += self.clear_method(m)
        
        logger.info(f"Очищен весь disk cache: {total} файлов")
        return total
