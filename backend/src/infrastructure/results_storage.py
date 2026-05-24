# backend/src/infrastructure/results_storage.py
# Хранилище результатов бенчмарков в SQLite
#
# Версия: 1.0
# Обновлено: 2026-04-09

"""
Модуль персистентного хранения результатов бенчмарков.

Сохраняет результаты в SQLite файл для долгосрочного хранения.
Результаты доступны даже после перезапуска Redis/Celery.
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Путь к файлу базы данных
RESULTS_DB_PATH = Path("/app/benchmark_results.db")


class ResultsStorage:
    """Хранилище результатов бенчмарков в SQLite."""
    
    def __init__(self, db_path: Optional[Path] = None) -> None:
        """Инициализация хранилища.
        
        Args:
            db_path: Путь к файлу SQLite. По умолчанию /data/benchmark_results.db
        """
        self.db_path = db_path or RESULTS_DB_PATH
        self._init_db()
    
    def _init_db(self) -> None:
        """Инициализация структуры базы данных."""
        # Создаём директорию если нужно
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Таблица результатов
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset TEXT NOT NULL,
                results_json TEXT NOT NULL,
                execution_time_sec REAL NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(dataset)
            )
        """)
        
        # Таблица истории (для каждого запуска)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset TEXT NOT NULL,
                results_json TEXT NOT NULL,
                execution_time_sec REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
        
        logger.info(f"[Storage] База данных инициализирована: {self.db_path}")
    
    def save_result(
        self,
        dataset: str,
        results: list[dict],
        execution_time_sec: float,
    ) -> bool:
        """Сохранение результатов бенчмарка.
        
        Args:
            dataset: Название датасета.
            results: Список результатов методов.
            execution_time_sec: Время выполнения в секундах.
        
        Returns:
            True если сохранено успешно.
        """
        try:
            now = datetime.utcnow().isoformat()
            results_json = json.dumps(results, ensure_ascii=False)
            
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Upsert в основную таблицу
            cursor.execute("""
                INSERT INTO benchmark_results (dataset, results_json, execution_time_sec, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(dataset) DO UPDATE SET
                    results_json = excluded.results_json,
                    execution_time_sec = excluded.execution_time_sec,
                    updated_at = excluded.updated_at
            """, (dataset, results_json, execution_time_sec, now, now))
            
            # Добавляем в историю
            cursor.execute("""
                INSERT INTO benchmark_history (dataset, results_json, execution_time_sec, created_at)
                VALUES (?, ?, ?, ?)
            """, (dataset, results_json, execution_time_sec, now))
            
            conn.commit()
            conn.close()
            
            logger.info(f"[Storage] Сохранены результаты для {dataset}")
            return True
            
        except Exception as e:
            logger.error(f"[Storage] Ошибка сохранения: {e}")
            return False
    
    def get_result(self, dataset: str) -> Optional[dict]:
        """Получение последних результатов для датасета.
        
        Args:
            dataset: Название датасета.
        
        Returns:
            Словарь с результатами или None если не найдено.
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT dataset, results_json, execution_time_sec, created_at, updated_at
                FROM benchmark_results
                WHERE dataset = ?
            """, (dataset,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    "dataset_name": row[0],
                    "results": json.loads(row[1]),
                    "execution_time_sec": row[2],
                    "created_at": row[3],
                    "updated_at": row[4],
                }
            
            return None
            
        except Exception as e:
            logger.error(f"[Storage] Ошибка чтения: {e}")
            return None
    
    def get_history(self, dataset: str, limit: int = 10) -> list[dict]:
        """Получение истории результатов для датасета.
        
        Args:
            dataset: Название датасета.
            limit: Максимальное количество записей.
        
        Returns:
            Список исторических результатов.
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT dataset, results_json, execution_time_sec, created_at
                FROM benchmark_history
                WHERE dataset = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (dataset, limit))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [
                {
                    "dataset_name": row[0],
                    "results": json.loads(row[1]),
                    "execution_time_sec": row[2],
                    "created_at": row[3],
                }
                for row in rows
            ]
            
        except Exception as e:
            logger.error(f"[Storage] Ошибка чтения истории: {e}")
            return []
    
    def list_datasets(self) -> list[str]:
        """Получение списка датасетов с сохранёнными результатами.
        
        Returns:
            Список названий датасетов.
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("SELECT dataset FROM benchmark_results ORDER BY updated_at DESC")
            rows = cursor.fetchall()
            conn.close()
            
            return [row[0] for row in rows]
            
        except Exception as e:
            logger.error(f"[Storage] Ошибка получения списка: {e}")
            return []


# Глобальный экземпляр хранилища
_storage: Optional[ResultsStorage] = None


def get_storage() -> ResultsStorage:
    """Получение глобального экземпляра хранилища.
    
    Returns:
        Экземпляр ResultsStorage.
    """
    global _storage
    if _storage is None:
        _storage = ResultsStorage()
    return _storage


def save_benchmark_results(
    dataset: str,
    results: list[dict],
    execution_time_sec: float,
) -> bool:
    """Удобная функция для сохранения результатов.
    
    Args:
        dataset: Название датасета.
        results: Список результатов методов.
        execution_time_sec: Время выполнения.
    
    Returns:
        True если сохранено успешно.
    """
    return get_storage().save_result(dataset, results, execution_time_sec)


def get_benchmark_result(dataset: str) -> Optional[dict]:
    """Удобная функция для получения результатов.
    
    Args:
        dataset: Название датасета.
    
    Returns:
        Результаты или None.
    """
    return get_storage().get_result(dataset)