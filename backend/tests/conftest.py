# backend/tests/conftest.py
# Общие фикстуры для тестов
#
# Версия: 1.0
# Обновлено: 2026-04-06

import sys
from pathlib import Path

import pytest
import numpy as np

# Добавляем корневую директорию backend в sys.path для импортов
backend_root = Path(__file__).parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))


@pytest.fixture
def sample_vectors():
    """Генерация тестовых векторов."""
    return {
        "vec_a": np.array([1.0, 0.0, 0.0]),
        "vec_b": np.array([1.0, 0.0, 0.0]),  # идентичный vec_a
        "vec_c": np.array([0.0, 1.0, 0.0]),  # ортогональный
        "vec_d": np.array([-1.0, 0.0, 0.0]),  # противоположный
        "vec_random": np.array([0.5, 0.5, 0.707]),
    }


@pytest.fixture
def sample_centroids():
    """Генерация тестовых центроидов."""
    return {
        "mathematics": np.array([1.0, 0.5, 0.2]),
        "physics": np.array([0.9, 0.6, 0.3]),
        "literature": np.array([-0.1, -0.2, 0.9]),
    }
