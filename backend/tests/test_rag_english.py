# backend/tests/test_rag_english.py
# Тесты для RAG-Centroid на английском языке
#
# Версия: 1.0
# Создано: 2026-05-01
# Задачи: Debug English RAG benchmarks

"""
Тесты для проверки RAG-Centroid на английских датасетах.
Проверяют:
1. Загрузку английского RAG индекса
2. Совпадение размерностей эмбеддингов
3. Вычисление близости на парах simlex999
"""

import pytest
from pathlib import Path
import sys
import numpy as np

# Добавляем корень проекта в путь
_project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_root))

from backend.src.infrastructure.embedding_service import EmbeddingService
from backend.src.infrastructure.retrieval_service import RetrievalService


class TestRAGEnglishIndex:
    """Тесты загрузки английского RAG индекса."""
    
    def test_simlex999_en_index_exists(self):
        """Индекс для simlex999_en должен существовать."""
        index_path = _project_root / "data" / "rag_indices" / "rag_simlex999_en_62e6e60189d0.index"
        meta_path = _project_root / "data" / "rag_indices" / "rag_simlex999_en_62e6e60189d0.meta.pkl"
        
        assert index_path.exists(), f"Index not found: {index_path}"
        assert meta_path.exists(), f"Metadata not found: {meta_path}"
    
    def test_simlex999_en_index_metadata(self):
        """Метаданные английского индекса должны содержать правильную информацию."""
        import pickle
        
        meta_path = _project_root / "data" / "rag_indices" / "rag_simlex999_en_62e6e60189d0.meta.pkl"
        
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        
        print(f"Index metadata:")
        print(f"  terms count: {len(meta['terms'])}")
        print(f"  embeddings shape: {meta['embeddings_shape']}")
        
        # Проверяем ожидаемые значения
        assert len(meta['terms']) > 500, f"Expected >500 terms, got {len(meta['terms'])}"
        assert meta['embeddings_shape'][0] > 500, "Expected >500 embeddings"
        # Размерность должна быть 384 (MiniLM)
        expected_dim = 384
        actual_dim = meta['embeddings_shape'][1]
        assert actual_dim == expected_dim, \
            f"Expected dimension {expected_dim} (MiniLM), got {actual_dim}"
    
    def test_embedding_model_dimension_matches_index(self):
        """Размерность модели должна совпадать с размерностью индекса."""
        import pickle
        
        # Загружаем метаданные индекса
        meta_path = _project_root / "data" / "rag_indices" / "rag_simlex999_en_62e6e60189d0.meta.pkl"
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        
        index_dim = meta['embeddings_shape'][1]
        
        # Создаём embedding service с MPNet моделью
        mpnet_model = EmbeddingService(model_name="sentence-transformers/all-mpnet-base-v2")
        mpnet_dim = mpnet_model.model.get_sentence_embedding_dimension()
        
        print(f"Index dimension: {index_dim}")
        print(f"MPNet dimension: {mpnet_dim}")
        
        # Это должно не совпадать!
        # Индекс построен с MiniLM (384 dim), а тестирование идёт с MPNet (768 dim)
        if index_dim != mpnet_dim:
            pytest.fail(
                f"DIMENSION MISMATCH! "
                f"Index built with {index_dim}dim model, "
                f"but testing uses {mpnet_dim}dim MPNet. "
                f"Need to rebuild index with MPNet or use MiniLM for testing."
            )


class TestRAGCentroidOnSimlex999:
    """Тесты RAG-Centroid на парах из SimLex-999."""
    
    @pytest.fixture
    def embedding_service(self):
        """Создание embedding service с правильной моделью."""
        # Используем MiniLM для совместимости с индексом
        service = EmbeddingService(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
        return service
    
    @pytest.fixture
    def retrieval_service(self, embedding_service):
        """Загрузка английского RAG индекса."""
        index_path = _project_root / "data" / "rag_indices" / "rag_simlex999_en_62e6e60189d0.index"
        
        service = RetrievalService.load_index(index_path, embedding_service)
        
        if service is None:
            pytest.skip("Failed to load RAG index")
        
        return service
    
    @pytest.fixture
    def simlex999_pairs(self):
        """Загрузка пар из SimLex-999."""
        csv_path = _project_root / "data" / "simlex999.csv"
        if not csv_path.exists():
            pytest.skip(f"SimLex-999 CSV not found: {csv_path}")
        
        import csv
        pairs = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= 20:
                    break
                pairs.append({
                    'word1': row['word1'],
                    'word2': row['word2'],
                    'sim': float(row['sim']),
                })
        
        return pairs
    
    def test_rag_centroid_basic_pair(self, retrieval_service, embedding_service):
        """RAG-Centroid вычисляет близость для простой английской пары."""
        k = 5
        alpha = 0.5
        
        # Тестируем простую пару
        term1 = "car"
        term2 = "automobile"
        
        # Получаем retrieval контекст
        retrieved1 = retrieval_service.get_retrieved_context(term1, k)
        retrieved2 = retrieval_service.get_retrieved_context(term2, k)
        
        assert len(retrieved1) > 0, f"No neighbors found for '{term1}'"
        assert len(retrieved2) > 0, f"No neighbors found for '{term2}'"
        
        print(f"Retrieved for 'car': {retrieved1}")
        print(f"Retrieved for 'automobile': {retrieved2}")
    
    def test_rag_centroid_on_simlex999_pairs(self, retrieval_service, embedding_service, simlex999_pairs):
        """RAG-Centroid вычисляет близость для пар simlex999."""
        k = 5
        alpha = 0.5
        
        nonzero_count = 0
        errors = []
        
        for pair in simlex999_pairs:
            term1 = pair['word1']
            term2 = pair['word2']
            
            try:
                # Вычисляем RAG эмбеддинги
                emb1 = embedding_service.get_embedding(term1)
                emb2 = embedding_service.get_embedding(term2)
                
                # Получаем retrieval контекст
                retrieved1 = retrieval_service.get_retrieved_context(term1, k)
                retrieved2 = retrieval_service.get_retrieved_context(term2, k)
                
                if retrieved1 and retrieved2:
                    # Вычисляем centroids
                    embs1 = [embedding_service.get_embedding(t) for t in retrieved1]
                    embs2 = [embedding_service.get_embedding(t) for t in retrieved2]
                    
                    centroid1 = np.mean(embs1, axis=0)
                    centroid2 = np.mean(embs2, axis=0)
                    
                    # RAG эмбеддинги
                    rag_emb1 = alpha * emb1 + (1 - alpha) * centroid1
                    rag_emb2 = alpha * emb2 + (1 - alpha) * centroid2
                    
                    # Косинусная близость
                    cos_sim = np.dot(rag_emb1, rag_emb2) / (
                        np.linalg.norm(rag_emb1) * np.linalg.norm(rag_emb2)
                    )
                    
                    if cos_sim != 0.0:
                        nonzero_count += 1
                else:
                    errors.append(f"No neighbors for ({term1}, {term2})")
                    
            except Exception as e:
                errors.append(f"Error for ({term1}, {term2}): {str(e)}")
        
        print(f"Non-zero similarities: {nonzero_count}/{len(simlex999_pairs)}")
        if errors:
            print(f"Errors: {errors[:5]}")  # Первые 5 ошибок
        
        # RAG должен давать ненулевые результаты для большинства пар
        assert nonzero_count >= len(simlex999_pairs) * 0.5, \
            f"Too many zero similarities: {nonzero_count}/{len(simlex999_pairs)}"


class TestRAGDimensionFix:
    """Тесты для проверки исправления размерностей."""
    
    def test_fix_requirement_for_rag_english(self):
        """Документируем требование по исправлению размерностей."""
        import pickle
        
        # Проверяем текущее состояние
        index_path = _project_root / "data" / "rag_indices" / "rag_simlex999_en_62e6e60189d0.index"
        
        # Если индекс существует, проверяем его размерность
        if index_path.exists():
            # Загружаем метаданные
            meta_path = index_path.with_suffix('.meta.pkl')
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            
            index_dim = meta['embeddings_shape'][1]
            
            # Получаем размерности разных моделей
            from sentence_transformers import SentenceTransformer
            
            models = {
                'MiniLM': 'sentence-transformers/paraphrase-MiniLM-L6-v2',
                'MPNet': 'sentence-transformers/all-mpnet-base-v2',
            }
            
            dims = {}
            for name, path in models.items():
                model = SentenceTransformer(path)
                dims[name] = model.get_sentence_embedding_dimension()
            
            print(f"Index dimension: {index_dim}")
            print(f"Model dimensions: {dims}")
            
            # Определяем, какая модель использовалась для индекса
            matching_models = [name for name, dim in dims.items() if dim == index_dim]
            
            if matching_models:
                print(f"Index was built with: {matching_models[0]} ({index_dim} dim)")
                print(f"Testing uses: MPNet (768 dim)")
            else:
                print(f"WARNING: No matching model found for {index_dim} dim!")
            
            # Если не совпадает - это known issue
            assert index_dim == 384, \
                f"Expected 384 dim (MiniLM), got {index_dim}. Check index metadata."
    
    def test_known_rag_indices_mapping(self):
        """Проверяем маппинг KNOWN_RAG_INDICES."""
        from scripts.benchmark.runners import BenchmarkRunner
        
        # Проверяем что simlex999_en есть в KNOWN_RAG_INDICES
        assert "simlex999_en" in BenchmarkRunner.KNOWN_RAG_INDICES, \
            "simlex999_en should be in KNOWN_RAG_INDICES"
        assert "simlex999" in BenchmarkRunner.KNOWN_RAG_INDICES, \
            "simlex999 should be in KNOWN_RAG_INDICES"
        
        print(f"KNOWN_RAG_INDICES: {BenchmarkRunner.KNOWN_RAG_INDICES}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])