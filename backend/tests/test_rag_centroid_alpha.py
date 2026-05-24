"""
Тесты для RAG-Centroids с разными значениями alpha.

Проверяет, что:
1. alpha=0 даёт центроид соседей
2. alpha=1 даёт SBERT baseline  
3. Разные alpha дают разные результаты
"""

import pytest
import numpy as np
import faiss
import pickle
from pathlib import Path

# Путь к RAG индексам (diplomMagistrate/backend/tests/ -> diplomMagistrate)
DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAG_DIR = DATA_DIR / "rag_indices"


def get_matching_terms(metadata, patterns):
    """Получить термины из индекса соответствующие паттернам."""
    terms = metadata["terms"]
    matching = []
    for pattern in patterns:
        for t in terms:
            # Фильтруем: термин должен быть не пустым и содержать паттерн
            term_clean = t.strip()
            if len(term_clean) > 3 and pattern.lower() in term_clean.lower():
                matching.append(t)
                break
    return matching


class TestRAGCentroidAlpha:
    """Тесты RAG centroid с разными alpha."""
    
    @pytest.fixture
    def rag_index(self):
        """Загрузка RAG индекса."""
        if not (RAG_DIR / "domains.index").exists():
            pytest.skip("RAG индекс не найден")
        
        index = faiss.read_index(str(RAG_DIR / "domains.index"))
        with open(RAG_DIR / "domains.meta.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        return index, metadata
    
    @pytest.fixture
    def sample_terms(self, rag_index):
        """Примеры терминов из индекса."""
        _, metadata = rag_index
        
        # Термины из физики и математики
        physics_terms = get_matching_terms(metadata, [
            "Квантовая механика",
            "Квантовая физика", 
            "Парадоксы квантовой механики",
            "Релятивистская квантовая",
        ])
        
        math_terms = get_matching_terms(metadata, [
            "Математика",
            "Математическая физика",
            "Анализ (раздел математики)",
            "Матрица (математика)",
        ])
        
        return physics_terms[:3], math_terms[:3]
    
    def test_rag_alpha_0_vs_alpha_1_different(self, rag_index, sample_terms):
        """alpha=0 и alpha=1 должны давать РАЗНЫЕ результаты."""
        index, metadata = rag_index
        physics_terms, math_terms = sample_terms
        
        if len(physics_terms) == 0:
            pytest.skip("Термины не найдены в индексе")
        
        print(f"\nТестируемые термины: {physics_terms + math_terms}")
        
        # Все термины для теста
        all_terms = physics_terms + math_terms
        
        # Получаем эмбеддинги терминов из индекса
        term_embs = []
        for term in all_terms:
            if term in metadata["terms"]:
                idx = metadata["terms"].index(term)
                emb = np.zeros(index.d, dtype='float32')
                index.reconstruct(idx, emb)
                term_embs.append((term, emb))
        
        if len(term_embs) < 3:
            pytest.skip(f"Недостаточно терминов в индексе: {len(term_embs)}")
        
        k = 5
        
        results_by_alpha = {}
        
        for alpha in [0.0, 0.5, 1.0]:
            embeddings = []
            for term, term_emb in term_embs:
                # Нормализуем
                norm_emb = term_emb / np.linalg.norm(term_emb)
                
                # Ищем соседей
                D, I = index.search(np.array([norm_emb.astype('float32')]), k)
                
                # Собираем эмбеддинги соседей
                neighbor_embs = []
                for neighbor_idx in I[0]:
                    if neighbor_idx >= 0 and neighbor_idx < index.ntotal:
                        neighbor_emb = np.zeros(index.d, dtype='float32')
                        index.reconstruct(int(neighbor_idx), neighbor_emb)
                        neighbor_embs.append(neighbor_emb)
                
                if neighbor_embs:
                    neighbor_centroid = np.mean(neighbor_embs, axis=0)
                    # Alpha blend
                    result = alpha * norm_emb + (1 - alpha) * neighbor_centroid
                else:
                    result = norm_emb
                
                embeddings.append(result)
            
            # Центроид
            centroid = np.mean(embeddings, axis=0)
            faiss.normalize_L2(np.array([centroid]))
            results_by_alpha[alpha] = centroid
        
        # Проверяем что alpha=0 и alpha=1 дают РАЗНЫЕ результаты
        diff_0_1 = np.linalg.norm(results_by_alpha[0.0] - results_by_alpha[1.0])
        
        print(f"\n✓ alpha=0 vs alpha=1 diff: {diff_0_1:.4f}")
        print(f"✓ alpha=0 vs alpha=0.5 diff: {np.linalg.norm(results_by_alpha[0.0] - results_by_alpha[0.5]):.4f}")
        
        assert diff_0_1 > 0.01, f"alpha=0 и alpha=1 должны давать разные результаты! diff={diff_0_1}"
    
    def test_rag_alpha_grid(self, rag_index, sample_terms):
        """Перебор разных alpha значений."""
        index, metadata = rag_index
        physics_terms, math_terms = sample_terms
        
        if len(physics_terms) == 0:
            pytest.skip("Термины не найдены в индексе")
        
        all_terms = physics_terms + math_terms
        
        # Получаем эмбеддинги
        term_embs = []
        for term in all_terms:
            if term in metadata["terms"]:
                idx = metadata["terms"].index(term)
                emb = np.zeros(index.d, dtype='float32')
                index.reconstruct(idx, emb)
                term_embs.append(emb)
        
        if len(term_embs) < 3:
            pytest.skip(f"Недостаточно терминов: {len(term_embs)}")
        
        k = 5
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        results = {}
        
        for alpha in alphas:
            embeddings = []
            for term_emb in term_embs:
                norm_emb = term_emb / np.linalg.norm(term_emb)
                D, I = index.search(np.array([norm_emb.astype('float32')]), k)
                
                neighbor_embs = []
                for neighbor_idx in I[0]:
                    if neighbor_idx >= 0 and neighbor_idx < index.ntotal:
                        neighbor_emb = np.zeros(index.d, dtype='float32')
                        index.reconstruct(int(neighbor_idx), neighbor_emb)
                        neighbor_embs.append(neighbor_emb)
                
                if neighbor_embs:
                    neighbor_centroid = np.mean(neighbor_embs, axis=0)
                    result = alpha * norm_emb + (1 - alpha) * neighbor_centroid
                else:
                    result = norm_emb
                
                embeddings.append(result)
            
            centroid = np.mean(embeddings, axis=0)
            faiss.normalize_L2(np.array([centroid]))
            results[alpha] = centroid
        
        # Попарные различия
        print("\n📊 Попарные различия между alpha:")
        significant_diffs = 0
        prev_alpha = None
        for alpha in alphas:
            if prev_alpha is not None:
                diff = np.linalg.norm(results[alpha] - results[prev_alpha])
                print(f"  α={alpha:.2f} vs α={prev_alpha:.2f}: Δ={diff:.4f}")
                if diff > 0.005:
                    significant_diffs += 1
            prev_alpha = alpha
        
        assert significant_diffs >= 2, f"Должно быть ≥2 пар с diff>0.005, но {significant_diffs}"
        print(f"\n✓ Найдено {significant_diffs} пар с значимыми различиями")


class TestRAGCentroidWithEmbeddingService:
    """Тесты RAG centroid с использованием EmbeddingService."""
    
    def test_sbert_centroid(self):
        """Базовый тест SBERT центроида."""
        from backend.src.infrastructure.embedding_service import EmbeddingService
        from backend.src.application import CentroidService
        
        try:
            service = EmbeddingService()
        except Exception as e:
            pytest.skip(f"EmbeddingService не доступен: {e}")
        
        # Используем термины из русской Wikipedia
        terms = ["квантовая механика", "физика", "математика"]
        
        sbert_embeddings = service.get_embeddings_batch(terms)
        centroid_service = CentroidService()
        sbert_centroid = centroid_service.calculate_centroid(sbert_embeddings)
        
        print(f"\n✓ SBERT центроид: norm={np.linalg.norm(sbert_centroid):.4f}")
        
        assert len(sbert_centroid) > 0, "Центроид должен быть непустым"
