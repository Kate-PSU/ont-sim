# backend/tests/test_domain.py
# Тесты для доменного слоя (Domain Layer)
#
# Версия: 1.0
# Обновлено: 2026-04-10

"""
Тесты для сущностей доменного слоя:
- Term (Термин)
- Domain (Предметная область)
- Similarity (Оценка близости)
- GraphData (Данные графа)
"""

import pytest
from src.domain import Term, Domain, Similarity, GraphData


class TestTerm:
    """Тесты для сущности Term."""

    def test_term_create_minimal(self):
        """Тест: создание термина с минимальными параметрами."""
        term = Term(name="нейронная сеть", domain="ML")
        
        assert term.name == "нейронная сеть"
        assert term.domain == "ML"
        assert term.frequency == 1  # значение по умолчанию

    def test_term_create_full(self):
        """Тест: создание термина со всеми параметрами."""
        term = Term(name="машинное обучение", domain="ML", frequency=500)
        
        assert term.name == "машинное обучение"
        assert term.domain == "ML"
        assert term.frequency == 500

    def test_term_equality(self):
        """Тест: равенство терминов."""
        term1 = Term(name="сеть", domain="ML", frequency=100)
        term2 = Term(name="сеть", domain="ML", frequency=100)
        term3 = Term(name="сеть", domain="ML", frequency=200)
        
        assert term1 == term2
        assert term1 != term3

    def test_term_repr(self):
        """Тест: строковое представление термина."""
        term = Term(name="BERT", domain="NLP")
        
        assert "BERT" in repr(term)
        assert "NLP" in repr(term)


class TestDomain:
    """Тесты для сущности Domain."""

    def test_domain_create_empty(self):
        """Тест: создание пустого домена."""
        domain = Domain(name="ML", terms=[])
        
        assert domain.name == "ML"
        assert domain.terms == []
        assert len(domain.terms) == 0

    def test_domain_create_with_terms(self):
        """Тест: создание домена с терминами."""
        terms = [
            Term(name="нейронная сеть", domain="ML"),
            Term(name="глубокое обучение", domain="ML"),
        ]
        domain = Domain(name="ML", terms=terms)
        
        assert domain.name == "ML"
        assert len(domain.terms) == 2
        assert domain.terms[0].name == "нейронная сеть"

    def test_domain_terms_count(self):
        """Тест: подсчёт количества терминов."""
        terms = [
            Term(name="т1", domain="ML"),
            Term(name="т2", domain="ML"),
            Term(name="т3", domain="ML"),
        ]
        domain = Domain(name="ML", terms=terms)
        
        assert len(domain.terms) == 3

    def test_domain_equality(self):
        """Тест: равенство доменов."""
        domain1 = Domain(name="ML", terms=[])
        domain2 = Domain(name="ML", terms=[])
        
        assert domain1 == domain2

    def test_domain_repr(self):
        """Тест: строковое представление домена."""
        domain = Domain(name="NLP", terms=[])
        
        assert "NLP" in repr(domain)


class TestSimilarity:
    """Тесты для сущности Similarity."""

    def test_similarity_create_minimal(self):
        """Тест: создание оценки близости с минимальными параметрами."""
        similarity = Similarity(
            domain1="ML",
            domain2="NLP",
            score=0.85
        )
        
        assert similarity.domain1 == "ML"
        assert similarity.domain2 == "NLP"
        assert similarity.score == 0.85
        assert similarity.metric == "cosine"  # значение по умолчанию

    def test_similarity_create_full(self):
        """Тест: создание оценки близости со всеми параметрами."""
        similarity = Similarity(
            domain1="ML",
            domain2="NLP",
            score=0.92,
            metric="euclidean"
        )
        
        assert similarity.domain1 == "ML"
        assert similarity.domain2 == "NLP"
        assert similarity.score == 0.92
        assert similarity.metric == "euclidean"

    def test_similarity_score_range(self):
        """Тест: проверка диапазона score."""
        # Косинусное сходство: [-1, 1]
        similarity = Similarity(domain1="A", domain2="B", score=1.0)
        assert similarity.score == 1.0

        similarity = Similarity(domain1="A", domain2="B", score=-1.0)
        assert similarity.score == -1.0

    def test_similarity_equality(self):
        """Тест: равенство оценок близости."""
        sim1 = Similarity(domain1="ML", domain2="NLP", score=0.8)
        sim2 = Similarity(domain1="ML", domain2="NLP", score=0.8)
        
        assert sim1 == sim2

    def test_similarity_repr(self):
        """Тест: строковое представление оценки близости."""
        similarity = Similarity(
            domain1="ML",
            domain2="NLP",
            score=0.75
        )
        
        assert "ML" in repr(similarity)
        assert "NLP" in repr(similarity)
        assert "0.75" in repr(similarity)


class TestGraphData:
    """Тесты для сущности GraphData."""

    def test_graph_data_create_empty(self):
        """Тест: создание пустого графа."""
        graph = GraphData(nodes=[], edges=[])
        
        assert graph.nodes == []
        assert graph.edges == []

    def test_graph_data_create_with_data(self):
        """Тест: создание графа с данными."""
        nodes = [
            {"id": "ML", "label": "Machine Learning"},
            {"id": "NLP", "label": "Natural Language Processing"},
        ]
        edges = [
            {"source": "ML", "target": "NLP", "weight": 0.85},
        ]
        graph = GraphData(nodes=nodes, edges=edges)
        
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1
        assert graph.nodes[0]["id"] == "ML"

    def test_graph_data_nodes_format(self):
        """Тест: формат узлов графа."""
        nodes = [{"id": "A", "label": "Domain A"}]
        graph = GraphData(nodes=nodes, edges=[])
        
        assert "id" in graph.nodes[0]
        assert "label" in graph.nodes[0]

    def test_graph_data_edges_format(self):
        """Тест: формат рёбер графа."""
        edges = [
            {"source": "A", "target": "B", "weight": 0.5}
        ]
        graph = GraphData(nodes=[], edges=edges)
        
        assert "source" in graph.edges[0]
        assert "target" in graph.edges[0]
        assert "weight" in graph.edges[0]

    def test_graph_data_empty_graph(self):
        """Тест: пустой граф имеет 0 узлов и рёбер."""
        graph = GraphData(nodes=[], edges=[])
        
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0


class TestTermEdgeCases:
    """Edge cases для Term."""

    def test_term_empty_name(self):
        """Тест: термин с пустым названием (допустимо)."""
        term = Term(name="", domain="ML")
        assert term.name == ""

    def test_term_unicode(self):
        """Тест: термин с юникодом."""
        term = Term(name="нейронная сеть", domain="ML")
        assert "нейронная" in term.name

    def test_term_frequency_zero(self):
        """Тест: частота равна нулю."""
        term = Term(name="редкий", domain="ML", frequency=0)
        assert term.frequency == 0

    def test_term_frequency_negative(self):
        """Тест: отрицательная частота (допустимо в модели)."""
        term = Term(name="термин", domain="ML", frequency=-5)
        assert term.frequency == -5


class TestDomainEdgeCases:
    """Edge cases для Domain."""

    def test_domain_single_term(self):
        """Тест: домен с одним термином."""
        terms = [Term(name="единственный", domain="ML")]
        domain = Domain(name="ML", terms=terms)
        
        assert len(domain.terms) == 1

    def test_domain_many_terms(self):
        """Тест: домен со многими терминами."""
        terms = [Term(name=f"термин_{i}", domain="ML") for i in range(100)]
        domain = Domain(name="ML", terms=terms)
        
        assert len(domain.terms) == 100

    def test_domain_cyrillic_name(self):
        """Тест: название домена на кириллице."""
        domain = Domain(name="Машинное обучение", terms=[])
        assert domain.name == "Машинное обучение"


class TestSimilarityEdgeCases:
    """Edge cases для Similarity."""

    def test_similarity_same_domain(self):
        """Тест: близость домена к самому себе."""
        similarity = Similarity(domain1="ML", domain2="ML", score=1.0)
        assert similarity.domain1 == similarity.domain2

    def test_similarity_zero_score(self):
        """Тест: нулевая близость (ортогональные векторы)."""
        similarity = Similarity(domain1="A", domain2="B", score=0.0)
        assert similarity.score == 0.0

    def test_similarity_extreme_scores(self):
        """Тест: экстремальные значения близости."""
        similarity_max = Similarity(domain1="A", domain2="B", score=1.0)
        similarity_min = Similarity(domain1="A", domain2="B", score=-1.0)
        
        assert similarity_max.score == 1.0
        assert similarity_min.score == -1.0
