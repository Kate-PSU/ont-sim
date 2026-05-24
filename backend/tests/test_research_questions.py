"""
Тесты для модуля Research Questions.

Тестирует формализованные исследовательские вопросы и их связь
с контрастными парами методов.
"""

import pytest

from src.application.research_questions import (
    ResearchQuestionID,
    ContrastPair,
    ResearchQuestion,
    ALL_RESEARCH_QUESTIONS,
    RQ1, RQ2, RQ3,
    RQ_BY_ID,
    get_research_question,
    get_contrast_pairs_for_method,
    get_methods_for_rq,
    format_rq_summary,
)


class TestResearchQuestionID:
    """Тесты для enum ResearchQuestionID."""
    
    def test_all_ids_exist(self):
        """Проверяет, что все ID определены."""
        assert ResearchQuestionID.RQ1 is not None
        assert ResearchQuestionID.RQ2 is not None
        assert ResearchQuestionID.RQ3 is not None
    
    def test_id_values(self):
        """Проверяет значения ID."""
        assert ResearchQuestionID.RQ1.value == "rq1"
        assert ResearchQuestionID.RQ2.value == "rq2"
        assert ResearchQuestionID.RQ3.value == "rq3"


class TestContrastPair:
    """Тесты для контрастных пар."""
    
    def test_pair_creation(self):
        """Тест создания контрастной пары."""
        pair = ContrastPair(
            method_a="sbert",
            method_b="doc2vec",
            rationale="Тестовое сравнение",
        )
        assert pair.method_a == "sbert"
        assert pair.method_b == "doc2vec"
        assert pair.rationale == "Тестовое сравнение"
    
    def test_pair_name_property(self):
        """Тест свойства name."""
        pair = ContrastPair(
            method_a="sbert",
            method_b="doc2vec",
            rationale="Тест",
        )
        assert pair.name == "sbert vs doc2vec"


class TestResearchQuestion:
    """Тесты для Research Question."""
    
    def test_rq1_structure(self):
        """Тест структуры RQ1."""
        assert RQ1.id == ResearchQuestionID.RQ1
        assert "SBERT" in RQ1.hypothesis
        assert "Doc2Vec" in RQ1.hypothesis
        assert len(RQ1.contrast_pairs) == 1
        assert "hj-rg" in RQ1.datasets
    
    def test_rq2_structure(self):
        """Тест структуры RQ2."""
        assert RQ2.id == ResearchQuestionID.RQ2
        assert "WordNet" in RQ2.hypothesis
        assert len(RQ2.contrast_pairs) == 2  # wordnet vs sbert, wordnet vs tfidf
    
    def test_rq3_structure(self):
        """Тест структуры RQ3."""
        assert RQ3.id == ResearchQuestionID.RQ3
        assert "RAG" in RQ3.hypothesis or "rag" in RQ3.hypothesis.lower()
        assert len(RQ3.contrast_pairs) == 2
    
    def test_success_criteria(self):
        """Тест критериев успеха."""
        for rq in ALL_RESEARCH_QUESTIONS:
            assert len(rq.success_criteria) > 0
            for method, min_score in rq.success_criteria.items():
                assert isinstance(method, str)
                assert 0 <= min_score <= 1


class TestRQRegistry:
    """Тесты реестра RQ."""
    
    def test_all_rq_defined(self):
        """Проверяет, что все 3 RQ определены."""
        assert len(ALL_RESEARCH_QUESTIONS) == 3
    
    def test_rq_by_id(self):
        """Тест словаря RQ_BY_ID."""
        assert RQ_BY_ID[ResearchQuestionID.RQ1] is RQ1
        assert RQ_BY_ID[ResearchQuestionID.RQ2] is RQ2
        assert RQ_BY_ID[ResearchQuestionID.RQ3] is RQ3


class TestGetResearchQuestion:
    """Тесты функции get_research_question."""
    
    def test_get_by_enum(self):
        """Тест получения по enum."""
        result = get_research_question(ResearchQuestionID.RQ1)
        assert result is RQ1
    
    def test_get_by_string(self):
        """Тест получения по строке."""
        assert get_research_question("rq1") is RQ1
        assert get_research_question("rq2") is RQ2
        assert get_research_question("rq3") is RQ3
    
    def test_get_invalid(self):
        """Тест получения несуществующего RQ."""
        assert get_research_question("rq99") is None


class TestGetContrastPairsForMethod:
    """Тесты функции get_contrast_pairs_for_method."""
    
    def test_sbert_in_rq1(self):
        """Тест: SBERT участвует в RQ1."""
        pairs = get_contrast_pairs_for_method("sbert")
        assert len(pairs) >= 1
        rqs = [rq for rq, _ in pairs]
        assert RQ1 in rqs
    
    def test_wordnet_in_rq2(self):
        """Тест: WordNet участвует в RQ2."""
        pairs = get_contrast_pairs_for_method("wordnet_lin")
        assert len(pairs) >= 1
        rqs = [rq for rq, _ in pairs]
        assert RQ2 in rqs
    
    def test_rag_centroid_in_rq3(self):
        """Тест: RAG-centroid участвует в RQ3."""
        pairs = get_contrast_pairs_for_method("rag_centroid")
        assert len(pairs) >= 1
        rqs = [rq for rq, _ in pairs]
        assert RQ3 in rqs


class TestGetMethodsForRQ:
    """Тесты функции get_methods_for_rq."""
    
    def test_methods_for_rq1(self):
        """Тест методов для RQ1."""
        methods = get_methods_for_rq(ResearchQuestionID.RQ1)
        assert "sbert" in methods
        assert "doc2vec" in methods
    
    def test_methods_for_rq2(self):
        """Тест методов для RQ2."""
        methods = get_methods_for_rq("rq2")
        assert "wordnet_lin" in methods
        assert "sbert" in methods
        assert "tfidf" in methods
    
    def test_methods_for_rq3(self):
        """Тест методов для RQ3."""
        methods = get_methods_for_rq("rq3")
        assert "rag_centroid" in methods
    
    def test_invalid_rq(self):
        """Тест невалидного RQ."""
        assert get_methods_for_rq("rq99") == ()


class TestFormatRQSummary:
    """Тесты функции форматирования."""
    
    def test_format_rq1(self):
        """Тест форматирования RQ1."""
        output = format_rq_summary(RQ1)
        assert "RQ1" in output or "rq1" in output.lower()
        assert "Гипотеза" in output
        assert "Контрастные пары" in output
        assert "Датасеты" in output
        assert "Критерии успеха" in output
    
    def test_format_all_rq(self):
        """Тест форматирования всех RQ."""
        for rq in ALL_RESEARCH_QUESTIONS:
            output = format_rq_summary(rq)
            assert len(output) > 100  # Не пустой вывод
            assert rq.id.value.upper() in output
