# backend/tests/test_routes_schemas.py
# Тесты для API routes и schemas
#
# Версия: 1.0
# Обновлено: 2026-04-15

"""
Тесты для API эндпоинтов — routes.py и schemas.py.
"""

import pytest
from pydantic import ValidationError

from src.presentation.schemas import (
    HealthResponse,
    DomainsResponse,
    SimilarityRequest,
    SimilarityResponse,
    GraphNode,
    GraphEdge,
    GraphResponse,
    GraphResponseDetailed,
    GraphNodeDetailed,
    TermInput,
    DomainInput,
    UploadDataRequest,
    UploadDataResponse,
    BenchmarkRequest,
    MethodResultSchema,
    BenchmarkComparisonSchema,
    BenchmarkResponse,
    BenchmarkDatasetsResponse,
    EnrichmentInfoResponse,
    EnrichedSimilarityResponse,
    SynsetInfoSchema,
    WordNetSimilarityResponse,
    WordNetDomainSimilarityResponse,
    HypernymsResponse,
    BenchmarkMetricsSchema,
    BenchmarkMatrixCell,
    BenchmarkMatrixRow,
    BenchmarkMatrixResponse,
    DatasetStatus,
    ActiveTask,
    SystemStateResponse,
    BenchmarkRunRequest,
    BenchmarkRunResponse,
)


class TestHealthResponse:
    """Тесты схемы HealthResponse."""
    
    def test_health_response_valid(self):
        """Тест: валидный ответ."""
        response = HealthResponse(status="ok")
        assert response.status == "ok"
    
    def test_health_response_status_values(self):
        """Тест: различные статусы."""
        for status in ["ok", "degraded", "error"]:
            response = HealthResponse(status=status)
            assert response.status == status


class TestDomainsResponse:
    """Тесты схемы DomainsResponse."""
    
    def test_domains_response_valid(self):
        """Тест: валидный ответ со списком доменов."""
        response = DomainsResponse(domains=["ML", "DL", "NLP"])
        assert len(response.domains) == 3
        assert "ML" in response.domains
    
    def test_domains_response_empty(self):
        """Тест: пустой список."""
        response = DomainsResponse(domains=[])
        assert len(response.domains) == 0


class TestSimilarityRequest:
    """Тесты схемы SimilarityRequest."""
    
    def test_similarity_request_valid(self):
        """Тест: валидный запрос близости."""
        request = SimilarityRequest(domain1="ML", domain2="DL")
        assert request.domain1 == "ML"
        assert request.domain2 == "DL"
    
    def test_similarity_request_with_metric(self):
        """Тест: запрос с метрикой."""
        request = SimilarityRequest(domain1="ML", domain2="DL", metric="euclidean")
        assert request.metric == "euclidean"
    
    def test_similarity_request_default_metric(self):
        """Тест: метрика по умолчанию."""
        request = SimilarityRequest(domain1="ML", domain2="DL")
        assert request.metric == "cosine"


class TestSimilarityResponse:
    """Тесты схемы SimilarityResponse."""
    
    def test_similarity_response_valid(self):
        """Тест: валидный ответ."""
        response = SimilarityResponse(domain1="ML", domain2="DL", score=0.85, metric="cosine")
        assert response.domain1 == "ML"
        assert response.domain2 == "DL"
        assert response.score == 0.85
        assert response.metric == "cosine"
    
    def test_similarity_score_bounds(self):
        """Тест: score должен быть в [0, 1]."""
        r1 = SimilarityResponse(domain1="a", domain2="b", score=0.0, metric="cosine")
        assert r1.score == 0.0
        r2 = SimilarityResponse(domain1="a", domain2="b", score=1.0, metric="cosine")
        assert r2.score == 1.0


class TestGraphNode:
    """Тесты схемы GraphNode."""
    
    def test_graph_node_valid(self):
        """Тест: валидный узел."""
        node = GraphNode(id="ML", label="ML")
        assert node.id == "ML"
        assert node.type == "domain"  # default
    
    def test_graph_node_with_type(self):
        """Тест: узел с типом."""
        node = GraphNode(id="term1", label="Term", type="term")
        assert node.type == "term"


class TestGraphNodeDetailed:
    """Тесты схемы GraphNodeDetailed."""
    
    def test_graph_node_detailed_valid(self):
        """Тест: детальный узел с родителем."""
        node = GraphNodeDetailed(id="term1", label="Neural Network", type="term", parent="ML")
        assert node.parent == "ML"
    
    def test_graph_node_detailed_no_parent(self):
        """Тест: детальный узел без родителя."""
        node = GraphNodeDetailed(id="ML", label="ML", type="domain")
        assert node.parent is None


class TestGraphEdge:
    """Тесты схемы GraphEdge."""
    
    def test_graph_edge_valid(self):
        """Тест: валидное ребро."""
        edge = GraphEdge(source="ML", target="DL", weight=0.8)
        assert edge.source == "ML"
        assert edge.target == "DL"
        assert edge.weight == 0.8
        assert edge.type == "similarity"  # default
    
    def test_graph_edge_belongs_to(self):
        """Тест: ребро принадлежности."""
        edge = GraphEdge(source="term1", target="ML", weight=1.0, type="belongs_to")
        assert edge.type == "belongs_to"


class TestGraphResponse:
    """Тесты схемы GraphResponse."""
    
    def test_graph_response_valid(self):
        """Тест: валидный ответ графа."""
        nodes = [GraphNode(id="ML", label="ML"), GraphNode(id="DL", label="DL")]
        edges = [GraphEdge(source="ML", target="DL", weight=0.8)]
        response = GraphResponse(nodes=nodes, edges=edges, threshold=0.5)
        assert len(response.nodes) == 2
        assert len(response.edges) == 1
        assert response.threshold == 0.5
    
    def test_graph_response_empty(self):
        """Тест: пустой граф."""
        response = GraphResponse(nodes=[], edges=[], threshold=0.5)
        assert len(response.nodes) == 0
        assert len(response.edges) == 0


class TestTermInput:
    """Тесты схемы TermInput."""
    
    def test_term_input_valid(self):
        """Тест: валидный ввод термина."""
        term = TermInput(name="нейронная сеть", frequency=10)
        assert term.name == "нейронная сеть"
        assert term.frequency == 10
    
    def test_term_input_default_frequency(self):
        """Тест: частота по умолчанию."""
        term = TermInput(name="нейронная сеть")
        assert term.frequency == 1


class TestDomainInput:
    """Тесты схемы DomainInput."""
    
    def test_domain_input_valid(self):
        """Тест: валидный ввод домена."""
        domain = DomainInput(name="ML", terms=[TermInput(name="нейронная сеть")])
        assert domain.name == "ML"
        assert len(domain.terms) == 1
    
    def test_domain_input_multiple_terms(self):
        """Тест: несколько терминов."""
        domain = DomainInput(
            name="ML",
            terms=[
                TermInput(name="нейронная сеть"),
                TermInput(name="gradient descent", frequency=5),
            ],
        )
        assert len(domain.terms) == 2


class TestUploadDataRequest:
    """Тесты схемы UploadDataRequest."""
    
    def test_upload_data_request_valid(self):
        """Тест: валидный запрос загрузки."""
        request = UploadDataRequest(
            domains=[DomainInput(name="ML", terms=[TermInput(name="нейронная сеть")])]
        )
        assert len(request.domains) == 1
    
    def test_upload_data_request_empty_rejected(self):
        """Тест: пустой список отклоняется."""
        with pytest.raises(ValidationError):
            UploadDataRequest(domains=[])


class TestUploadDataResponse:
    """Тесты схемы UploadDataResponse."""
    
    def test_upload_data_response_valid(self):
        """Тест: валидный ответ загрузки."""
        response = UploadDataResponse(
            success=True, domains_loaded=2, terms_loaded=10, message="OK"
        )
        assert response.success is True
        assert response.domains_loaded == 2
        assert response.terms_loaded == 10


class TestBenchmarkRequest:
    """Тесты схемы BenchmarkRequest."""
    
    def test_benchmark_request_valid(self):
        """Тест: валидный запрос бенчмарка."""
        request = BenchmarkRequest(dataset="hj-rg")
        assert request.dataset == "hj-rg"
    
    def test_benchmark_request_with_methods(self):
        """Тест: запрос с методами."""
        request = BenchmarkRequest(dataset="hj-rg", methods=["sbert", "wordnet"])
        assert len(request.methods) == 2
    
    def test_benchmark_request_methods_none(self):
        """Тест: методы None по умолчанию."""
        request = BenchmarkRequest(dataset="hj-rg")
        assert request.methods is None


class TestMethodResultSchema:
    """Тесты схемы MethodResultSchema."""
    
    def test_method_result_valid(self):
        """Тест: валидный результат метода."""
        result = MethodResultSchema(
            method="sbert",
            spearman=0.85,
            pearson=0.82,
            mse=0.05,
            missing=0,
            predictions_count=100,
        )
        assert result.method == "sbert"
        assert result.spearman == 0.85
        assert result.predictions_count == 100


class TestBenchmarkComparisonSchema:
    """Тесты схемы BenchmarkComparisonSchema."""
    
    def test_benchmark_comparison_valid(self):
        """Тест: валидное сравнение."""
        comparison = BenchmarkComparisonSchema(
            dataset_name="hj-rg", dataset_size=500, execution_time_sec=45.0, results=[]
        )
        assert comparison.dataset_name == "hj-rg"
        assert comparison.dataset_size == 500


class TestBenchmarkResponse:
    """Тесты схемы BenchmarkResponse."""
    
    def test_benchmark_response_success(self):
        """Тест: успешный ответ."""
        response = BenchmarkResponse(
            success=True,
            comparison=BenchmarkComparisonSchema(
                dataset_name="hj-rg", dataset_size=500, execution_time_sec=45.0, results=[]
            ),
        )
        assert response.success is True
        assert response.comparison is not None
    
    def test_benchmark_response_error(self):
        """Тест: ответ с ошибкой."""
        response = BenchmarkResponse(success=False, error="Benchmark failed")
        assert response.success is False
        assert response.error == "Benchmark failed"


class TestBenchmarkDatasetsResponse:
    """Тесты схемы BenchmarkDatasetsResponse."""
    
    def test_benchmark_datasets_valid(self):
        """Тест: валидный ответ."""
        response = BenchmarkDatasetsResponse(
            datasets=[
                {"name": "hj-rg", "path": "/path/to/hj-rg.csv", "size": 500},
                {"name": "simlex999", "path": "/path/to/simlex999.csv", "size": 999},
            ]
        )
        assert len(response.datasets) == 2


class TestSynsetInfoSchema:
    """Тесты схемы SynsetInfoSchema."""
    
    def test_synset_info_valid(self):
        """Тест: валидный синсет."""
        synset = SynsetInfoSchema(
            synset_id="123-N", title="кошка", gloss="домашнее животное", depth=8, ic=0.01
        )
        assert synset.synset_id == "123-N"
        assert synset.depth == 8


class TestWordNetSimilarityResponse:
    """Тесты схемы WordNetSimilarityResponse."""
    
    def test_wordnet_similarity_valid(self):
        """Тест: валидный ответ близости."""
        response = WordNetSimilarityResponse(
            term1="кот",
            term2="собака",
            similarity=0.78,
            algorithm="lin",
            synset1=SynsetInfoSchema(
                synset_id="1", title="кот", gloss="...", depth=5, ic=0.1
            ),
            synset2=SynsetInfoSchema(
                synset_id="2", title="собака", gloss="...", depth=5, ic=0.1
            ),
        )
        assert response.similarity == 0.78
        assert response.algorithm == "lin"


class TestHypernymsResponse:
    """Тесты схемы HypernymsResponse."""
    
    def test_hypernyms_valid(self):
        """Тест: валидный ответ."""
        response = HypernymsResponse(
            term="нейронная сеть",
            hypernyms=["нейронная сеть", "модель", "алгоритм", "абстракция"],
        )
        assert len(response.hypernyms) == 4


class TestBenchmarkMetricsSchema:
    """Тесты схемы BenchmarkMetricsSchema."""
    
    def test_metrics_valid(self):
        """Тест: валидные метрики."""
        metrics = BenchmarkMetricsSchema(spearman=0.65, pearson=0.68, missing=5)
        assert metrics.spearman == 0.65
        assert metrics.pearson == 0.68
        assert metrics.missing == 5


class TestBenchmarkMatrixCell:
    """Тесты схемы BenchmarkMatrixCell."""
    
    def test_matrix_cell_valid(self):
        """Тест: валидная ячейка."""
        cell = BenchmarkMatrixCell(spearman=0.65, pearson=0.68, missing=5, predictions=495)
        assert cell.predictions == 495


class TestBenchmarkMatrixRow:
    """Тесты схемы BenchmarkMatrixRow."""
    
    def test_matrix_row_valid(self):
        """Тест: валидная строка."""
        row = BenchmarkMatrixRow(
            method="SBERT",
            hj_rg=BenchmarkMatrixCell(
                spearman=0.65, pearson=0.68, missing=5, predictions=495
            ),
        )
        assert row.method == "SBERT"
        assert row.hj_rg is not None


class TestDatasetStatus:
    """Тесты схемы DatasetStatus."""
    
    def test_dataset_status_valid(self):
        """Тест: валидный статус."""
        status = DatasetStatus(status="completed", progress=1.0)
        assert status.status == "completed"
        assert status.progress == 1.0


class TestActiveTask:
    """Тесты схемы ActiveTask."""
    
    def test_active_task_valid(self):
        """Тест: валидная задача."""
        task = ActiveTask(id="1", dataset="hj-rg", method="sbert", progress=0.5)
        assert task.id == "1"
        assert task.status == "running"  # default


class TestSystemStateResponse:
    """Тесты схемы SystemStateResponse."""
    
    def test_system_state_ready(self):
        """Тест: готовый статус."""
        state = SystemStateResponse(
            system_status="ready",
            domains_loaded=True,
            domains_count=5,
            datasets={},
            active_tasks=[],
        )
        assert state.system_status == "ready"
        assert state.domains_loaded is True


class TestBenchmarkRunRequest:
    """Тесты схемы BenchmarkRunRequest."""
    
    def test_run_request_valid(self):
        """Тест: валидный запрос."""
        request = BenchmarkRunRequest(dataset="hj-rg")
        assert request.dataset == "hj-rg"
        assert request.method == "all"  # default
        assert request.force_recalculate is False


class TestBenchmarkRunResponse:
    """Тесты схемы BenchmarkRunResponse."""
    
    def test_run_response_success(self):
        """Тест: успешный ответ."""
        response = BenchmarkRunResponse(success=True, task_id="123")
        assert response.success is True
        assert response.task_id == "123"
    
    def test_run_response_error(self):
        """Тест: ответ с ошибкой."""
        response = BenchmarkRunResponse(success=False, error="Failed")
        assert response.error == "Failed"
