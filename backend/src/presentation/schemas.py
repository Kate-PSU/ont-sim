# backend/src/presentation/schemas.py
# Pydantic схемы для API
#
# Версия: 1.3
# Обновлено: 2026-04-10
# Изменения: вынесена функция create_examples в отдельную утилиту

"""
Схемы для валидации запросов и ответов API.
"""

from typing import Any
from pydantic import BaseModel, Field


def create_examples(example_data: Any) -> dict[str, Any]:
    """Создание конфигурации примеров для Pydantic схем.
    
    Используется для устранения дублирования model_config с json_schema_extra.
    
    Args:
        example_data: Данные примера (dict или list).
    
    Returns:
        Конфигурация для model_config.
    """
    if isinstance(example_data, list):
        return {"json_schema_extra": {"examples": example_data}}
    return {"json_schema_extra": {"examples": [example_data]}}


class HealthResponse(BaseModel):
    """Ответ проверки здоровья.
    
    Используется для мониторинга доступности сервиса.
    """
    status: str = Field(
        description="Статус сервиса",
        examples=["ok"]
    )
    
    model_config = create_examples({"status": "ok"})


class DomainsResponse(BaseModel):
    """Список доменов.
    
    Возвращает список названий загруженных предметных областей.
    """
    domains: list[str] = Field(
        description="Список названий доменов",
        examples=[["machine_learning", "deep_learning", "statistics", "biology"]]
    )
    
    model_config = create_examples({
        "domains": ["machine_learning", "deep_learning", "statistics"]
    })


class SimilarityResponse(BaseModel):
    """Ответ с близостью доменов.
    
    Возвращает значение семантической близости между двумя доменами.
    """
    domain1: str = Field(
        description="Первый домен",
        examples=["ML", "machine_learning"]
    )
    domain2: str = Field(
        description="Второй домен",
        examples=["DL", "deep_learning"]
    )
    score: float = Field(
        description="Значение близости [0, 1]",
        ge=0.0,
        le=1.0,
        examples=[0.847, 0.65]
    )
    metric: str = Field(
        description="Используемая метрика",
        examples=["cosine", "euclidean"]
    )
    
    model_config = create_examples({
        "domain1": "ML",
        "domain2": "DL",
        "score": 0.847,
        "metric": "cosine"
    })


class GraphNode(BaseModel):
    """Узел графа.
    
    Представляет домен или термин в графе связей.
    """
    id: str = Field(
        description="Идентификатор узла",
        examples=["ML", "computer_science"]
    )
    label: str = Field(
        description="Метка узла для отображения",
        examples=["Machine Learning", "ML"]
    )
    type: str = Field(
        default="domain",
        description="Тип узла: domain или term",
        examples=["domain", "term"]
    )


class GraphNodeDetailed(GraphNode):
    """Детальный узел графа с информацией о родителе.
    
    Расширяет GraphNode информацией о родительском домене
    для узлов-терминов.
    """
    parent: str | None = Field(
        default=None,
        description="Родительский домен (для терминов)",
        examples=[None, "ML", "machine_learning"]
    )


class GraphEdge(BaseModel):
    """Ребро графа.
    
    Представляет связь между узлами с весом близости.
    """
    source: str = Field(
        description="Источник ребра",
        examples=["ML", "computer_science"]
    )
    target: str = Field(
        description="Цель ребра",
        examples=["DL", "biology"]
    )
    weight: float = Field(
        description="Вес ребра (близость)",
        ge=0.0,
        le=1.0,
        examples=[0.72, 0.45]
    )
    type: str = Field(
        default="similarity",
        description="Тип ребра: similarity или belongs_to",
        examples=["similarity", "belongs_to"]
    )


class GraphResponse(BaseModel):
    """Ответ с данными графа.
    
    Возвращает полный граф связей между доменами.
    """
    nodes: list[GraphNode] = Field(
        description="Список узлов",
        examples=[[
            {"id": "ML", "label": "ML", "type": "domain"},
            {"id": "DL", "label": "DL", "type": "domain"}
        ]]
    )
    edges: list[GraphEdge] = Field(
        description="Список рёбер",
        examples=[[
            {"source": "ML", "target": "DL", "weight": 0.72, "type": "similarity"}
        ]]
    )
    threshold: float = Field(
        description="Используемый порог",
        examples=[0.5, 0.6]
    )
    
    model_config = create_examples({
        "nodes": [
            {"id": "ML", "label": "ML", "type": "domain"},
            {"id": "DL", "label": "DL", "type": "domain"}
        ],
        "edges": [
            {"source": "ML", "target": "DL", "weight": 0.72, "type": "similarity"}
        ],
        "threshold": 0.5
    })


class GraphResponseDetailed(GraphResponse):
    """Ответ с детальными данными графа (включая термины).
    
    Расширяет GraphResponse, включая термины как отдельные узлы.
    """
    nodes: list[GraphNodeDetailed] = Field(
        description="Список узлов (домены + термины)"
    )


class SimilarityRequest(BaseModel):
    """Запрос на расчёт близости (не используется напрямую).
    
    Схема для будущего POST эндпоинта расчёта близости.
    """
    domain1: str = Field(
        description="Первый домен",
        examples=["ML", "machine_learning"]
    )
    domain2: str = Field(
        description="Второй домен",
        examples=["DL", "deep_learning"]
    )
    metric: str = Field(
        default="cosine",
        description="Метрика близости: cosine или euclidean",
        examples=["cosine"]
    )


# ==== Схемы для загрузки данных ====

class TermInput(BaseModel):
    """Входные данные термина.
    
    Используется при загрузке данных о предметных областях.
    """
    name: str = Field(
        description="Название термина",
        min_length=1,
        examples=["нейронная сеть", "gradient descent"]
    )
    frequency: int = Field(
        default=1,
        description="Частота в корпусе",
        ge=0,
        examples=[10, 5, 1]
    )
    
    model_config = create_examples({"name": "нейронная сеть", "frequency": 10})


class DomainInput(BaseModel):
    """Входные данные домена.
    
    Описывает предметную область с набором терминов.
    """
    name: str = Field(
        description="Название домена",
        min_length=1,
        examples=["ML", "machine_learning"]
    )
    terms: list[TermInput] = Field(
        description="Список терминов домена",
        examples=[[
            {"name": "нейронная сеть", "frequency": 10},
            {"name": "gradient descent", "frequency": 5}
        ]]
    )


class UploadDataRequest(BaseModel):
    """Запрос на загрузку данных (JSON формат).
    
    Содержит список доменов с их терминами.
    """
    domains: list[DomainInput] = Field(
        description="Список доменов с терминами",
        min_length=1,
        examples=[[
            {
                "name": "ML",
                "terms": [
                    {"name": "нейронная сеть", "frequency": 10},
                    {"name": "gradient descent", "frequency": 5}
                ]
            }
        ]]
    )
    
    model_config = create_examples({
        "domains": [
            {
                "name": "ML",
                "terms": [
                    {"name": "нейронная сеть", "frequency": 10},
                    {"name": "gradient descent", "frequency": 5}
                ]
            }
        ]
    })


class UploadDataResponse(BaseModel):
    """Ответ загрузки данных.
    
    Возвращает результат операции загрузки данных.
    """
    success: bool = Field(
        description="Успешность операции",
        examples=[True, False]
    )
    domains_loaded: int = Field(
        description="Количество загруженных доменов",
        ge=0,
        examples=[1, 5, 10]
    )
    terms_loaded: int = Field(
        description="Количество загруженных терминов",
        ge=0,
        examples=[10, 50, 100]
    )
    message: str = Field(
        description="Сообщение о результате",
        examples=["Загружено 10 терминов из 2 доменов"]
    )
    
    model_config = create_examples({
        "success": True,
        "domains_loaded": 2,
        "terms_loaded": 10,
        "message": "Загружено 10 терминов из 2 доменов"
    })


# ==== Схемы для RuWordNet ====

class SynsetInfoSchema(BaseModel):
    """Информация о синсете.
    
    Содержит данные о синсете (синонимическом множестве) из WordNet.
    """
    synset_id: str = Field(
        description="ID синсета",
        examples=["123-N", "45678-N"]
    )
    title: str = Field(
        description="Название синсета",
        examples=["кошка", "нейронная сеть"]
    )
    gloss: str = Field(
        description="Глосса (краткое определение)",
        examples=["домашнее животное семейства кошачьих"]
    )
    depth: int = Field(
        description="Глубина в иерархии",
        ge=0,
        examples=[8, 6, 10]
    )
    ic: float = Field(
        description="Information Content (содержание информации)",
        ge=0.0,
        examples=[0.01, 0.05, 0.1]
    )
    
    model_config = create_examples({
        "synset_id": "123-N",
        "title": "кошка",
        "gloss": "домашнее животное семейства кошачьих",
        "depth": 8,
        "ic": 0.01
    })


class WordNetSimilarityResponse(BaseModel):
    """Ответ с близостью терминов через WordNet.
    
    Возвращает значение близости и информацию о синсетах.
    """
    term1: str = Field(
        description="Первый термин",
        examples=["кот", "нейронная сеть"]
    )
    term2: str = Field(
        description="Второй термин",
        examples=["собака", "perceptron"]
    )
    similarity: float = Field(
        description="Значение близости [0, 1]",
        ge=0.0,
        le=1.0,
        examples=[0.78, 0.65]
    )
    algorithm: str = Field(
        description="Используемый алгоритм",
        examples=["lin", "wup", "path"]
    )
    synset1: SynsetInfoSchema | None = Field(
        default=None,
        description="Синсет первого термина"
    )
    synset2: SynsetInfoSchema | None = Field(
        default=None,
        description="Синсет второго термина"
    )
    lcs: SynsetInfoSchema | None = Field(
        default=None,
        description="Lowest Common Subsumer (ближайший общий предок)"
    )
    
    model_config = create_examples({
        "term1": "кот",
        "term2": "собака",
        "similarity": 0.78,
        "algorithm": "lin",
        "synset1": {
            "synset_id": "123-N",
            "title": "кошка",
            "gloss": "домашнее животное...",
            "depth": 8,
            "ic": 0.01
        },
        "synset2": {
            "synset_id": "124-N",
            "title": "собака",
            "gloss": "домашнее животное...",
            "depth": 8,
            "ic": 0.01
        },
        "lcs": {
            "synset_id": "100-N",
            "title": "млекопитающее",
            "gloss": "...",
            "depth": 6,
            "ic": 0.05
        }
    })


class WordNetDomainSimilarityResponse(BaseModel):
    """Ответ с близостью доменов через WordNet.
    
    Возвращает близость между двумя доменами на основе
    попарного сравнения их терминов через WordNet.
    """
    domain1: str = Field(
        description="Первый домен",
        examples=["ML", "machine_learning"]
    )
    domain2: str = Field(
        description="Второй домен",
        examples=["DL", "deep_learning"]
    )
    similarity: float = Field(
        description="Значение близости [0, 1]",
        ge=0.0,
        le=1.0,
        examples=[0.65, 0.42]
    )
    algorithm: str = Field(
        description="Используемый алгоритм",
        examples=["lin", "wup", "path"]
    )
    pairs_count: int = Field(
        description="Количество проверенных пар терминов",
        ge=0,
        examples=[100, 500]
    )
    aggregation: str = Field(
        description="Метод агрегации",
        examples=["max", "mean", "min"]
    )
    
    model_config = create_examples({
        "domain1": "ML",
        "domain2": "DL",
        "similarity": 0.65,
        "algorithm": "lin",
        "pairs_count": 100,
        "aggregation": "max"
    })


class HypernymsResponse(BaseModel):
    """Ответ с иерархией гиперонимов.
    
    Возвращает цепочку обобщений термина.
    """
    term: str = Field(
        description="Термин",
        examples=["нейронная сеть", "кот"]
    )
    hypernyms: list[str] = Field(
        description="Иерархия гиперонимов от частного к общему",
        examples=[["нейронная сеть", "модель", "алгоритм", "абстракция", "сущность"]]
    )
    
    model_config = create_examples({
        "term": "нейронная сеть",
        "hypernyms": ["нейронная сеть", "модель", "алгоритм", "абстракция", "сущность"]
    })


# ==== Схемы для бенчмаркинга ====

class BenchmarkRequest(BaseModel):
    """Запрос на запуск бенчмарка.
    
    Указывает датасет и опционально методы для тестирования.
    """
    dataset: str = Field(
        description="Название датасета: hj-rg, simlex999, simlex999_rus или путь к CSV",
        examples=["hj-rg", "simlex999", "/path/to/dataset.csv"]
    )
    methods: list[str] | None = Field(
        default=None,
        description="Список методов для тестирования (по умолчанию все)",
        examples=[["sbert", "sbert_tfidf"], None]
    )
    
    model_config = create_examples([
        {"dataset": "hj-rg"},
        {"dataset": "hj-rg", "methods": ["sbert", "sbert_tfidf"]}
    ])


class MethodResultSchema(BaseModel):
    """Результат одного метода.
    
    Содержит метрики качества для одного метода семантической близости.
    """
    method: str = Field(
        description="Название метода",
        examples=["SBERT (baseline)", "RuWordNet (Lin)", "SBERT + Z-score"]
    )
    spearman: float = Field(
        description="Корреляция Спирмена (rank correlation)",
        ge=-1.0,
        le=1.0,
        examples=[0.65, 0.42, 0.78]
    )
    pearson: float = Field(
        description="Корреляция Пирсона (linear correlation)",
        ge=-1.0,
        le=1.0,
        examples=[0.68, 0.45, 0.80]
    )
    mse: float = Field(
        description="Среднеквадратичная ошибка",
        ge=0.0,
        examples=[0.15, 0.32, 0.08]
    )
    missing: int = Field(
        description="Количество пропущенных пар (не найдены в WordNet)",
        ge=0,
        examples=[5, 100, 0]
    )
    predictions_count: int = Field(
        description="Количество успешных предсказаний",
        ge=0,
        examples=[495, 400, 999]
    )
    
    model_config = create_examples({
        "method": "SBERT (baseline)",
        "spearman": 0.65,
        "pearson": 0.68,
        "mse": 0.15,
        "missing": 5,
        "predictions_count": 495
    })


class BenchmarkComparisonSchema(BaseModel):
    """Сравнение всех методов на бенчмарке.
    
    Содержит результаты всех протестированных методов.
    """
    dataset_name: str = Field(
        description="Название датасета",
        examples=["hj-rg", "simlex999", "simlex999_rus"]
    )
    dataset_size: int = Field(
        description="Размер датасета (количество пар)",
        ge=0,
        examples=[500, 999]
    )
    execution_time_sec: float = Field(
        description="Время выполнения в секундах",
        ge=0.0,
        examples=[45.2, 120.5]
    )
    results: list[MethodResultSchema] = Field(
        description="Результаты методов",
        examples=[[
            {
                "method": "SBERT (baseline)",
                "spearman": 0.65,
                "pearson": 0.68,
                "mse": 0.15,
                "missing": 5,
                "predictions_count": 495
            }
        ]]
    )
    
    model_config = create_examples({
        "dataset_name": "hj-rg",
        "dataset_size": 500,
        "execution_time_sec": 45.2,
        "results": [
            {
                "method": "SBERT (baseline)",
                "spearman": 0.65,
                "pearson": 0.68,
                "mse": 0.15,
                "missing": 5,
                "predictions_count": 495
            }
        ]
    })


class BenchmarkResponse(BaseModel):
    """Ответ с результатами бенчмарка.
    
    Возвращает результаты сравнения методов или информацию об ошибке.
    """
    success: bool = Field(
        description="Успешность выполнения",
        examples=[True, False]
    )
    comparison: BenchmarkComparisonSchema | None = Field(
        default=None,
        description="Сравнение методов"
    )
    error: str | None = Field(
        default=None,
        description="Ошибка при выполнении",
        examples=[None, "Бенчмарк уже выполняется"]
    )
    
    model_config = create_examples([
        {
            "success": True,
            "comparison": {
                "dataset_name": "hj-rg",
                "dataset_size": 500,
                "execution_time_sec": 45.2,
                "results": [
                    {
                        "method": "SBERT (baseline)",
                        "spearman": 0.65,
                        "pearson": 0.68,
                        "mse": 0.15,
                        "missing": 5,
                        "predictions_count": 495
                    }
                ]
            }
        },
        {
            "success": False,
            "error": "Бенчмарк уже выполняется. Подождите завершения текущей операции."
        }
    ])


class BenchmarkDatasetsResponse(BaseModel):
    """Список доступных датасетов.
    
    Возвращает информацию о доступных бенчмарк-датасетах.
    """
    datasets: list[dict[str, str | int]] = Field(
        description="Список датасетов с информацией",
        examples=[[
            {"name": "hj-rg", "path": "/path/to/hj-rg.csv", "size": 500},
            {"name": "simlex999", "path": "/path/to/simlex999.csv", "size": 999}
        ]]
    )
    
    model_config = create_examples({
        "datasets": [
            {"name": "hj-rg", "path": "/path/to/hj-rg.csv", "size": 500},
            {"name": "simlex999", "path": "/path/to/simlex999.csv", "size": 999}
        ]
    })


# ==== Схемы для обогащённых эмбеддингов ====

class EnrichmentInfoResponse(BaseModel):
    """Информация об обогащении термина.
    
    Возвращает информацию о гиперонимах и статусе обогащения.
    """
    term: str = Field(
        description="Исходный термин",
        examples=["нейронная сеть", "gradient descent"]
    )
    hypernyms: list[str] = Field(
        description="Найденные гиперонимы из RuWordNet",
        examples=[["сеть", "модель", "алгоритм"]]
    )
    hypernym_count: int = Field(
        description="Количество гиперонимов",
        ge=0,
        examples=[3, 5, 0]
    )
    enriched: bool = Field(
        description="True если обогащение применимо (найдены гиперонимы)",
        examples=[True, False]
    )
    
    model_config = create_examples({
        "term": "нейронная сеть",
        "hypernyms": ["сеть", "модель", "алгоритм"],
        "hypernym_count": 3,
        "enriched": True
    })


class EnrichedSimilarityResponse(BaseModel):
    """Ответ с близостью доменов с обогащёнными эмбеддингами.
    
    Возвращает близость с учётом гиперонимов из RuWordNet.
    """
    domain1: str = Field(
        description="Первый домен",
        examples=["ML", "machine_learning"]
    )
    domain2: str = Field(
        description="Второй домен",
        examples=["DL", "deep_learning"]
    )
    score: float = Field(
        description="Значение близости [0, 1]",
        ge=0.0,
        le=1.0,
        examples=[0.72, 0.58]
    )
    metric: str = Field(
        description="Используемая метрика",
        examples=["cosine", "euclidean"]
    )
    alpha: float = Field(
        description="Вес оригинального термина",
        ge=0.0,
        le=1.0,
        examples=[0.7, 0.5, 1.0]
    )
    terms_enriched: int = Field(
        description="Количество обогащённых терминов",
        ge=0,
        examples=[15, 30]
    )
    terms_total: int = Field(
        description="Общее количество терминов",
        ge=0,
        examples=[20, 50]
    )
    
    model_config = create_examples({
        "domain1": "ML",
        "domain2": "DL",
        "score": 0.72,
        "metric": "cosine",
        "alpha": 0.7,
        "terms_enriched": 15,
        "terms_total": 20
    })


# ==== Схемы для интегральной матрицы бенчмарков ====

class BenchmarkMetricsSchema(BaseModel):
    """Базовые метрики качества для методов семантической близости.
    
    Содержит корреляции Спирмена и Пирсона, а также информацию о пропусках.
    """
    spearman: float = Field(
        description="Корреляция Спирмена (rank correlation)",
        ge=-1.0,
        le=1.0,
        examples=[0.65, 0.42]
    )
    pearson: float = Field(
        description="Корреляция Пирсона (linear correlation)",
        ge=-1.0,
        le=1.0,
        examples=[0.68, 0.45]
    )
    missing: int = Field(
        description="Количество пропущенных пар (не найдены в WordNet)",
        ge=0,
        examples=[5, 100]
    )
    
    model_config = create_examples({
        "spearman": 0.65,
        "pearson": 0.68,
        "missing": 5
    })


class BenchmarkMatrixCell(BenchmarkMetricsSchema):
    """Ячейка интегральной матрицы бенчмарков.
    
    Содержит метрики одного метода на одном датасете.
    """
    predictions: int = Field(
        description="Количество предсказаний",
        ge=0,
        examples=[495, 400]
    )
    
    model_config = create_examples({
        "spearman": 0.65,
        "pearson": 0.68,
        "missing": 5,
        "predictions": 495
    })


class BenchmarkMatrixRow(BaseModel):
    """Строка интегральной матрицы (один метод на всех датасетах).
    
    Поддерживает 3 датасета:
    - hj_rg: Русский бенчмарк
    - simlex999: English SimLex-999
    - simlex999_rus: Russian SimLex-999
    """
    method: str = Field(
        description="Название метода",
        examples=["SBERT (baseline)", "RuWordNet (Lin)"]
    )
    hj_rg: BenchmarkMatrixCell | None = Field(
        default=None,
        description="Результат на hj-rg (Russian)"
    )
    simlex999: BenchmarkMatrixCell | None = Field(
        default=None,
        description="Результат на SimLex-999 (English)"
    )
    simlex999_rus: BenchmarkMatrixCell | None = Field(
        default=None,
        description="Результат на SimLex-999 (Russian)"
    )
    
    model_config = create_examples({
        "method": "SBERT (baseline)",
        "hj_rg": {
            "spearman": 0.65,
            "pearson": 0.68,
            "missing": 5,
            "predictions": 495
        }
    })


class BenchmarkMatrixResponse(BaseModel):
    """Интегральная матрица результатов бенчмарков.
    
    Возвращает матрицу метод × датасет со всеми метриками.
    """
    success: bool = Field(
        description="Успешность выполнения",
        examples=[True, False]
    )
    results: list[BenchmarkMatrixRow] = Field(
        description="Строки матрицы (методы)"
    )
    execution_time_sec: float = Field(
        description="Общее время выполнения",
        ge=0.0,
        examples=[120.5, 300.0]
    )
    error: str | None = Field(
        default=None,
        description="Ошибка при выполнении",
        examples=[None, "Бенчмарк уже выполняется"]
    )
    
    model_config = create_examples({
        "success": True,
        "results": [
            {
                "method": "SBERT (baseline)",
                "hj_rg": {
                    "spearman": 0.65,
                    "pearson": 0.68,
                    "missing": 5,
                    "predictions": 495
                }
            }
        ],
        "execution_time_sec": 120.5
    })


# ==== Схемы для единой точки входа ====

class DatasetStatus(BaseModel):
    """Статус датасета."""
    status: str = Field(description="pending | running | completed | failed")
    progress: float = Field(default=0.0, description="Прогресс 0.0-1.0")
    results: dict[str, MethodResultSchema] | None = Field(default=None, description="Результаты по методам")
    saved_at: str | None = Field(default=None, description="Время сохранения")
    error: str | None = Field(default=None, description="Ошибка")


class ActiveTask(BaseModel):
    """Активная задача."""
    id: str = Field(description="ID задачи")
    dataset: str = Field(description="Датасет")
    method: str = Field(description="Метод")
    progress: float = Field(default=0.0, description="Прогресс 0.0-1.0")
    status: str = Field(default="running", description="Статус")


class SystemStateResponse(BaseModel):
    """Состояние системы - единая точка входа."""
    system_status: str = Field(description="ready | busy | error")
    busy_reason: str | None = Field(default=None, description="Причина занятости")
    domains_loaded: bool = Field(description="Загружены ли домены")
    domains_count: int = Field(default=0, description="Количество доменов")
    datasets: dict[str, DatasetStatus] = Field(description="Статусы датасетов")
    active_tasks: list[ActiveTask] = Field(default=[], description="Активные задачи")


class BenchmarkRunRequest(BaseModel):
    """Запрос на запуск бенчмарка."""
    dataset: str = Field(description="Название датасета: hj-rg, simlex999, simlex999_rus")
    method: str = Field(
        default="all",
        description="Метод: sbert, sbert_tfidf, sbert_zscore, sbert_filter, wordnet_lin, wordnet_wup, hybrid, bertopic, doc2vec, lda, all"
    )
    force_recalculate: bool = Field(default=False, description="Игнорировать кэш")


class BenchmarkRunResponse(BaseModel):
    """Ответ на запуск бенчмарка."""
    success: bool = Field(description="Успешность")
    task_id: str | None = Field(default=None, description="ID задачи")
    error: str | None = Field(default=None, description="Ошибка")
    active_task: ActiveTask | None = Field(default=None, description="Активная задача")


# ==== Схемы для Wikipedia similarity ====

class WikipediaSimilarityResponse(BaseModel):
    """Ответ с матрицей близости доменов на основе Wikipedia.
    
    Возвращает матрицу близости между доменами, вычисленную
    с помощью zero-shot классификации на Wikipedia данных.
    """
    domains: list[str] = Field(
        description="Список доменов",
        examples=[["Machine Learning", "Deep Learning", "Statistics"]]
    )
    matrix: list[list[float]] = Field(
        description="Матрица близости (N×N)",
        examples=[[[1.0, 0.8, 0.6], [0.8, 1.0, 0.7], [0.6, 0.7, 1.0]]]
    )
    top_pairs: list[dict] = Field(
        description="Топ пар доменов по близости",
        examples=[[{"d1": "ML", "d2": "DL", "score": 0.85}]]
    )
    
    model_config = create_examples({
        "domains": ["Machine Learning", "Deep Learning", "Statistics"],
        "matrix": [[1.0, 0.8, 0.6], [0.8, 1.0, 0.7], [0.6, 0.7, 1.0]],
        "top_pairs": [{"d1": "Machine Learning", "d2": "Deep Learning", "score": 0.85}]
    })
