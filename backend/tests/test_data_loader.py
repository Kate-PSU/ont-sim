# backend/tests/test_data_loader.py
# Тесты для DataLoader
#
# Версия: 1.0
# Обновлено: 2026-04-06

"""
Тесты для загрузчика данных из CSV/JSON файлов.
"""

import pytest
import json
import tempfile
from pathlib import Path

from src.infrastructure.data_loader import (
    DataLoader,
    DataLoaderError,
    CSVDataLoader,
    JSONDataLoader,
    get_loader,
)
from src.domain import Term, Domain
from src.presentation.schemas import UploadDataRequest, DomainInput, TermInput


class TestCSVDataLoader:
    """Тесты для CSVDataLoader."""

    def test_load_csv_basic(self):
        """Тест: базовая загрузка CSV данных."""
        loader = CSVDataLoader()
        csv_content = """term,domain,frequency
машинное обучение,ML,15
нейронная сеть,ML,12
регрессия,ML,8
база данных,DB,10
SQL,DB,20
"""
        count = loader.load_string(csv_content)
        
        assert count == 5
        assert len(loader.domain_names) == 2
        assert "ML" in loader.domain_names
        assert "DB" in loader.domain_names

    def test_load_csv_without_frequency(self):
        """Тест: CSV без колонки frequency (по умолчанию 1)."""
        loader = CSVDataLoader()
        csv_content = """term,domain
термин1,domain1
термин2,domain1
"""
        count = loader.load_string(csv_content)
        
        assert count == 2
        ml_domain = loader._domains["domain1"]
        assert all(term.frequency == 1 for term in ml_domain.terms)

    def test_load_csv_domains_and_terms(self):
        """Тест: проверка структуры доменов и терминов."""
        loader = CSVDataLoader()
        csv_content = """term,domain,frequency
кот,животные,100
собака,животные,95
липа,растения,50
"""
        loader.load_string(csv_content)
        
        animals_domain = loader._domains["животные"]
        assert len(animals_domain.terms) == 2
        
        cat = next(t for t in animals_domain.terms if t.name == "кот")
        assert cat.frequency == 100
        assert cat.domain == "животные"

    def test_load_csv_missing_columns(self):
        """Тест: ошибка при отсутствии обязательных колонок."""
        loader = CSVDataLoader()
        csv_content = """word,category
кот,животные
"""
        
        with pytest.raises(DataLoaderError) as exc_info:
            loader.load_string(csv_content)
        assert "Отсутствуют обязательные колонки" in str(exc_info.value)

    def test_load_csv_empty_file(self):
        """Тест: ошибка при пустом CSV."""
        loader = CSVDataLoader()
        
        with pytest.raises(DataLoaderError) as exc_info:
            loader.load_string("")
        assert "пуст" in str(exc_info.value).lower() or "заголовков" in str(exc_info.value).lower()

    def test_load_csv_file(self):
        """Тест: загрузка из файла."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False
        ) as f:
            f.write("term,domain,frequency\n")
            f.write("python,programming,10\n")
            f.write("java,programming,8\n")
            temp_path = f.name
        
        try:
            loader = CSVDataLoader()
            count = loader.load_file(temp_path)
            assert count == 2
            assert "programming" in loader.domain_names
        finally:
            Path(temp_path).unlink()

    def test_load_csv_file_not_found(self):
        """Тест: ошибка при отсутствии файла."""
        loader = CSVDataLoader()
        
        with pytest.raises(DataLoaderError) as exc_info:
            loader.load_file("/nonexistent/path/file.csv")
        assert "не найден" in str(exc_info.value)

    def test_load_csv_skips_empty_rows(self):
        """Тест: пропуск пустых строк."""
        loader = CSVDataLoader()
        csv_content = """term,domain,frequency
term1,domain1,5

term2,domain1,10

"""
        count = loader.load_string(csv_content)
        assert count == 2


class TestJSONDataLoader:
    """Тесты для JSONDataLoader."""

    def test_load_json_basic(self):
        """Тест: базовая загрузка JSON данных."""
        loader = JSONDataLoader()
        json_content = {
            "domains": [
                {
                    "name": "ML",
                    "terms": [
                        {"name": "машинное обучение", "frequency": 15},
                        {"name": "нейронная сеть", "frequency": 12}
                    ]
                }
            ]
        }
        
        count = loader.load_string(json.dumps(json_content, ensure_ascii=False))
        
        assert count == 2
        assert len(loader.domain_names) == 1
        assert "ML" in loader.domain_names

    def test_load_json_multiple_domains(self):
        """Тест: несколько доменов в JSON."""
        loader = JSONDataLoader()
        json_content = {
            "domains": [
                {
                    "name": "math",
                    "terms": [
                        {"name": "алгебра", "frequency": 20}
                    ]
                },
                {
                    "name": "physics",
                    "terms": [
                        {"name": "механика", "frequency": 18}
                    ]
                }
            ]
        }
        
        loader.load_string(json.dumps(json_content, ensure_ascii=False))
        
        assert len(loader.domain_names) == 2
        assert "math" in loader.domain_names
        assert "physics" in loader.domain_names

    def test_load_json_from_request(self):
        """Тест: загрузка из UploadDataRequest."""
        loader = JSONDataLoader()
        request = UploadDataRequest(
            domains=[
                DomainInput(
                    name="test_domain",
                    terms=[
                        TermInput(name="term1", frequency=10),
                        TermInput(name="term2", frequency=20)
                    ]
                )
            ]
        )
        
        count = loader.load_from_request(request)
        
        assert count == 2
        assert "test_domain" in loader.domain_names

    def test_load_json_invalid_structure(self):
        """Тест: ошибка при невалидной структуре JSON."""
        loader = JSONDataLoader()
        json_content = {"wrong_key": []}
        
        with pytest.raises(DataLoaderError) as exc_info:
            loader.load_string(json.dumps(json_content))
        assert "domains" in str(exc_info.value)

    def test_load_json_invalid_json(self):
        """Тест: ошибка при некорректном JSON."""
        loader = JSONDataLoader()
        
        with pytest.raises(DataLoaderError) as exc_info:
            loader.load_string("not valid json {")
        assert "JSON" in str(exc_info.value)

    def test_load_json_domains_not_list(self):
        """Тест: ошибка когда domains не список."""
        loader = JSONDataLoader()
        json_content = {"domains": "not a list"}
        
        with pytest.raises(DataLoaderError) as exc_info:
            loader.load_string(json.dumps(json_content))
        assert "списком" in str(exc_info.value)


class TestDataLoaderBase:
    """Тесты для базового класса DataLoader."""

    def test_clear(self):
        """Тест: очистка загруженных данных."""
        loader = CSVDataLoader()
        loader.load_string("term,domain\nterm1,dom1")
        assert len(loader.domains) == 1
        
        loader.clear()
        assert len(loader.domains) == 0
        assert len(loader.domain_names) == 0

    def test_domain_property(self):
        """Тест: свойство domains возвращает объекты Domain."""
        loader = CSVDataLoader()
        loader.load_string("term,domain\nt1,d1\nt2,d2")
        
        domains = loader.domains
        assert all(isinstance(d, Domain) for d in domains)


class TestGetLoader:
    """Тесты для функции get_loader."""

    def test_get_csv_loader(self):
        """Тест: получение CSV загрузчика."""
        loader = get_loader("csv")
        assert isinstance(loader, CSVDataLoader)

    def test_get_json_loader(self):
        """Тест: получение JSON загрузчика."""
        loader = get_loader("json")
        assert isinstance(loader, JSONDataLoader)

    def test_get_loader_case_insensitive(self):
        """Тест: нечувствительность к регистру."""
        loader1 = get_loader("CSV")
        loader2 = get_loader("Json")
        assert isinstance(loader1, CSVDataLoader)
        assert isinstance(loader2, JSONDataLoader)

    def test_get_loader_unknown_format(self):
        """Тест: ошибка при неизвестном формате."""
        with pytest.raises(DataLoaderError) as exc_info:
            get_loader("xml")
        assert "Неизвестный формат" in str(exc_info.value)


class TestDataLoaderEdgeCases:
    """Тесты для граничных случаев."""

    def test_load_csv_whitespace_in_names(self):
        """Тест: обработка пробелов в названиях."""
        loader = CSVDataLoader()
        csv_content = """term,domain,frequency
  термин с пробелами  ,  domain with spaces  , 5
"""
        count = loader.load_string(csv_content)
        assert count == 1
        
        domain = loader.domains[0]
        assert domain.terms[0].name == "термин с пробелами"
        assert domain.name == "domain with spaces"

    def test_load_json_default_frequency(self):
        """Тест: частота по умолчанию в JSON."""
        loader = JSONDataLoader()
        json_content = {
            "domains": [
                {
                    "name": "d1",
                    "terms": [
                        {"name": "t1"}  # без frequency
                    ]
                }
            ]
        }
        
        loader.load_string(json.dumps(json_content))
        assert loader.domains[0].terms[0].frequency == 1
