# backend/src/infrastructure/data_loader.py
# Загрузчик данных из CSV/JSON
#
# Версия: 1.0
# Обновлено: 2026-04-06

"""
Загрузчик данных для терминов и доменов.

Поддерживаемые форматы:
- CSV: term,domain,frequency
- JSON: {"domains": [{"name": "...", "terms": [...]}]}
"""

import csv
import json
from io import StringIO
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from src.domain import Domain, Term
from src.presentation.schemas import DomainInput, TermInput, UploadDataRequest


class DataLoaderError(Exception):
    """Ошибка загрузки данных."""
    pass


class DataLoader:
    """Базовый загрузчик данных.
    
    Загружает термины и домены из файла, валидирует и преобразует
    в доменные сущности Domain/Term.
    """
    
    def __init__(self) -> None:
        """Инициализация загрузчика."""
        self._domains: dict[str, Domain] = {}
    
    @property
    def domains(self) -> list[Domain]:
        """Получить список всех загруженных доменов."""
        return list(self._domains.values())
    
    @property
    def domain_names(self) -> list[str]:
        """Получить список названий доменов."""
        return list(self._domains.keys())
    
    def clear(self) -> None:
        """Очистить загруженные данные."""
        self._domains.clear()
    
    def _add_term_to_domain(self, term: Term) -> None:
        """Добавить термин в домен.
        
        Args:
            term: Термин для добавления.
        """
        domain_name = term.domain
        if domain_name not in self._domains:
            self._domains[domain_name] = Domain(name=domain_name, terms=[])
        self._domains[domain_name].terms.append(term)
    
    def _validate_and_convert_terms(
        self,
        domain_input: DomainInput
    ) -> list[Term]:
        """Валидация и преобразование терминов.
        
        Args:
            domain_input: Входные данные домена.
            
        Returns:
            Список валидных терминов.
            
        Raises:
            DataLoaderError: Если данные невалидны.
        """
        terms = []
        for term_input in domain_input.terms:
            try:
                term = Term(
                    name=term_input.name,
                    domain=domain_input.name,
                    frequency=term_input.frequency
                )
                terms.append(term)
            except ValidationError as e:
                raise DataLoaderError(
                    f"Невалидный термин '{term_input.name}': {e}"
                )
        return terms


class CSVDataLoader(DataLoader):
    """Загрузчик данных из CSV файлов.
    
    Формат CSV:
        term,domain,frequency
        машинное обучение,ML,15
        нейронная сеть,ML,12
    """
    
    def load_file(self, file_path: str | Path) -> int:
        """Загрузить данные из CSV файла.
        
        Args:
            file_path: Путь к CSV файлу.
            
        Returns:
            Количество загруженных терминов.
            
        Raises:
            DataLoaderError: Если файл не найден или некорректен.
        """
        path = Path(file_path)
        if not path.exists():
            raise DataLoaderError(f"Файл не найден: {file_path}")
        
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise DataLoaderError(
                f"Не удалось декодировать файл: {file_path}"
            )
        
        return self._parse_csv_content(content)
    
    def load_string(self, csv_content: str) -> int:
        """Загрузить данные из CSV строки.
        
        Args:
            csv_content: Содержимое CSV файла.
            
        Returns:
            Количество загруженных терминов.
        """
        return self._parse_csv_content(csv_content)
    
    def _parse_csv_content(self, content: str) -> int:
        """Парсинг CSV содержимого.
        
        Args:
            content: Текст CSV файла.
            
        Returns:
            Количество загруженных терминов.
            
        Raises:
            DataLoaderError: При ошибках парсинга.
        """
        self.clear()
        terms_count = 0
        
        reader = csv.DictReader(StringIO(content))
        
        # Проверяем наличие обязательных колонок
        if reader.fieldnames is None:
            raise DataLoaderError("CSV файл пуст или не содержит заголовков")
        
        required_columns = {"term", "domain"}
        missing = required_columns - set(reader.fieldnames)
        if missing:
            raise DataLoaderError(
                f"Отсутствуют обязательные колонки: {missing}"
            )
        
        for row_num, row in enumerate(reader, start=2):
            try:
                # Защита от None значений
                term_name = (row.get("term") or "").strip()
                domain_name = (row.get("domain") or "").strip()
                
                if not term_name:
                    raise DataLoaderError(f"Строка {row_num}: пустой term")
                if not domain_name:
                    raise DataLoaderError(f"Строка {row_num}: пустой domain")
                
                # frequency опциональна, по умолчанию 1
                freq_str = row.get("frequency") or "1"
                try:
                    frequency = int(freq_str)
                except ValueError:
                    frequency = 1
                
                term = Term(
                    name=term_name,
                    domain=domain_name,
                    frequency=frequency
                )
                self._add_term_to_domain(term)
                terms_count += 1
                
            except DataLoaderError:
                raise
            except Exception as e:
                raise DataLoaderError(f"Строка {row_num}: {e}")
        
        return terms_count


class JSONDataLoader(DataLoader):
    """Загрузчик данных из JSON файлов.
    
    Формат JSON:
        {
          "domains": [
            {
              "name": "ML",
              "terms": [
                {"name": "машинное обучение", "frequency": 15}
              ]
            }
          ]
        }
    """
    
    def load_file(self, file_path: str | Path) -> int:
        """Загрузить данные из JSON файла.
        
        Args:
            file_path: Путь к JSON файлу.
            
        Returns:
            Количество загруженных терминов.
            
        Raises:
            DataLoaderError: Если файл не найден или некорректен.
        """
        path = Path(file_path)
        if not path.exists():
            raise DataLoaderError(f"Файл не найден: {file_path}")
        
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise DataLoaderError(
                f"Не удалось декодировать файл: {file_path}"
            )
        
        return self._parse_json_content(content)
    
    def load_string(self, json_content: str) -> int:
        """Загрузить данные из JSON строки.
        
        Args:
            json_content: Содержимое JSON файла.
            
        Returns:
            Количество загруженных терминов.
        """
        return self._parse_json_content(json_content)
    
    def load_from_request(self, request: UploadDataRequest) -> int:
        """Загрузить данные из валидного запроса.
        
        Args:
            request: Валидный запрос UploadDataRequest.
            
        Returns:
            Количество загруженных терминов.
        """
        self.clear()
        terms_count = 0
        
        for domain_input in request.domains:
            for term_input in domain_input.terms:
                term = Term(
                    name=term_input.name,
                    domain=domain_input.name,
                    frequency=term_input.frequency
                )
                self._add_term_to_domain(term)
                terms_count += 1
        
        return terms_count
    
    def _parse_json_content(self, content: str) -> int:
        """Парсинг JSON содержимого.
        
        Args:
            content: Текст JSON файла.
            
        Returns:
            Количество загруженных терминов.
            
        Raises:
            DataLoaderError: При ошибках парсинга.
        """
        self.clear()
        
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise DataLoaderError(f"Некорректный JSON: {e}")
        
        if not isinstance(data, dict):
            raise DataLoaderError("JSON должен быть объектом")
        
        if "domains" not in data:
            raise DataLoaderError("JSON должен содержать поле 'domains'")
        
        domains_data = data["domains"]
        if not isinstance(domains_data, list):
            raise DataLoaderError("'domains' должно быть списком")
        
        terms_count = 0
        for domain_num, domain_data in enumerate(domains_data):
            try:
                domain_input = DomainInput.model_validate(domain_data)
            except ValidationError as e:
                raise DataLoaderError(
                    f"Домен {domain_num}: невалидные данные: {e}"
                )
            
            terms = self._validate_and_convert_terms(domain_input)
            for term in terms:
                self._add_term_to_domain(term)
                terms_count += 1
        
        return terms_count


def get_loader(file_format: str) -> DataLoader:
    """Получить загрузчик по формату файла.
    
    Args:
        file_format: Формат файла ('csv' или 'json').
        
    Returns:
        Соответствующий загрузчик.
        
    Raises:
        DataLoaderError: При неизвестном формате.
    """
    format_lower = file_format.lower()
    if format_lower == "csv":
        return CSVDataLoader()
    elif format_lower == "json":
        return JSONDataLoader()
    else:
        raise DataLoaderError(
            f"Неизвестный формат: {file_format}. "
            "Поддерживаемые форматы: csv, json"
        )
