# backend/src/infrastructure/en_wordnet_service.py
# Сервис семантической близости на основе English WordNet (NLTK)
#
# Версия: 1.1
# Обновлено: 2026-04-10
# Изменения: вынесена параметризованная функция _traverse_hypernyms

"""
Модуль для расчёта семантической близости терминов через English WordNet.

Алгоритмы:
- Lin similarity: 2 * IC(lcs) / (IC(a) + IC(b))
- Wu-Palmer: 2 * depth(lcs) / (depth(a) + depth(b))
- Shortest path: расстояние между синсетами в графе

Зависимости:
- nltk>=3.8.1
- wordnet, omw-1.4 (скачиваются через nltk.download)
"""

from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional, TypeVar

import numpy as np

try:
    import nltk
    from nltk.corpus import wordnet
    NLTK_WORDNET_AVAILABLE = True
except ImportError:
    NLTK_WORDNET_AVAILABLE = False


# Типовые переменные для generic-функций
T = TypeVar('T')
SynsetType = TypeVar('SynsetType')


@dataclass
class EnSynsetInfo:
    """Информация о синсете для отладки."""
    synset_id: str
    name: str
    definition: str
    depth: int
    ic: float  # Information Content


@dataclass
class EnSimilarityResult:
    """Результат расчёта близости между терминами."""
    term1: str
    term2: str
    similarity: float
    algorithm: str
    synset1: Optional[EnSynsetInfo]
    synset2: Optional[EnSynsetInfo]
    lcs: Optional[EnSynsetInfo]  # Lowest Common Subsumer


class EnglishWordNetError(Exception):
    """Ошибка сервиса English WordNet."""
    pass


class EnglishWordNetNotInitializedError(EnglishWordNetError):
    """English WordNet не инициализирован."""
    pass


class EnglishWordNetService:
    """Сервис расчёта семантической близости через English WordNet (NLTK).
    
    Использует иерархию синсетов для расчёта близости между терминами
    с помощью алгоритмов Lin, Wu-Palmer и Shortest Path.
    
    Атрибуты:
        _initialized: Флаг инициализации.
        _ic_cache: Кеш Information Content для синсетов.
        _depth_cache: Кеш глубины синсетов.
    """
    
    def __init__(self) -> None:
        """Инициализация сервиса."""
        self._initialized = False
        self._ic_cache: dict[str, float] = {}
        self._depth_cache: dict[str, int] = {}
    
    def initialize(self) -> None:
        """Инициализация подключения к English WordNet.
        
        Загружает необходимые данные NLTK (wordnet, omw-1.4).
        """
        if not NLTK_WORDNET_AVAILABLE:
            raise ImportError(
                "nltk не установлен. Установите: pip install nltk"
            )
        
        try:
            # Загружаем wordnet данные если их нет
            try:
                wordnet.synsets('dog')
            except LookupError:
                nltk.download('wordnet', quiet=True)
            
            try:
                wordnet.synsets('cat')
            except LookupError:
                nltk.download('omw-1.4', quiet=True)
            
            self._initialized = True
        except Exception as e:
            raise EnglishWordNetError(f"Не удалось инициализировать English WordNet: {e}")
    
    def _traverse_hypernyms(
        self,
        synset,
        accumulator: Callable[[int, str], int],
        initial: int = 0
    ) -> int:
        """Обход иерархии гиперонимов с накоплением значения.
        
        Параметризованная функция для устранения дублирования кода
        в _get_depth и get_hypernyms.
        
        Args:
            synset: Начальный синсет.
            accumulator: Функция (текущее_значение, имя_гиперонима) -> новое_значение.
            initial: Начальное значение аккумулятора.
        
        Returns:
            Результат накопления.
        """
        result = initial
        current = synset
        visited = set()
        
        while current.hypernyms():
            next_hyper = None
            for h in current.hypernyms():
                if h.name() not in visited:
                    next_hyper = h
                    visited.add(h.name())
                    break
            
            if next_hyper is None:
                break
            
            current = next_hyper
            result = accumulator(result, current.name())
        
        return result
    
    def _get_depth(self, synset) -> int:
        """Получение глубины синсета в иерархии.
        
        Args:
            synset: Синсет WordNet.
        
        Returns:
            Глубина (количество рёбер до корня).
        """
        cache_key = synset.name()
        if cache_key in self._depth_cache:
            return self._depth_cache[cache_key]
        
        depth = self._traverse_hypernyms(
            synset,
            accumulator=lambda d, _: d + 1,
            initial=0
        )
        
        self._depth_cache[cache_key] = depth
        return depth
    
    def _compute_ic(self, synset) -> float:
        """Вычисление Information Content синсета.
        
        IC = -log(p(synset)), где p — вероятность синсета.
        
        Args:
            synset: Синсет WordNet.
        
        Returns:
            Значение IC в диапазоне [0, 1].
        """
        cache_key = synset.name()
        if cache_key in self._ic_cache:
            return self._ic_cache[cache_key]
        
        # Количество лемм у синсета как мера специфичности
        synset_lemmas = len(synset.lemmas()) if synset.lemmas() else 1
        
        # IC пропорционален количеству лемм (более специфичные = больше IC)
        # Нормализуем: больше лемм = выше IC
        ic = np.log2(1 + synset_lemmas) / np.log2(1 + 100)  # Нормализация
        
        self._ic_cache[cache_key] = ic
        return ic
    
    def _get_synsets_for_term(self, term: str):
        """Получение всех синсетов для термина.
        
        Args:
            term: Термин для поиска.
        
        Returns:
            Список синсетов.
        """
        return wordnet.synsets(term.lower().strip())
    
    def _get_best_synset(self, term: str):
        """Получение первого (наиболее частотного) синсета для термина.
        
        Args:
            term: Термин.
        
        Returns:
            Первый синсет или None.
        """
        synsets = self._get_synsets_for_term(term)
        if synsets:
            return synsets[0]
        return None
    
    def _find_lcs(self, synset1, synset2):
        """Поиск Lowest Common Subsumer (LCS).
        
        Находит ближайшего общего предка в иерархии.
        
        Args:
            synset1: Первый синсет.
            synset2: Второй синсет.
        
        Returns:
            LCS синсет или None.
        """
        if synset1 is None or synset2 is None:
            return None
        
        # Используем встроенный метод NLTK
        lcs_sims = synset1.lowest_common_hypernyms(synset2)
        if lcs_sims:
            return lcs_sims[0]
        return None
    
    def _shortest_path(self, synset1, synset2) -> int:
        """Расчёт кратчайшего пути между синсетами.
        
        Args:
            synset1: Первый синсет.
            synset2: Второй синсет.
        
        Returns:
            Количество рёбер между синсетами.
        """
        if synset1 is None or synset2 is None:
            return -1
        
        if synset1.name() == synset2.name():
            return 0
        
        # BFS от synset1
        queue = deque([(synset1, 0)])
        visited = {synset1.name()}
        
        while queue:
            current, distance = queue.popleft()
            
            if current.name() == synset2.name():
                return distance
            
            # Проверяем гиперонимы и гипонимы
            neighbors = []
            if current.hypernyms():
                neighbors.extend(current.hypernyms())
            if current.hyponyms():
                neighbors.extend(current.hyponyms())
            
            for neighbor in neighbors:
                if neighbor.name() not in visited:
                    visited.add(neighbor.name())
                    queue.append((neighbor, distance + 1))
        
        return -1  # Не найдено
    
    def get_similarity(
        self,
        term1: str,
        term2: str,
        algorithm: str = "lin",
    ) -> EnSimilarityResult:
        """Расчёт семантической близости между терминами.
        
        Args:
            term1: Первый термин.
            term2: Второй термин.
            algorithm: Алгоритм ('lin', 'wup', 'path').
        
        Returns:
            Результат с мерой близости.
        """
        if not self._initialized:
            raise EnglishWordNetNotInitializedError()
        
        synset1 = self._get_best_synset(term1)
        synset2 = self._get_best_synset(term2)
        
        similarity = 0.0
        lcs = None
        
        if synset1 is None or synset2 is None:
            # Один из терминов не найден в WordNet
            return EnSimilarityResult(
                term1=term1,
                term2=term2,
                similarity=0.0,
                algorithm=algorithm,
                synset1=self._synset_info(synset1) if synset1 else None,
                synset2=self._synset_info(synset2) if synset2 else None,
                lcs=None,
            )
        
        if algorithm == "lin":
            similarity = self._lin_similarity(synset1, synset2)
        elif algorithm == "wup":
            similarity = self._wu_palmer(synset1, synset2)
        elif algorithm == "path":
            similarity = self._path_similarity(synset1, synset2)
        else:
            raise ValueError(f"Неизвестный алгоритм: {algorithm}")
        
        return EnSimilarityResult(
            term1=term1,
            term2=term2,
            similarity=similarity,
            algorithm=algorithm,
            synset1=self._synset_info(synset1),
            synset2=self._synset_info(synset2),
            lcs=self._synset_info(lcs) if lcs else None,
        )
    
    def _synset_info(self, synset) -> Optional[EnSynsetInfo]:
        """Получение информации о синсете."""
        if synset is None:
            return None
        return EnSynsetInfo(
            synset_id=synset.name(),
            name=synset.name().split('.')[0],  # Имя без pos и номера
            definition=synset.definition(),
            depth=self._get_depth(synset),
            ic=self._compute_ic(synset),
        )
    
    def _lin_similarity(self, synset1, synset2) -> float:
        """Расчёт Lin similarity.
        
        Формула: 2 * IC(LCS) / (IC(a) + IC(b))
        
        Args:
            synset1: Первый синсет.
            synset2: Второй синсет.
        
        Returns:
            Значение близости [0, 1].
        """
        lcs = self._find_lcs(synset1, synset2)
        
        if lcs is None:
            return 0.0
        
        ic_lcs = self._compute_ic(lcs)
        ic_a = self._compute_ic(synset1)
        ic_b = self._compute_ic(synset2)
        
        sum_ic = ic_a + ic_b
        if sum_ic == 0:
            return 0.0
        
        # Lin similarity может быть > 1, ограничиваем [0, 1]
        similarity = 2 * ic_lcs / sum_ic
        return float(np.clip(similarity, 0, 1))
    
    def _wu_palmer(self, synset1, synset2) -> float:
        """Расчёт Wu-Palmer similarity.
        
        Формула: 2 * depth(LCS) / (depth(a) + depth(b))
        
        Args:
            synset1: Первый синсет.
            synset2: Второй синсет.
        
        Returns:
            Значение близости [0, 1].
        """
        lcs = self._find_lcs(synset1, synset2)
        
        if lcs is None:
            return 0.0
        
        depth_lcs = self._get_depth(lcs)
        depth_a = self._get_depth(synset1)
        depth_b = self._get_depth(synset2)
        
        sum_depth = depth_a + depth_b
        if sum_depth == 0:
            return 0.0
        
        return 2 * depth_lcs / sum_depth
    
    def _path_similarity(self, synset1, synset2) -> float:
        """Расчёт Path similarity через кратчайший путь.
        
        Формула: 1 / (1 + shortest_path)
        
        Args:
            synset1: Первый синсет.
            synset2: Второй синсет.
        
        Returns:
            Значение близости [0, 1].
        """
        path_length = self._shortest_path(synset1, synset2)
        
        if path_length < 0:
            return 0.0
        
        return 1.0 / (1.0 + path_length)
    
    def get_hypernyms(self, term: str) -> list[str]:
        """Получение иерархии гиперонимов для термина.
        
        Args:
            term: Термин.
        
        Returns:
            Список названий синсетов (от частного к общему).
        """
        if not self._initialized:
            raise EnglishWordNetNotInitializedError()
        
        synset = self._get_best_synset(term)
        if synset is None:
            return []
        
        # Используем параметризованную функцию для сбора имён гиперонимов
        names = self._traverse_hypernyms(
            synset,
            accumulator=lambda lst, name: lst + [name.split('.')[0]],
            initial=[]
        )
        
        return names
    
    def get_similarity_batch(
        self,
        term_pairs: list[tuple[str, str]],
        algorithm: str = "lin",
    ) -> list[EnSimilarityResult]:
        """Расчёт близости для батча пар терминов.
        
        Args:
            term_pairs: Список пар (term1, term2).
            algorithm: Алгоритм.
        
        Returns:
            Список результатов.
        """
        return [
            self.get_similarity(t1, t2, algorithm)
            for t1, t2 in term_pairs
        ]
