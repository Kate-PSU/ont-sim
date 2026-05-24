# backend/src/infrastructure/wordnet_service.py
# Сервис семантической близости на основе RuWordNet
#
# Версия: 1.0
# Обновлено: 2026-04-08

"""
Модуль для расчёта семантической близости терминов через RuWordNet.

Алгоритмы:
- Lin similarity: 2 * IC(lcs) / (IC(a) + IC(b))
- Wu-Palmer: 2 * depth(lcs) / (depth(a) + depth(b))
- Shortest path: расстояние между синсетами в графе

Зависимости:
- ruwordnet>=0.0.6
- База данных: ruwordnet-2021.db
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from ruwordnet import RuWordNet
    RUWORDNET_AVAILABLE = True
except ImportError:
    RUWORDNET_AVAILABLE = False


@dataclass
class SynsetInfo:
    """Информация о синсете для отладки."""
    synset_id: int
    title: str
    gloss: str
    depth: int
    ic: float  # Information Content


@dataclass
class SimilarityResult:
    """Результат расчёта близости между терминами."""
    term1: str
    term2: str
    similarity: float
    algorithm: str
    synset1: Optional[SynsetInfo]
    synset2: Optional[SynsetInfo]
    lcs: Optional[SynsetInfo]  # Lowest Common Subsumer


class WordNetServiceError(Exception):
    """Ошибка сервиса WordNet."""
    pass


class WordNetNotInitializedError(WordNetServiceError):
    """WordNet не инициализирован."""
    pass


class WordNetService:
    """Сервис расчёта семантической близости через RuWordNet.
    
    Использует иерархию синсетов для расчёта близости между терминами
    с помощью алгоритмов Lin, Wu-Palmer и Shortest Path.
    
    Атрибуты:
        wn: Экземпляр RuWordNet.
        _ic_cache: Кеш Information Content для синсетов.
        _depth_cache: Кеш глубины синсетов.
        _synset_index: Индекс синсетов по названию.
    """
    
    def __init__(self, db_path: Optional[str] = None) -> None:
        """Инициализация сервиса.
        
        Args:
            db_path: Путь к базе данных. По умолчанию читается из RUWORDNET_DB_PATH.
        """
        import os
        
        self.wn: Optional[RuWordNet] = None
        self._ic_cache: dict[int, float] = {}
        self._depth_cache: dict[int, int] = {}
        self._synset_index: dict[str, list] = {}
        
        # Приоритет: переданный db_path > env RUWORDNET_DB_PATH > None
        if db_path is None:
            db_path = os.environ.get("RUWORDNET_DB_PATH")
        self._db_path = db_path
    
    def initialize(self) -> None:
        """Инициализация подключения к RuWordNet."""
        if not RUWORDNET_AVAILABLE:
            raise ImportError(
                "ruwordnet не установлен. Установите: pip install ruwordnet"
            )
        
        try:
            if self._db_path:
                self.wn = RuWordNet(self._db_path)
            else:
                self.wn = RuWordNet()
            
            # Построить индекс синсетов
            self._build_synset_index()
        except Exception as e:
            raise WordNetServiceError(f"Не удалось инициализировать RuWordNet: {e}")
    
    def _build_synset_index(self) -> None:
        """Построение индекса синсетов для быстрого поиска."""
        if self.wn is None:
            raise WordNetNotInitializedError()
        
        # Собираем все синсеты
        self._synset_index = {}
        
        # Получаем все синсеты из базы
        for synset in self.wn.synsets:
            title_lower = synset.title.lower()
            if title_lower not in self._synset_index:
                self._synset_index[title_lower] = []
            self._synset_index[title_lower].append(synset)
    
    def _get_depth(self, synset) -> int:
        """Получение глубины синсета в иерархии.
        
        Args:
            synset: Синсет RuWordNet.
        
        Returns:
            Глубина (количество рёбер до корня).
        """
        if synset.id in self._depth_cache:
            return self._depth_cache[synset.id]
        
        depth = 0
        current = synset
        visited = set()
        
        while current.hypernyms and depth < 100:  # Защита от циклов
            next_hyper = None
            for h in current.hypernyms:
                if h.id not in visited:
                    next_hyper = h
                    visited.add(h.id)
                    break
            
            if next_hyper is None:
                break
                
            current = next_hyper
            depth += 1
        
        self._depth_cache[synset.id] = depth
        return depth
    
    def _compute_ic(self, synset) -> float:
        """Вычисление Information Content синсета.
        
        IC = -log(p(synset)), где p — вероятность синсета.
        Вычисляется на основе частот в корпусе.
        
        Args:
            synset: Синсет RuWordNet.
        
        Returns:
            Значение IC в диапазоне [0, 1].
        """
        if synset.id in self._ic_cache:
            return self._ic_cache[synset.id]
        
        # Количество sense у синсета как мера специфичности
        synset_senses = len(synset.senses) if synset.senses else 1
        
        # IC пропорционален количеству sense (более специфичные = больше IC)
        # Нормализуем: больше sense = выше IC
        ic = np.log2(1 + synset_senses) / np.log2(1 + 100)  # Нормализация
        
        self._ic_cache[synset.id] = ic
        return ic
    
    def _get_senses(self, term: str) -> list:
        """Получение всех смыслов (senses) для термина.
        
        Args:
            term: Термин для поиска.
        
        Returns:
            Список senses.
        """
        if self.wn is None:
            raise WordNetNotInitializedError()
        
        return self.wn.get_senses(term)
    
    def _get_synset_for_term(self, term: str) -> Optional:
        """Получение первого синсета для термина.
        
        Args:
            term: Термин.
        
        Returns:
            Первый синсет или None.
        """
        senses = self._get_senses(term)
        if senses:
            return senses[0].synset
        return None
    
    def _find_lcs(self, synset1, synset2) -> Optional:
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
        
        # Получаем предков synset1
        ancestors1 = set()
        current = synset1
        visited = set()
        
        while current.hypernyms:
            next_hyper = None
            for h in current.hypernyms:
                if h.id not in visited:
                    next_hyper = h
                    visited.add(h.id)
                    break
            
            if next_hyper is None:
                break
            current = next_hyper
            ancestors1.add(current.id)
        
        # Ищем ближайшего предка synset2
        current = synset2
        visited = set()
        
        while current.hypernyms:
            if current.id in ancestors1:
                return current
            
            next_hyper = None
            for h in current.hypernyms:
                if h.id not in visited:
                    next_hyper = h
                    visited.add(h.id)
                    break
            
            if next_hyper is None:
                break
            current = next_hyper
        
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
        
        if synset1.id == synset2.id:
            return 0
        
        # BFS от synset1
        from collections import deque
        
        queue = deque([(synset1, 0)])
        visited = {synset1.id}
        
        while queue:
            current, distance = queue.popleft()
            
            if current.id == synset2.id:
                return distance
            
            # Проверяем гиперонимы и гипонимы
            neighbors = []
            if current.hypernyms:
                neighbors.extend(current.hypernyms)
            if current.hyponyms:
                neighbors.extend(current.hyponyms)
            
            for neighbor in neighbors:
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    queue.append((neighbor, distance + 1))
        
        return -1  # Не найдено
    
    def get_similarity(
        self,
        term1: str,
        term2: str,
        algorithm: str = "lin",
    ) -> SimilarityResult:
        """Расчёт семантической близости между терминами.
        
        Args:
            term1: Первый термин.
            term2: Второй термин.
            algorithm: Алгоритм ('lin', 'wup', 'path').
        
        Returns:
            Результат с мерой близости.
        """
        if self.wn is None:
            raise WordNetNotInitializedError()
        
        synset1 = self._get_synset_for_term(term1)
        synset2 = self._get_synset_for_term(term2)
        
        similarity = 0.0
        lcs = None
        
        if synset1 is None or synset2 is None:
            # Один из терминов не найден в WordNet
            return SimilarityResult(
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
        
        return SimilarityResult(
            term1=term1,
            term2=term2,
            similarity=similarity,
            algorithm=algorithm,
            synset1=self._synset_info(synset1),
            synset2=self._synset_info(synset2),
            lcs=self._synset_info(lcs) if lcs else None,
        )
    
    def _synset_info(self, synset) -> Optional[SynsetInfo]:
        """Получение информации о синсете."""
        if synset is None:
            return None
        return SynsetInfo(
            synset_id=synset.id,
            title=synset.title,
            gloss=synset.gloss if hasattr(synset, 'gloss') else "",
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
        if self.wn is None:
            raise WordNetNotInitializedError()
        
        synset = self._get_synset_for_term(term)
        if synset is None:
            return []
        
        hypernyms = []
        current = synset
        visited = set()
        
        while current.hypernyms:
            next_hyper = None
            for h in current.hypernyms:
                if h.id not in visited:
                    next_hyper = h
                    visited.add(h.id)
                    break
            
            if next_hyper is None:
                break
            
            current = next_hyper
            hypernyms.append(current.title)
        
        return hypernyms
    
    def get_similarity_batch(
        self,
        term_pairs: list[tuple[str, str]],
        algorithm: str = "lin",
    ) -> list[SimilarityResult]:
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
    
    def domain_similarity(
        self,
        terms1: list[str],
        terms2: list[str],
        algorithm: str = "lin",
        aggregation: str = "max",
    ) -> float:
        """Расчёт близости между доменами (наборами терминов).
        
        Args:
            terms1: Термины первого домена.
            terms2: Термины второго домена.
            algorithm: Алгоритм близости.
            aggregation: Метод агрегации ('max', 'mean', 'min').
        
        Returns:
            Значение близости между доменами.
        """
        if not terms1 or not terms2:
            return 0.0
        
        similarities = []
        for t1 in terms1:
            for t2 in terms2:
                result = self.get_similarity(t1, t2, algorithm)
                similarities.append(result.similarity)
        
        if not similarities:
            return 0.0
        
        if aggregation == "max":
            return max(similarities)
        elif aggregation == "mean":
            return sum(similarities) / len(similarities)
        elif aggregation == "min":
            return min(similarities)
        else:
            raise ValueError(f"Неизвестный метод агрегации: {aggregation}")
    def get_term_centroid(self, terms: list[str]) -> np.ndarray:
        """Вычисление центроида списка терминов через WordNet гиперонимы.
        
        Использует иерархию гиперонимов для создания онтологического 
        представления каждого термина, затем усредняет для получения
        центроида домена.
        
        Args:
            terms: Список терминов домена.
        
        Returns:
            Центроид (нормализованный псевдо-вектор).
        """
        if not terms:
            return np.zeros(1024)
        
        if self.wn is None:
            raise WordNetNotInitializedError()
        
        # Собираем синсеты с весами
        synset_weights: dict[str, float] = {}
        
        for term in terms:
            synset = self._get_synset_for_term(term)
            if synset is None:
                continue
            
            # Базовый вес
            base_weight = 1.0
            if synset.id not in synset_weights:
                synset_weights[synset.id] = 0.0
            synset_weights[synset.id] += base_weight
            
            # Гиперонимы с убывающими весами
            current = synset
            depth = 0
            hypernym_weight = base_weight
            visited = set()
            
            while current.hypernyms and depth < 10:
                next_hyper = None
                for h in current.hypernyms:
                    if h.id not in visited:
                        next_hyper = h
                        visited.add(h.id)
                        break
                
                if next_hyper is None:
                    break
                
                current = next_hyper
                depth += 1
                hypernym_weight *= 0.7
                
                if current.id not in synset_weights:
                    synset_weights[current.id] = 0.0
                synset_weights[current.id] += hypernym_weight
        
        if not synset_weights:
            return np.zeros(1024)
        
        # Нормализуем веса
        total_weight = sum(synset_weights.values())
        if total_weight > 0:
            for synset_id in synset_weights:
                synset_weights[synset_id] /= total_weight
        
        # Создаём псевдо-центроид
        embedding_size = 1024
        centroid = np.zeros(embedding_size)
        
        # Детерминированный seed на основе хеша ключей
        seed_value = hash(tuple(sorted(synset_weights.keys()))) % 2**32
        np.random.seed(seed_value)
        
        for synset_id, weight in synset_weights.items():
            pseudo_vector = np.random.randn(embedding_size)
            pseudo_vector = pseudo_vector / np.linalg.norm(pseudo_vector)
            centroid += weight * pseudo_vector
        
        # Нормализуем
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        
        return centroid
