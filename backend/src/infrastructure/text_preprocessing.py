# backend/src/infrastructure/text_preprocessing.py
# Предобработка текста (лемматизация + стоп-слова)
#
# Версия: 1.0
# Обновлено: 2026-04-06

"""
Модуль предобработки текста для подготовки терминов к векторизации.

Функции:
- Лемматизация с помощью pymorphy3
- Удаление стоп-слов
- Токенизация текста
"""

from typing import Optional

import pymorphy3

# Инициализируем морфологический анализатор один раз
_MORPH: Optional[pymorphy3.MorphAnalyzer] = None

# Русские стоп-слова
STOP_WORDS: set[str] = {
    "и", "в", "не", "на", "что", "он", "с", "как", "а", "то",
    "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же",
    "вы", "за", "по", "только", "её", "мне", "было", "вот", "от",
    "меня", "ещё", "нет", "о", "из", "ему", "теперь", "когда",
    "уже", "вам", "ни", "был", "это", "чтобы", "сначала", "им",
    "их", "кем", "чем", "ли", "до", "поэтому", "без", "также",
    "этих", "для", "при", "или", "это", "того", "как", "или",
    "при", "это", "бы", "весь", "ещё", "свой", "там", "тут",
    # Дополнительные стоп-слова (части слов не должны быть!)
    "об",  # предлог "об/обо"
    "со",  # предлог
}


def get_morph() -> pymorphy3.MorphAnalyzer:
    """Получение глобального экземпляра MorphAnalyzer.
    
    Returns:
        Экземпляр морфологического анализатора.
    """
    global _MORPH
    if _MORPH is None:
        _MORPH = pymorphy3.MorphAnalyzer()
    return _MORPH


def lemmatize(word: str) -> str:
    """Лемматизация слова (приведение к нормальной форме).
    
    Args:
        word: Слово для лемматизации.
    
    Returns:
        Лемма слова в нижнем регистре.
    """
    morph = get_morph()
    parsed = morph.parse(word)
    if parsed:
        # Берём первое валидное разбор, возвращаем нормальную форму
        return parsed[0].normal_form
    return word.lower()


def remove_stop_words(tokens: list[str]) -> list[str]:
    """Удаление стоп-слов из списка токенов.
    
    Args:
        tokens: Список токенов.
    
    Returns:
        Список токенов без стоп-слов.
    """
    return [t for t in tokens if t.lower() not in STOP_WORDS]


def tokenize(text: str) -> list[str]:
    """Токенизация текста.
    
    Args:
        text: Текст для токенизации.
    
    Returns:
        Список токенов (слов).
    """
    # Простая токенизация:_split по пробелам и пунктуации
    import re
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens


def preprocess_term(term: str, use_lemmatization: bool = True) -> str:
    """Полная предобработка термина.
    
    Args:
        term: Термин для предобработки.
        use_lemmatization: Использовать лемматизацию.
    
    Returns:
        Предобработанный термин.
    """
    # Токенизация
    tokens = tokenize(term)
    
    # Удаление стоп-слов
    tokens = remove_stop_words(tokens)
    
    # Лемматизация
    if use_lemmatization:
        tokens = [lemmatize(t) for t in tokens]
    
    return " ".join(tokens)


def preprocess_terms_batch(
    terms: list[str],
    use_lemmatization: bool = True,
) -> dict[str, str]:
    """Пакетная предобработка терминов.
    
    Args:
        terms: Список терминов.
        use_lemmatization: Использовать лемматизацию.
    
    Returns:
        Словарь {оригинальный_термин: предобработанный_термин}.
    """
    return {term: preprocess_term(term, use_lemmatization) for term in terms}
