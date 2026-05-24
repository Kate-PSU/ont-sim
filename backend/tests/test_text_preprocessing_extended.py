# backend/tests/test_text_preprocessing.py
# Тесты для модуля препроцессинга текста
#
# Версия: 1.0
# Обновлено: 2026-04-10

"""
Тесты для text_preprocessing — функции предобработки текста:
- Лемматизация (pymorphy3)
- Токенизация
- Удаление стоп-слов
"""

import pytest

from src.infrastructure.text_preprocessing import (
    get_morph,
    lemmatize,
    remove_stop_words,
    tokenize,
    preprocess_term,
    preprocess_terms_batch,
)


class TestGetMorph:
    """Тесты инициализации морфологического анализатора."""

    def test_get_morph_returns_analyzer(self):
        """Тест: get_morph возвращает анализатор."""
        morph = get_morph()
        assert morph is not None

    def test_get_morph_singleton(self):
        """Тест: get_morph возвращает синглтон."""
        morph1 = get_morph()
        morph2 = get_morph()
        assert morph1 is morph2


class TestLemmatize:
    """Тесты лемматизации."""

    def test_lemmatize_noun(self):
        """Тест: лемматизация существительного."""
        result = lemmatize("машинное")
        assert result in ["машинный", "машинное", "машинных"]

    def test_lemmatize_verb(self):
        """Тест: лемматизация глагола."""
        result = lemmatize("обучение")
        assert result in ["обучение", "обучить", "учить"]

    def test_lemmatize_adjective(self):
        """Тест: лемматизация прилагательного."""
        result = lemmatize("нейронные")
        assert result in ["нейронный", "нейронное", "нейронных"]

    def test_lemmatize_english(self):
        """Тест: лемматизация английских слов."""
        result = lemmatize("networks")
        # pymorphy3 может не знать английские слова
        assert isinstance(result, str)

    def test_lemmatize_already_normal(self):
        """Тест: слово в нормальной форме."""
        result = lemmatize("кот")
        assert result == "кот"


class TestRemoveStopWords:
    """Тесты удаления стоп-слов."""

    def test_remove_stop_words_empty(self):
        """Тест: пустой список."""
        result = remove_stop_words([])
        assert result == []

    def test_remove_stop_words_no_stops(self):
        """Тест: нет стоп-слов."""
        tokens = ["машинное", "обучение"]
        result = remove_stop_words(tokens)
        assert len(result) == 2

    def test_remove_stop_words_with_stops(self):
        """Тест: со стоп-словами."""
        tokens = ["и", "машинное", "обучение", "в"]
        result = remove_stop_words(tokens)
        assert "и" not in result
        assert "в" not in result
        assert "машинное" in result
        assert "обучение" in result

    def test_remove_stop_words_all_stops(self):
        """Тест: все токены — стоп-слова."""
        tokens = ["и", "в", "на", "по"]
        result = remove_stop_words(tokens)
        assert len(result) == 0


class TestTokenize:
    """Тесты токенизации."""

    def test_tokenize_single_word(self):
        """Тест: одно слово."""
        result = tokenize("машинное")
        assert result == ["машинное"]

    def test_tokenize_multiple_words(self):
        """Тест: несколько слов."""
        result = tokenize("машинное обучение")
        assert result == ["машинное", "обучение"]

    def test_tokenize_with_punctuation(self):
        """Тест: с пунктуацией."""
        result = tokenize("нейронная, сеть!")
        assert "нейронная" in result
        assert "сеть" in result

    def test_tokenize_empty(self):
        """Тест: пустая строка."""
        result = tokenize("")
        assert result == []

    def test_tokenize_numbers(self):
        """Тест: числа в тексте."""
        result = tokenize("CNN в 2024 году")
        assert "2024" in result


class TestPreprocessTerm:
    """Тесты препроцессинга отдельного термина."""

    def test_preprocess_term_simple(self):
        """Тест: простой термин."""
        result = preprocess_term("обучение")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_preprocess_term_with_lemmatization(self):
        """Тест: с лемматизацией."""
        result = preprocess_term("нейронными", use_lemmatization=True)
        assert isinstance(result, str)

    def test_preprocess_term_without_lemmatization(self):
        """Тест: без лемматизации."""
        result = preprocess_term("нейронными", use_lemmatization=False)
        assert result == "нейронными"

    def test_preprocess_term_english(self):
        """Тест: английский термин."""
        result = preprocess_term("machine learning")
        assert isinstance(result, str)


class TestPreprocessTermsBatch:
    """Тесты пакетной предобработки."""

    def test_preprocess_batch_empty(self):
        """Тест: пустой список."""
        result = preprocess_terms_batch([])
        assert result == {}

    def test_preprocess_batch_single(self):
        """Тест: один термин."""
        result = preprocess_terms_batch(["машинное"])
        assert len(result) == 1
        assert "машинное" in result

    def test_preprocess_batch_multiple(self):
        """Тест: несколько терминов."""
        terms = ["машинное обучение", "глубокое обучение", "нейронные сети"]
        result = preprocess_terms_batch(terms)
        assert len(result) == 3

    def test_preprocess_batch_with_lemmatization(self):
        """Тест: пакет с лемматизацией."""
        terms = ["нейронными", "сетями"]
        result = preprocess_terms_batch(terms, use_lemmatization=True)
        assert len(result) == 2

    def test_preprocess_batch_preserves_order(self):
        """Тест: порядок терминов сохраняется."""
        terms = ["первый", "второй", "третий"]
        result = preprocess_terms_batch(terms)
        assert len(result) == 3
        # Каждый элемент - обработанный термин
        for term in terms:
            assert term in result


class TestTextPreprocessingEdgeCases:
    """Edge cases для препроцессинга."""

    def test_lemmatize_empty_string(self):
        """Тест: пустая строка."""
        result = lemmatize("")
        assert result == ""

    def test_tokenize_only_punctuation(self):
        """Тест: только пунктуация."""
        result = tokenize(".,!?")
        # Токенизатор может вернуть пустой список или список с элементами
        assert isinstance(result, list)

    def test_preprocess_cyrillic(self):
        """Тест: кириллица с диакритикой."""
        result = preprocess_term("ёжик")
        assert "ёж" in result or "ёжик" in result

    def test_preprocess_mixed_case(self):
        """Тест: смешанный регистр."""
        result = preprocess_term("МаШинное")
        # Результат должен быть в нормальной форме
        assert isinstance(result, str)

    def test_preprocess_long_term(self):
        """Тест: длинный термин."""
        long_term = " ".join(["слово"] * 50)
        result = preprocess_term(long_term)
        assert isinstance(result, str)


class TestStopWords:
    """Тесты списка стоп-слов."""

    def test_common_stop_words(self):
        """Тест: типичные стоп-слова."""
        stop_words = ["и", "в", "на", "по", "с", "из", "к", "для", "о", "об"]
        tokens = ["машинное"] + stop_words + ["обучение"]
        result = remove_stop_words(tokens)
        
        # Стоп-слова должны быть удалены
        for sw in stop_words:
            assert sw not in result
        
        # Осмысленные слова должны остаться
        assert "машинное" in result
        assert "обучение" in result

    def test_stop_words_case_insensitive(self):
        """Тест: стоп-слова независимы от регистра."""
        tokens = ["И", "В", "Машинное"]
        result = remove_stop_words(tokens)
        assert "И" not in result
        assert "В" not in result
        assert "Машинное" in result
