# backend/tests/test_text_preprocessing.py
# Тесты для модуля предобработки текста
#
# Версия: 1.0
# Обновлено: 2026-04-06

"""
Тесты для text_preprocessing.py
TDD: сначала тесты, потом код
"""

import pytest

from src.infrastructure.text_preprocessing import (
    STOP_WORDS,
    lemmatize,
    preprocess_term,
    preprocess_terms_batch,
    remove_stop_words,
    tokenize,
)


class TestStopWords:
    """Тесты стоп-слов."""
    
    def test_stop_words_not_empty(self):
        """Тест: набор стоп-слов не пуст."""
        assert len(STOP_WORDS) > 0
    
    def test_common_stop_words_present(self):
        """Тест: популярные стоп-слова присутствуют."""
        common = {"и", "в", "не", "на", "что", "он", "а", "но"}
        for word in common:
            assert word in STOP_WORDS


class TestTokenize:
    """Тесты токенизации."""
    
    def test_tokenize_simple(self):
        """Тест: простая токенизация."""
        tokens = tokenize("машинное обучение")
        assert tokens == ["машинное", "обучение"]
    
    def test_tokenize_with_punctuation(self):
        """Тест: токенизация с пунктуацией."""
        tokens = tokenize("нейронная сеть, глубокое обучение!")
        assert "нейронная" in tokens
        assert "сеть" in tokens
        assert "глубокое" in tokens
        assert "обучение" in tokens
    
    def test_tokenize_lowercase(self):
        """Тест: результат в нижнем регистре."""
        tokens = tokenize("МАШИННОЕ ОБУЧЕНИЕ")
        assert tokens == ["машинное", "обучение"]
    
    def test_tokenize_empty(self):
        """Тест: пустая строка."""
        tokens = tokenize("")
        assert tokens == []


class TestRemoveStopWords:
    """Тесты удаления стоп-слов."""
    
    def test_remove_common_stop_words(self):
        """Тест: удаление популярных стоп-слов."""
        tokens = ["машинное", "обучение", "и", "нейронная", "сеть"]
        result = remove_stop_words(tokens)
        assert "и" not in result
        assert "машинное" in result
        assert "обучение" in result
        assert "нейронная" in result
        assert "сеть" in result
    
    def test_remove_all_stop_words(self):
        """Тест: все стоп-слова удаляются."""
        tokens = list(STOP_WORDS)
        result = remove_stop_words(tokens)
        assert len(result) == 0
    
    def test_remove_none(self):
        """Тест: если нет стоп-слов — ничего не удаляется."""
        tokens = ["машинное", "обучение"]
        result = remove_stop_words(tokens)
        assert result == tokens


class TestLemmatize:
    """Тесты лемматизации."""
    
    def test_lemmatize_noun(self):
        """Тест: лемматизация существительного в мужском роде."""
        lemma = lemmatize("машина")
        # pymorphy3 может возвращать разные формы
        assert lemma in ["машина", "машинный", "машин"]
    
    def test_lemmatize_verb(self):
        """Тест: лемматизация глагола."""
        lemma = lemmatize("обучается")
        assert lemma == "обучаться"
    
    def test_lemmatize_adjective(self):
        """Тест: лемматизация прилагательного."""
        lemma = lemmatize("нейронной")
        assert lemma == "нейронный"
    
    def test_lemmatize_lowercase_input(self):
        """Тест: входной параметр в нижнем регистре."""
        lemma = lemmatize("сеть")
        assert lemma == "сеть"


class TestPreprocessTerm:
    """Тесты полной предобработки термина."""
    
    def test_preprocess_single_word(self):
        """Тест: предобработка однословного термина."""
        result = preprocess_term("машинное")
        assert len(result) > 0
    
    def test_preprocess_multi_word(self):
        """Тест: предобработка многословного термина."""
        result = preprocess_term("машинное обучение")
        # Результат не пустой
        assert len(result) > 0
        # Содержит осмысленные слова
        assert "обучение" in result
        assert "машинный" in result or "машинное" in result
    
    def test_preprocess_without_lemmatization(self):
        """Тест: предобработка без лемматизации."""
        result = preprocess_term("машинное обучение", use_lemmatization=False)
        # Слова должны быть в нижнем регистре
        assert result == "машинное обучение"
    
    def test_preprocess_with_stop_words(self):
        """Тест: стоп-слова удаляются."""
        result = preprocess_term("методы машинного обучения")
        # Проверяем, что результат содержит полезные слова
        assert "обучение" in result
    
    def test_preprocess_empty(self):
        """Тест: пустой термин."""
        result = preprocess_term("")
        assert result == ""


class TestPreprocessTermsBatch:
    """Тесты пакетной предобработки."""
    
    def test_batch_empty(self):
        """Тест: пустой список."""
        result = preprocess_terms_batch([])
        assert result == {}
    
    def test_batch_single_term(self):
        """Тест: список с одним термином."""
        result = preprocess_terms_batch(["машинное обучение"])
        assert "машинное обучение" in result
        assert len(result) == 1
    
    def test_batch_multiple_terms(self):
        """Тест: список с несколькими терминами."""
        terms = [
            "машинное обучение",
            "глубокий нейронный сеть",
            "компьютерный зрение",
        ]
        result = preprocess_terms_batch(terms)
        assert len(result) == 3
        for term in terms:
            assert term in result
    
    def test_batch_without_lemmatization(self):
        """Тест: пакетная обработка без лемматизации."""
        terms = ["машинное обучение", "нейронная сеть"]
        result = preprocess_terms_batch(terms, use_lemmatization=False)
        assert result["машинное обучение"] == "машинное обучение"
        assert result["нейронная сеть"] == "нейронная сеть"
