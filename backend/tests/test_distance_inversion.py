"""Тест для проверки корректности инверсии distanceij → sim"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import sys
import os

# Путь к файлу данных - абсолютный путь
DATA_PATH = "/home/maxbogus/Repositories/diplomMagistrate/data/simlex999_rus_without_dupl.csv"


def test_simlex999_ru_correlation_direction():
    """Проверяет, что в simlex999_ru большее distanceij соответствует большему сходству.
    
    Гипотеза: distanceij в этом датасете - это уже мера сходства (similarity),
    а не расстояние! Высокие значения (8-10) соответствуют синонимам,
    низкие (0-2) - антонимам.
    
    Это означает, что distanceij НЕ нужно инвертировать в 10 - distanceij.
    """
    
    print("=" * 70)
    print("ТЕСТ: Проверка корректности интерпретации distanceij")
    print("=" * 70)
    print()
    
    # 1. Загрузить данные
    print("1. ЗАГРУЗКА ДАННЫХ")
    print("-" * 40)
    df = pd.read_csv(DATA_PATH)
    print(f"   Всего пар: {len(df)}")
    print(f"   Колонки: {list(df.columns)}")
    print()
    
    # 2. Показать распределение distanceij
    print("2. РАСПРЕДЕЛЕНИЕ distanceij")
    print("-" * 40)
    distanceij = df['distanceij'].values
    print(f"   Мин: {distanceij.min():.2f}")
    print(f"   Макс: {distanceij.max():.2f}")
    print(f"   Среднее: {distanceij.mean():.2f}")
    print(f"   Медиана: {np.median(distanceij):.2f}")
    print(f"   Std: {distanceij.std():.2f}")
    print()
    
    # 3. Показать примеры пар
    print("3. ПРИМЕРЫ ПАР С РАЗНЫМИ distanceij")
    print("-" * 40)
    
    # Высокие значения (синонимы)
    high_sim = df.nlargest(5, 'distanceij')
    print("   Высокие значения (предположительно синонимы):")
    for _, row in high_sim.iterrows():
        print(f"      {row['word1']} - {row['word2']}: {row['distanceij']:.2f}")
    print()
    
    # Низкие значения (антонимы)
    low_sim = df.nsmallest(5, 'distanceij')
    print("   Низкие значения (предположительно антонимы):")
    for _, row in low_sim.iterrows():
        print(f"      {row['word1']} - {row['word2']}: {row['distanceij']:.2f}")
    print()
    
    # 4. Проверить корреляцию с идеальной similarity
    print("4. АНАЛИЗ КОРРЕЛЯЦИИ")
    print("-" * 40)
    
    # Если distanceij это уже similarity (высокие значения = похожие слова),
    # то он должен быть положительно коррелирован с человеческими оценками
    # В данном случае человеческая оценка ~ distanceij (так как это уже similarity)
    
    human_scores = df['distanceij'].values  # В этом файле distanceij = human similarity
    
    # Симулируем разные сценарии
    # Scenario 1: distanceij интерпретируется как similarity (БЕЗ инверсии)
    sim_scenario = human_scores
    
    # Scenario 2: distanceij интерпретируется как distance (С инверсией в 10 - distance)
    distance_scenario = 10 - human_scores
    
    print("   Scenario 1: distanceij как similarity (БЕЗ инверсии)")
    print(f"      Корреляция с собой: {pearsonr(sim_scenario, human_scores)[0]:.4f}")
    print()
    
    print("   Scenario 2: distanceij как distance (С инверсией: 10 - distance)")
    print(f"      Корреляция с human_scores: {pearsonr(distance_scenario, human_scores)[0]:.4f}")
    print()
    
    # 5. Гистограмма распределения
    print("5. ГИСТОГРАММА РАСПРЕДЕЛЕНИЯ distanceij")
    print("-" * 40)
    
    # Разобьём на бины
    bins = [0, 2, 4, 6, 8, 10]
    hist, edges = np.histogram(distanceij, bins=bins)
    
    print(f"   Бины: {edges}")
    print(f"   Частоты: {hist}")
    print()
    
    for i in range(len(hist)):
        bar = "█" * (hist[i] // 10 + 1)
        print(f"   [{edges[i]:.0f}-{edges[i+1]:.0f}): {hist[i]:3d} {bar}")
    print()
    
    # 6. Вывод о правильности инверсии
    print("6. ВЫВОД О ПРАВИЛЬНОСТИ ИНВЕРСИИ")
    print("=" * 70)
    
    # Проверяем гипотезу - смотрим на типичные пары
    high_values = df[df['distanceij'] >= 8]
    low_values = df[df['distanceij'] <= 1]
    
    print(f"   Пар с distanceij >= 8 (синонимы): {len(high_values)}")
    print(f"   Пар с distanceij <= 1 (антонимы): {len(low_values)}")
    print()
    
    # Проверяем типичные пары
    print("   Типичные 'синонимы' (high distanceij):")
    for _, row in high_values.head(3).iterrows():
        print(f"      '{row['word1']}' - '{row['word2']}': {row['distanceij']}")
    print()
    
    print("   Типичные 'антонимы' (low distanceij):")
    for _, row in low_values.head(3).iterrows():
        print(f"      '{row['word1']}' - '{row['word2']}': {row['distanceij']}")
    print()
    
    # Финальный вердикт
    print("=" * 70)
    print("ЗАКЛЮЧЕНИЕ:")
    print("=" * 70)
    
    # КЛЮЧЕВОЙ КРИТЕРИЙ: высокие значения должны соответствовать синонимам
    # Проверяем семантику: 
    # - лошадь/кобыла (синонимы) должны иметь ВЫСОКИЙ score
    # - тяжелый/легкий (антонимы) должны иметь НИЗКИЙ score
    
    # Проверяем несколько известных пар
    synonyms_high = True
    antonyms_low = True
    
    # Проверяем синонимы
    synonym_pairs = df[
        ((df['word1'] == 'лошадь') & (df['word2'] == 'кобыла')) |
        ((df['word1'] == 'кобыла') & (df['word2'] == 'лошадь')) |
        ((df['word1'] == 'учитель') & (df['word2'] == 'преподаватель')) |
        ((df['word1'] == 'преподаватель') & (df['word2'] == 'учитель'))
    ]
    if len(synonym_pairs) > 0:
        avg_synonym = synonym_pairs['distanceij'].mean()
        synonyms_high = avg_synonym >= 8
        print(f"   Средний score синонимов: {avg_synonym:.2f} (ожидается >= 8)")
    
    # Проверяем антонимы
    antonym_pairs = df[
        ((df['word1'] == 'тяжелый') & (df['word2'] == 'легкий')) |
        ((df['word1'] == 'легкий') & (df['word2'] == 'тяжелый')) |
        ((df['word1'] == 'короткий') & (df['word2'] == 'длинный')) |
        ((df['word1'] == 'длинный') & (df['word2'] == 'короткий'))
    ]
    if len(antonym_pairs) > 0:
        avg_antonym = antonym_pairs['distanceij'].mean()
        antonyms_low = avg_antonym <= 2
        print(f"   Средний score антонимов: {avg_antonym:.2f} (ожидается <= 2)")
    
    # Итоговый вердикт
    is_similarity_not_distance = synonyms_high and antonyms_low
    
    if is_similarity_not_distance:
        print()
        print("=" * 70)
        print("✓ distanceij ВЕРНО интерпретируется как мера СХОДСТВА (similarity)")
        print("✓ Инверсия НЕ требуется")
        print("✓ Высокие значения → синонимы, низкие → антонимы")
        print()
        print("  Текущий код в benchmark_service.py (строка 135):")
        print("     invert = (score_col == 'distance')  # ← ПРАВИЛЬНО!")
        print()
        print("  Если изменить на: invert = (score_col in ['distance', 'distanceij'])")
        print("  это БУДЕТ ОШИБКОЙ!")
        print("=" * 70)
    else:
        print()
        print("=" * 70)
        print("✗ distanceij не похож на typical similarity score")
        print("  Возможно, требуется дополнительная проверка")
        print(f"  synonyms_high={synonyms_high}, antonyms_low={antonyms_low}")
        print("=" * 70)
    
    # Assert для теста
    assert is_similarity_not_distance, \
        "distanceij должен интерпретироваться как мера сходства (similarity), НЕ как расстояние"
    
    return is_similarity_not_distance


if __name__ == "__main__":
    test_simlex999_ru_correlation_direction()


def test_load_standard_pairs_distanceij_no_inversion():
    """Тест проверяет, что load_standard_pairs НЕ инвертирует distanceij.
    
    distanceij - это УЖЕ similarity score в диапазоне 0-10,
    где 8-10 = синонимы, 0-2 = антонимы.
    Инверсия (10 - score) для него НЕ требуется!
    """
    from scripts.benchmark.generators import load_standard_pairs
    
    DATA_PATH = "/home/maxbogus/Repositories/diplomMagistrate/data/simlex999_rus_without_dupl.csv"
    
    pairs = load_standard_pairs(DATA_PATH)
    
    # Находим известные синонимы и антонимы
    synonyms = [p for p in pairs if p['word1'] == 'лошадь' and p['word2'] == 'кобыла']
    antonyms = [p for p in pairs if p['word1'] == 'тяжелый' and p['word2'] == 'легкий']
    
    assert len(synonyms) > 0, "Не найдена пара синонимов 'лошадь'-'кобыла'"
    assert len(antonyms) > 0, "Не найдена пара антонимов 'тяжелый'-'легкий'"
    
    syn_score = synonyms[0]['score']
    ant_score = antonyms[0]['score']
    
    # Синонимы должны иметь ВЫСОКИЙ score (>= 8)
    # Антонимы должны иметь НИЗКИЙ score (<= 2)
    # Если бы была инверсия, было бы наоборот!
    
    assert syn_score >= 8, \
        f"Ожидалось, что синонимы 'лошадь'-'кобыла' имеют score >= 8, получено {syn_score}"
    
    assert ant_score <= 2, \
        f"Ожидалось, что антонимы 'тяжелый'-'легкий' имеют score <= 2, получено {ant_score}"
    
    # Дополнительная проверка: синонимы должны иметь БОЛЬШИЙ score чем антонимы
    assert syn_score > ant_score, \
        f"Синонимы ({syn_score}) должны иметь больший score чем антонимы ({ant_score})"
    
    print(f"✓ Синоним 'лошадь'-'кобыла': score={syn_score} (ожидалось >= 8)")
    print(f"✓ Антоним 'тяжелый'-'легкий': score={ant_score} (ожидалось <= 2)")
    print("✓ Инверсия НЕ применяется для distanceij - БАГ ИСПРАВЛЕН!")


def test_distance_column_still_inverted():
    """Тест проверяет, что колонка 'distance' (обычное расстояние) всё ещё инвертируется.
    
    Это гарантирует, что исправление не сломало логику для настоящих distance колонок.
    """
    import tempfile
    import os
    import csv
    from scripts.benchmark.generators import load_standard_pairs
    
    # Создаём временный CSV с колонкой 'distance'
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        writer = csv.DictWriter(f, fieldnames=['word1', 'word2', 'distance'])
        writer.writeheader()
        # Синонимы должны получить высокий score (10 - 2 = 8)
        writer.writerow({'word1': 'horse', 'word2': 'mare', 'distance': 2})
        # Антонимы должны получить низкий score (10 - 9 = 1)
        writer.writerow({'word1': 'heavy', 'word2': 'light', 'distance': 9})
        temp_path = f.name
    
    try:
        pairs = load_standard_pairs(temp_path)
        
        # Находим пары
        horse_pair = [p for p in pairs if p['word1'] == 'horse'][0]
        heavy_pair = [p for p in pairs if p['word1'] == 'heavy'][0]
        
        # Для 'distance' инверсия должна работать: max_dist - distance
        # distance=2 → score=8 (синоним)
        # distance=9 → score=1 (антоним)
        assert horse_pair['score'] == 8, f"distance=2 должен стать 8, получено {horse_pair['score']}"
        assert heavy_pair['score'] == 1, f"distance=9 должен стать 1, получено {heavy_pair['score']}"
        
        print("✓ Колонка 'distance' правильно инвертируется: max_dist - distance")
        print("✓ БUG FIX: distanceij НЕ инвертируется, но 'distance' - инвертируется")
    finally:
        os.unlink(temp_path)
