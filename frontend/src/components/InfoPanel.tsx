// frontend/src/components/InfoPanel.tsx
// Панель информации с подсказками для разных режимов визуализации
//
// Версия: 1.0

import React from 'react'

export type InfoPanelMode = 'graph' | 'matrix' | 'domains' | 'benchmark'

interface InfoPanelProps {
  mode: InfoPanelMode
}

const styles: Record<string, React.CSSProperties> = {
  panel: {
    padding: '1.25rem',
    background: 'rgba(255, 255, 255, 0.05)',
    borderRadius: '12px',
    border: '1px solid rgba(255, 255, 255, 0.1)',
    textAlign: 'left' as const,
    fontSize: '0.9rem',
    lineHeight: '1.6',
    color: '#ccc',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    marginBottom: '1rem',
    paddingBottom: '0.75rem',
    borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
  },
  title: {
    fontSize: '1.1rem',
    fontWeight: 600,
    color: '#fff',
    margin: 0,
  },
  section: {
    marginBottom: '1rem',
  },
  sectionTitle: {
    fontSize: '0.95rem',
    fontWeight: 600,
    color: '#aaa',
    marginBottom: '0.5rem',
  },
  list: {
    margin: 0,
    paddingLeft: '1.25rem',
  },
  listItem: {
    marginBottom: '0.5rem',
  },
  subList: {
    marginTop: '0.5rem',
    paddingLeft: '1.5rem',
  },
  highlight: {
    color: '#69ff94',
    fontWeight: 500,
  },
  emoji: {
    marginRight: '0.35rem',
  },
  note: {
    marginTop: '0.75rem',
    padding: '0.75rem',
    background: 'rgba(255, 255, 255, 0.05)',
    borderRadius: '6px',
    border: '1px solid rgba(255, 255, 255, 0.08)',
  },
  noteTitle: {
    fontWeight: 600,
    color: '#fff',
    marginBottom: '0.25rem',
  },
  qa: {
    marginTop: '0.75rem',
  },
  q: {
    color: '#646cff',
    fontWeight: 600,
    marginBottom: '0.25rem',
  },
  a: {
    color: '#aaa',
    paddingLeft: '1rem',
  },
  tree: {
    fontFamily: 'monospace',
    fontSize: '0.85rem',
    paddingLeft: '1.5rem',
  },
  treeLine: {
    display: 'flex',
    alignItems: 'center',
    marginBottom: '0.25rem',
  },
  treeIcon: {
    width: '20px',
    color: '#666',
    marginRight: '0.5rem',
  },
}

const BenchmarkContent: React.FC = () => (
  <>
    <div style={styles.section}>
      <div style={styles.sectionTitle}>
        <span style={styles.emoji}>📊</span> Метрики корреляции
      </div>
      <ul style={styles.list}>
        <li style={styles.listItem}>
          <strong>Корреляция Спирмена (ρ)</strong> — основная метрика
          <br />
          Измеряет монотонную зависимость между предсказаниями и экспертными оценками.
          <br />
          Значение от -1 до 1, где 1 = идеальная монотонная зависимость.
        </li>
        <li style={styles.listItem}>
          <strong>Корреляция Пирсона (r)</strong> — вспомогательная метрика
          <br />
          Измеряет линейную зависимость.
          <br />
          Может быть отрицательной, если связь обратная.
        </li>
      </ul>
    </div>

    <div style={styles.section}>
      <div style={styles.sectionTitle}>
        <span style={styles.emoji}>💡</span> Интерпретация
      </div>
      <ul style={styles.list}>
        <li style={styles.listItem}>
          <span style={styles.highlight}>ρ &gt; 0.7</span> — отличный результат
        </li>
        <li style={styles.listItem}>
          ρ = <strong>0.4–0.7</strong> — хороший
        </li>
        <li style={styles.listItem}>
          ρ = <strong>0.2–0.4</strong> — умеренный
        </li>
        <li style={styles.listItem}>
          ρ &lt; <strong>0.2</strong> — слабый
        </li>
      </ul>
    </div>

    <div style={styles.section}>
      <div style={styles.sectionTitle}>
        <span style={styles.emoji}>🎯</span> Лучший метод: SBERT
      </div>
      <p>
        Контекстные эмбеддинги показывают стабильно лучшие результаты
        на всех датасетах (hj-rg, simlex999_ru, simlex999).
      </p>
    </div>

    <div style={{ ...styles.section, ...styles.qa }}>
      <div style={styles.sectionTitle}>
        <span style={styles.emoji}>❓</span> Частые вопросы
      </div>
      <div style={styles.note}>
        <div style={styles.q}>Q: Почему TF-IDF показывает ~0 для английского?</div>
        <div style={styles.a}>
          A: Char n-grams не различают морфологически различные синонимы.
          <br />
          Например "smart" vs "intelligent" = 0.5 из-за отсутствия пересечения символов.
        </div>
      </div>
    </div>
  </>
)

const DomainsContent: React.FC = () => (
  <>
    <div style={styles.section}>
      <div style={styles.sectionTitle}>
        <span style={styles.emoji}>🏛️</span> Анализ предметных областей
      </div>
      <ul style={styles.list}>
        <li style={styles.listItem}>
          <strong>Что такое домен?</strong>
          <br />
          Домен — это предметная область (например "Биология", "Физика").
        </li>
        <li style={styles.listItem}>
          <strong>Как читать матрицу близости?</strong>
          <br />
          Значения от 0 (далёкие) до 1 (близкие).
          <br />
          Диагональ = 1.0 (домен сам с собой).
        </li>
      </ul>
    </div>

    <div style={styles.section}>
      <div style={styles.sectionTitle}>
        <span style={styles.emoji}>🎨</span> Цветовая шкала
      </div>
      <ul style={styles.list}>
        <li style={styles.listItem}>
          <span style={{ color: '#ef4444' }}>🔴 Красный</span> — близкие домены (высокая семантическая близость)
        </li>
        <li style={styles.listItem}>
          <span style={{ color: '#3b82f6' }}>🔵 Синий</span> — далёкие домены
        </li>
      </ul>
    </div>

    <div style={styles.note}>
      <div style={styles.noteTitle}>💡 Интерпретация</div>
      Одинаковые дисциплины (физика-математика) более близки,
      чем разные (биология-психология).
    </div>
  </>
)

const GraphContent: React.FC = () => (
  <>
    <div style={styles.section}>
      <div style={styles.sectionTitle}>
        <span style={styles.emoji}>🕸️</span> Визуальный граф доменов
      </div>
      <ul style={styles.list}>
        <li style={styles.listItem}>
          <strong>Узлы</strong> — домены (круги с названиями)
        </li>
        <li style={styles.listItem}>
          <strong>Связи</strong> — типы отношений:
          <ul style={styles.subList}>
            <li style={styles.listItem}>
              <span style={styles.treeIcon}>├─</span>
              <span style={{ borderBottom: '1px solid #6b7280', paddingBottom: '1px' }}>Сплошная линия</span> — similarity (семантическая близость)
            </li>
            <li style={styles.listItem}>
              <span style={styles.treeIcon}>└─</span>
              <span style={{ borderBottom: '1px dashed #6b7280', paddingBottom: '1px' }}>Пунктирная линия</span> — belongs_to (термин принадлежит домену)
            </li>
          </ul>
        </li>
      </ul>
    </div>

    <div style={styles.section}>
      <div style={styles.sectionTitle}>
        <span style={styles.emoji}>🎨</span> Цветовая легенда
      </div>
      <p>Каждый цвет = свой домен</p>
    </div>

    <div style={styles.section}>
      <div style={styles.sectionTitle}>
        <span style={styles.emoji}>🖱️</span> Управление
      </div>
      <ul style={styles.list}>
        <li style={styles.listItem}>
          <strong>Прокрутка</strong> — zoom (приблизить/отдалить)
        </li>
        <li style={styles.listItem}>
          <strong>Перетаскивание</strong> — move (двигать граф)
        </li>
        <li style={styles.listItem}>
          <strong>Hover</strong> — подсветка связанных узлов
        </li>
      </ul>
    </div>

    <div style={styles.note}>
      <div style={styles.noteTitle}>💡 Как читать</div>
      Близкие домены расположены рядом,
      толстые линии = сильная связь.
    </div>
  </>
)

const MatrixContent: React.FC = () => (
  <>
    <div style={styles.section}>
      <div style={styles.sectionTitle}>
        <span style={styles.emoji}>📊</span> Матрица корреляций методов
      </div>
      <ul style={styles.list}>
        <li style={styles.listItem}>
          <strong>Строки и столбцы</strong> — методы семантической близости
        </li>
        <li style={styles.listItem}>
          <strong>Значения</strong> — корреляция между предсказаниями пар методов
        </li>
        <li style={styles.listItem}>
          <strong>Диагональ</strong> = 1.0 (метод согласован сам с собой)
        </li>
      </ul>
    </div>

    <div style={styles.section}>
      <div style={styles.sectionTitle}>
        <span style={styles.emoji}>🎨</span> Цветовая шкала
      </div>
      <ul style={styles.list}>
        <li style={styles.listItem}>
          <span style={{ color: '#ef4444' }}>🔴 Красный/тёплый</span> — высокая корреляция (методы похожи)
        </li>
        <li style={styles.listItem}>
          <span style={{ color: '#3b82f6' }}>🔵 Синий/холодный</span> — низкая корреляция (методы различаются)
        </li>
      </ul>
    </div>

    <div style={styles.note}>
      <div style={styles.noteTitle}>💡 Что показывает</div>
      <ul style={styles.list}>
        <li style={styles.listItem}>
          SBERT-методы сильно коррелируют друг с другом
        </li>
        <li style={styles.listItem}>
          WordNet-методы образуют отдельный кластер
        </li>
        <li style={styles.listItem}>
          Гибридные методы связывают оба кластера
        </li>
      </ul>
    </div>
  </>
)

const modeConfigs: Record<InfoPanelMode, { icon: string; title: string; Content: React.FC }> = {
  benchmark: {
    icon: '📊',
    title: 'Метрики бенчмарка',
    Content: BenchmarkContent,
  },
  matrix: {
    icon: '📈',
    title: 'Матрица методов',
    Content: MatrixContent,
  },
  domains: {
    icon: '🏛️',
    title: 'Домены',
    Content: DomainsContent,
  },
  graph: {
    icon: '🕸️',
    title: 'Граф доменов',
    Content: GraphContent,
  },
}

export const InfoPanel: React.FC<InfoPanelProps> = ({ mode }) => {
  const config = modeConfigs[mode]
  const ContentComponent = config.Content

  return (
    <div style={styles.panel} className="info-panel">
      <div style={styles.header}>
        <span style={{ fontSize: '1.2rem' }}>{config.icon}</span>
        <h3 style={styles.title}>{config.title}</h3>
      </div>
      <ContentComponent />
    </div>
  )
}

export default InfoPanel
