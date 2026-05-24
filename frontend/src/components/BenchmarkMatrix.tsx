// frontend/src/components/BenchmarkMatrix.tsx
// Интегральная матрица результатов бенчмарков
//
// Версия: 1.1
// Обновлено: 2026-04-09

import { API_BASE_URL } from '@/services/api'
import { useState, useEffect, useRef, useCallback } from 'react'

/**
 * Типы для интегральной матрицы бенчмарков
 */
interface BenchmarkMatrixCell {
  spearman: number
  pearson: number
  missing: number
  predictions: number
}

interface BenchmarkMatrixRow {
  method: string
  hj_rg: BenchmarkMatrixCell | null      // Русский бенчмарк
  simlex999: BenchmarkMatrixCell | null  // English SimLex-999
  simlex999_rus: BenchmarkMatrixCell | null  // Russian SimLex-999
}

interface BenchmarkMatrixResponse {
  success: boolean
  results: BenchmarkMatrixRow[]
  execution_time_sec: number
  error: string | null
}

/**
 * Цветовая схема для методов
 */
const METHOD_COLORS: Record<string, string> = {
  'SBERT (baseline)': '#3B82F6',      // blue
  'SBERT + TF-IDF': '#10B981',         // green
  'SBERT + Z-score': '#8B5CF6',        // purple
  'RuWordNet (Lin)': '#F59E0B',        // amber
  'RuWordNet (Wu-Palmer)': '#EF4444',  // red
  'English WordNet (Lin)': '#06B6D4',  // cyan
  'English WordNet (Wu-Palmer)': '#84CC16', // lime
  'Hybrid (SBERT + RuWordNet)': '#EC4899', // pink
}

/**
 * Хук для безопасного fetch с AbortController
 */
function useAbortFetch() {
  const abortControllerRef = useRef<AbortController | null>(null)

  const fetchWithAbort = useCallback(async (
    url: string,
    options?: RequestInit,
    timeoutMs: number = 300000 // 5 минут по умолчанию для матрицы
  ): Promise<{ data: unknown; error: string | null }> => {
    // Отменяем предыдущий запрос если есть
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }

    // Создаём новый AbortController
    abortControllerRef.current = new AbortController()

    // Таймаут
    const timeoutId = setTimeout(() => {
      abortControllerRef.current?.abort()
    }, timeoutMs)

    try {
      const response = await fetch(url, {
        ...options,
        signal: abortControllerRef.current.signal,
      })

      clearTimeout(timeoutId)

      const data = await response.json()

      if (!response.ok) {
        return { data: null, error: data.detail || `HTTP ${response.status}` }
      }

      return { data, error: null }
    } catch (err) {
      clearTimeout(timeoutId)

      if (err instanceof Error) {
        if (err.name === 'AbortError') {
          return { data: null, error: 'Превышен таймаут ожидания. Попробуйте ещё раз.' }
        }
        return { data: null, error: err.message }
      }

      return { data: null, error: 'Неизвестная ошибка' }
    }
  }, [])

  const abort = useCallback(() => {
    abortControllerRef.current?.abort()
  }, [])

  return { fetchWithAbort, abort }
}

/**
 * Компонент интегральной матрицы бенчмарков
 */
export const BenchmarkMatrix: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [running, _setRunning] = useState(false)
  const [matrixData, setMatrixData] = useState<BenchmarkMatrixRow[]>([])
  const [executionTime, setExecutionTime] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [selectedMetric, setSelectedMetric] = useState<'spearman' | 'pearson'>('spearman')

  const { fetchWithAbort, abort } = useAbortFetch()

  // Загрузка матрицы
  const loadMatrix = async () => {
    setLoading(true)
    setError(null)

    try {
      const { data, error: fetchError } = await fetchWithAbort(
        `${API_BASE_URL}/api/v1/benchmark/matrix`,
        {},
        600000 // 10 минут для матрицы
      )

      if (fetchError) {
        setError(fetchError)
        return
      }

      const matrixResponse = data as BenchmarkMatrixResponse

      if (matrixResponse.success) {
        setMatrixData(matrixResponse.results)
        setExecutionTime(matrixResponse.execution_time_sec)
      } else {
        setError(matrixResponse.error || 'Ошибка загрузки матрицы')
      }
    } catch (err) {
      console.error('Ошибка загрузки матрицы:', err)
      setError('Не удалось загрузить матрицу результатов')
    } finally {
      setLoading(false)
    }
  }

  // Начальная загрузка
  useEffect(() => {
    loadMatrix()
  }, [])

  // Построение тепловой карты
  const getHeatColor = (value: number | undefined, max: number): string => {
    if (value === undefined || value === 0) return '#E5E7EB'

    const ratio = value / max
    // Градиент от красного (0) через жёлтый (0.5) к зелёному (1)
    if (ratio < 0.5) {
      // Красный -> Жёлтый
      const r = Math.round(239 + (69 - 239) * (ratio * 2))
      const g = Math.round(68 + (196 - 68) * (ratio * 2))
      const b = Math.round(68 + (18 - 68) * (ratio * 2))
      return `rgb(${r}, ${g}, ${b})`
    } else {
      // Жёлтый -> Зелёный
      const r = Math.round(69 + (16 - 69) * ((ratio - 0.5) * 2))
      const g = Math.round(196 + (163 - 196) * ((ratio - 0.5) * 2))
      const b = Math.round(18 + (86 - 18) * ((ratio - 0.5) * 2))
      return `rgb(${r}, ${g}, ${b})`
    }
  }

  // Максимальное значение для нормализации (все 3 колонки)
  const maxSpearman = Math.max(
    ...matrixData.flatMap(row => [
      row.hj_rg?.spearman || 0,
      row.simlex999?.spearman || 0,
      row.simlex999_rus?.spearman || 0,
    ])
  )

  const maxPearson = Math.max(
    ...matrixData.flatMap(row => [
      row.hj_rg?.pearson || 0,
      row.simlex999?.pearson || 0,
      row.simlex999_rus?.pearson || 0,
    ])
  )

  const maxValue = selectedMetric === 'spearman' ? maxSpearman : maxPearson

  // Находим лучший метод
  const getBestMethod = (): BenchmarkMatrixRow | null => {
    if (matrixData.length === 0) return null

    let best: BenchmarkMatrixRow | null = null
    let bestScore = -1

    for (const row of matrixData) {
      const hjScore = row.hj_rg?.[selectedMetric] || 0
      const simlexScore = row.simlex999?.[selectedMetric] || 0
      const simlexRusScore = row.simlex999_rus?.[selectedMetric] || 0
      const avgScore = (hjScore + simlexScore + simlexRusScore) / 3

      if (avgScore > bestScore) {
        bestScore = avgScore
        best = row
      }
    }

    return best
  }

  const bestMethod = getBestMethod()

  return (
    <div className="benchmark-matrix">
      <h2>📊 Интегральная матрица бенчмарков</h2>

      {/* Описание */}
      <div className="matrix-description">
        <p>
          Матрица сравнивает методы семантической близости на трёх датасетах:
        </p>
        <ul>
          <li><strong>hj-rg</strong> — русский бенчмарк (Rubachev et al., 2022)</li>
          <li><strong>SimLex-999 (EN)</strong> — английский бенчмарк (Hill et al., 2015)</li>
          <li><strong>SimLex-999 (RU)</strong> — русский бенчмарк (перевод SimLex-999)</li>
        </ul>
      </div>

      {/* Управление */}
      <div className="matrix-controls">
        <div className="metric-selector">
          <label>Метрика:</label>
          <select
            value={selectedMetric}
            onChange={(e) => setSelectedMetric(e.target.value as 'spearman' | 'pearson')}
            disabled={loading || running}
          >
            <option value="spearman">Корреляция Спирмена</option>
            <option value="pearson">Корреляция Пирсона</option>
          </select>
        </div>

        <button
          onClick={loadMatrix}
          disabled={loading || running}
          className="refresh-button"
        >
          {loading ? '⏳ Загрузка...' : '🔄 Обновить матрицу'}
        </button>

        {running && (
          <button onClick={abort} className="cancel-button">
            Отменить
          </button>
        )}
      </div>

      {/* Ошибка */}
      {error && (
        <div className="matrix-error">
          ❌ {error}
        </div>
      )}

      {/* Загрузка */}
      {(loading || running) && (
        <div className="matrix-loading">
          <span className="loading-spinner">⚙️</span>
          <span>Выполнение бенчмарков на всех датасетах...</span>
          <span className="loading-hint">(Это может занять 3-5 минут)</span>
        </div>
      )}

      {/* Матрица */}
      {!loading && !running && matrixData.length > 0 && (
        <div className="matrix-container">
          {/* Информация о выполнении */}
          <div className="matrix-meta">
            Время выполнения: {executionTime.toFixed(1)}с
          </div>

          {/* Таблица с 3 датасетами */}
          <table className="matrix-table">
            <thead>
              <tr>
                <th className="method-header">Метод</th>
                <th className="dataset-header">hj-rg (RU)</th>
                <th className="dataset-header">SimLex-999 (EN)</th>
                <th className="dataset-header">SimLex-999 (RU)</th>
                <th className="avg-header">Среднее</th>
              </tr>
            </thead>
            <tbody>
              {matrixData.map((row) => {
                const hjValue = row.hj_rg?.[selectedMetric] || 0
                const simlexValue = row.simlex999?.[selectedMetric] || 0
                const simlexRusValue = row.simlex999_rus?.[selectedMetric] || 0
                const avgValue = (hjValue + simlexValue + simlexRusValue) / 3
                const isBest = bestMethod?.method === row.method

                return (
                  <tr key={row.method} className={isBest ? 'best-row' : ''}>
                    <td className="method-cell">
                      <span
                        className="method-indicator"
                        style={{ backgroundColor: METHOD_COLORS[row.method] || '#6B7280' }}
                      />
                      <span className="method-name">
                        {row.method}
                        {isBest && <span className="best-badge">🏆</span>}
                      </span>
                    </td>
                    <td
                      className="value-cell"
                      style={{ backgroundColor: getHeatColor(hjValue, maxValue) }}
                    >
                      {row.hj_rg ? (
                        <>
                          <span className="value-main">{(hjValue * 100).toFixed(1)}%</span>
                          <span className="value-meta">
                            ({row.hj_rg.predictions} предск.)
                          </span>
                        </>
                      ) : (
                        <span className="value-missing">N/A</span>
                      )}
                    </td>
                    <td
                      className="value-cell"
                      style={{ backgroundColor: getHeatColor(simlexValue, maxValue) }}
                    >
                      {row.simlex999 ? (
                        <>
                          <span className="value-main">{(simlexValue * 100).toFixed(1)}%</span>
                          <span className="value-meta">
                            ({row.simlex999.predictions} предск.)
                          </span>
                        </>
                      ) : (
                        <span className="value-missing">N/A</span>
                      )}
                    </td>
                    <td
                      className="value-cell"
                      style={{ backgroundColor: getHeatColor(simlexRusValue, maxValue) }}
                    >
                      {row.simlex999_rus ? (
                        <>
                          <span className="value-main">{(simlexRusValue * 100).toFixed(1)}%</span>
                          <span className="value-meta">
                            ({row.simlex999_rus.predictions} предск.)
                          </span>
                        </>
                      ) : (
                        <span className="value-missing">N/A</span>
                      )}
                    </td>
                    <td className="avg-cell">
                      {(avgValue * 100).toFixed(1)}%
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>

          {/* Легенда */}
          <div className="matrix-legend">
            <span className="legend-title">Шкала:</span>
            <div className="legend-gradient">
              <span>0%</span>
              <div className="gradient-bar" />
              <span>100%</span>
            </div>
          </div>

          {/* Лучший метод */}
          {bestMethod && (
            <div className="matrix-summary">
              🏆 Лучший метод: <strong>{bestMethod.method}</strong>
              {' '}(средняя {selectedMetric === 'spearman' ? 'корреляция Спирмена' : 'корреляция Пирсона'}:
              {' '}
              <strong>
                {(((bestMethod.hj_rg?.[selectedMetric] || 0) + (bestMethod.simlex999?.[selectedMetric] || 0) + (bestMethod.simlex999_rus?.[selectedMetric] || 0)) / 3 * 100).toFixed(1)}%
              </strong>
              )
            </div>
          )}

          {/* Примечание о языке */}
          <div className="matrix-notes">
            <h4>Примечание о методах и языках:</h4>
            <ul>
              <li>
                <strong>SBERT методы</strong> — используют эмбеддинги для расчёта косинусной близости
                <ul>
                  <li>На русских датасетах (hj-rg, SimLex-999 RU): русская модель ai-forever/sbert_large_nlu_ru</li>
                  <li>На английском датасете (SimLex-999 EN): мультиязычная модель paraphrase-multilingual-MiniLM-L12-v2</li>
                </ul>
              </li>
              <li>
                <strong>RuWordNet методы</strong> — работают только на русских датасетах
                (hj-rg, SimLex-999 RU), возвращают N/A для английского
              </li>
              <li>
                <strong>English WordNet методы</strong> — работают только на английском датасете
                (SimLex-999 EN), возвращают N/A для русских
              </li>
              <li>
                <strong>Hybrid</strong> — комбинирует SBERT и RuWordNet для русского,
                только SBERT для английского
              </li>
            </ul>
          </div>
        </div>
      )}

      {/* Пустое состояние */}
      {!loading && !running && matrixData.length === 0 && !error && (
        <div className="matrix-empty">
          <p>Нажмите "Обновить матрицу" для загрузки результатов.</p>
        </div>
      )}
    </div>
  )
}
