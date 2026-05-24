// frontend/src/components/BenchmarkPanel.tsx
// Панель бенчмаркинга методов
//
// Версия: 1.3
// Обновлено: 2026-04-09
// Особенности: IndexedDB кеширование, polling результатов через /tasks/{id}/result

import { useState, useEffect, useRef, useCallback } from 'react'
import { indexedDBService, BenchmarkComparison } from '../services/IndexedDBService'
import { API_BASE_URL } from '@/services/api'

/**
 * Типы для данных бенчмарка
 */
interface Dataset {
  name: string
  path: string
  size: number
}

interface BenchmarkResponse {
  success: boolean
  comparison: BenchmarkComparison | null
  error: string | null
}

/**
 * Цветовая схема для методов
 */
const METHOD_COLORS: Record<string, string> = {
  'SBERT (baseline)': '#3B82F6',           // blue
  'SBERT + TF-IDF': '#10B981',             // green
  'SBERT + Z-score': '#8B5CF6',            // purple
  'RuWordNet (Lin)': '#F59E0B',            // amber
  'RuWordNet (Wu-Palmer)': '#EF4444',       // red
  'English WordNet (Lin)': '#14B8A6',      // teal
  'English WordNet (Wu-Palmer)': '#F97316', // orange
  'Hybrid (SBERT + RuWordNet)': '#EC4899',  // pink
}

/**
 * Хук для безопасного fetch с AbortController
 */
function useAbortFetch() {
  const abortControllerRef = useRef<AbortController | null>(null)

  const fetchWithAbort = useCallback(async (
    url: string,
    options?: RequestInit,
    timeoutMs: number = 120000
  ): Promise<{ data: unknown; error: string | null }> => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }

    abortControllerRef.current = new AbortController()

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

export const BenchmarkPanel: React.FC = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [selectedDataset, setSelectedDataset] = useState<string>('')
  const [loading, setLoading] = useState(false)
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<BenchmarkComparison | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [systemBusy, setSystemBusy] = useState(false)
  const [fromCache, setFromCache] = useState(false)

  // Async task state
  const [taskId, setTaskId] = useState<string | null>(null)
  const [taskStatus, setTaskStatus] = useState<string>('')
  const [taskProgress, setTaskProgress] = useState<number>(0)

  const { fetchWithAbort } = useAbortFetch()

  // Проверка статуса системы перед операциями
  const checkSystemStatus = useCallback(async (): Promise<boolean> => {
    try {
      const { data } = await fetchWithAbort(`${API_BASE_URL}/api/v1/status`, {}, 5000)
      if (data && (data as { busy: boolean }).busy) {
        setSystemBusy(true)
        setError('Система занята выполнением другой операции. Подождите...')
        return false
      }
      setSystemBusy(false)
      return true
    } catch {
      return true
    }
  }, [fetchWithAbort])

  // Загрузка кешированного результата
  const loadCachedResult = useCallback(async (dataset: string) => {
    try {
      const cached = await indexedDBService.getBenchmark(dataset)
      if (cached) {
        console.log('[BenchmarkPanel] Загружено из IndexedDB:', dataset)
        setResult(cached)
        setFromCache(true)
      }
    } catch (err) {
      console.error('Ошибка загрузки из IndexedDB:', err)
    }
  }, [])

  // Загрузка списка датасетов и кешированных результатов
  useEffect(() => {
    const init = async () => {
      try {
        // Инициализируем IndexedDB
        await indexedDBService.init()

        // Загружаем список датасетов
        await fetchDatasets()
      } catch (err) {
        console.error('Ошибка инициализации:', err)
      }
    }

    init()
  }, [])

  // При изменении выбранного датасета проверяем кеш
  useEffect(() => {
    if (selectedDataset) {
      setFromCache(false)
      setResult(null)
      loadCachedResult(selectedDataset)
    }
  }, [selectedDataset, loadCachedResult])

  // Загрузка списка датасетов
  const fetchDatasets = async () => {
    setLoading(true)
    setError(null)
    try {
      const { data, error: fetchError } = await fetchWithAbort(`${API_BASE_URL}/api/v1/benchmark/datasets`, {}, 30000)
      if (fetchError) {
        setError(fetchError)
        return
      }
      const resultData = data as { datasets: Dataset[] }
      setDatasets(resultData.datasets || [])
      if (resultData.datasets && resultData.datasets.length > 0 && !selectedDataset) {
        setSelectedDataset(resultData.datasets[0].name)
      }
    } catch (err) {
      console.error('Ошибка загрузки датасетов:', err)
      setError('Не удалось загрузить список датасетов')
    } finally {
      setLoading(false)
    }
  }

  // Сохранение результата в IndexedDB
  const saveResult = useCallback(async (dataset: string, comparison: BenchmarkComparison) => {
    try {
      await indexedDBService.saveBenchmark(dataset, comparison)
      console.log('[BenchmarkPanel] Сохранено в IndexedDB:', dataset)
    } catch (err) {
      console.error('Ошибка сохранения в IndexedDB:', err)
    }
  }, [])

  const runBenchmark = async () => {
    if (!selectedDataset) return

    const canProceed = await checkSystemStatus()
    if (!canProceed) return

    setRunning(true)
    setError(null)
    setFromCache(false)
    setResult(null)

    try {
      const { data, error: fetchError } = await fetchWithAbort(
        `${API_BASE_URL}/api/v1/benchmark/run`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ dataset: selectedDataset }),
        },
        180000
      )

      if (fetchError) {
        setError(fetchError)
        return
      }

      const benchmarkData = data as BenchmarkResponse

      if (benchmarkData.success && benchmarkData.comparison) {
        setResult(benchmarkData.comparison)
        // Сохраняем в IndexedDB
        await saveResult(selectedDataset, benchmarkData.comparison)
      } else {
        setError(benchmarkData.error || 'Ошибка выполнения бенчмарка')
      }
    } catch (err) {
      console.error('Ошибка запуска бенчмарка:', err)
      setError('Не удалось выполнить бенчмарк')
    } finally {
      setRunning(false)
    }
  }

  // Polling для статуса задачи через /tasks/{id}/result
  useEffect(() => {
    if (!taskId || taskStatus === 'SUCCESS' || taskStatus === 'FAILURE') {
      return
    }

    const pollInterval = setInterval(async () => {
      try {
        const { data } = await fetchWithAbort(`${API_BASE_URL}/api/v1/tasks/${taskId}/result`, {}, 10000)
        if (data) {
          const taskData = data as {
            status: string
            result?: BenchmarkComparison
            progress?: number
            status_message?: string
            error?: string
          }

          setTaskStatus(taskData.status)

          if (taskData.status === 'SUCCESS' && taskData.result) {
            setResult(taskData.result)
            setFromCache(false)
            await saveResult(selectedDataset, taskData.result)
            setRunning(false)
            setTaskId(null)
            setTaskStatus('')
          } else if (taskData.status === 'FAILURE') {
            setError(taskData.error || 'Задача не выполнена')
            setRunning(false)
            setTaskId(null)
            setTaskStatus('')
          } else if (taskData.status === 'PROGRESS') {
            setTaskProgress(taskData.progress || 0)
            setTaskStatus(taskData.status_message || 'Обработка...')
          }
        }
      } catch (err) {
        console.error('Ошибка опроса результата:', err)
      }
    }, 2000)

    return () => clearInterval(pollInterval)
  }, [taskId, taskStatus, selectedDataset, saveResult, fetchWithAbort])

  // Запуск асинхронного бенчмарка
  const runBenchmarkAsync = async () => {
    if (!selectedDataset) return

    setRunning(true)
    setError(null)
    setFromCache(false)
    setResult(null)

    try {
      const { data, error: fetchError } = await fetchWithAbort(
        `${API_BASE_URL}/api/v1/tasks/benchmark`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ dataset: selectedDataset }),
        },
        30000
      )

      if (fetchError) {
        console.log('Async не доступен, используем sync...')
        runBenchmark()
        return
      }

      const taskData = data as { task_id: string }
      setTaskId(taskData.task_id)
      setTaskStatus('PENDING')
      setTaskProgress(0)
    } catch {
      runBenchmark()
    }
  }

  // Находим лучший метод по Спирмену
  const bestMethod = result?.results.reduce((best, current) =>
    current.spearman > best.spearman ? current : best
    , result.results[0])

  return (
    <div className="benchmark-panel">
      <h2>📊 Сравнение методов на бенчмарках</h2>

      {/* Выбор датасета и запуск */}
      <div className="benchmark-controls">
        <div className="dataset-selector">
          <label htmlFor="dataset-select">Датасет:</label>
          <select
            id="dataset-select"
            value={selectedDataset}
            onChange={(e) => setSelectedDataset(e.target.value)}
            disabled={loading || running}
          >
            {datasets.map((ds) => (
              <option key={ds.name} value={ds.name}>
                {ds.name} ({ds.size} пар)
              </option>
            ))}
          </select>
        </div>

        <button
          onClick={runBenchmarkAsync}
          disabled={loading || running || !selectedDataset}
          className="run-button"
        >
          {running ? '⏳ Запуск...' : '▶ Запустить бенчмарк'}
        </button>
      </div>

      {/* Индикатор загрузки из кеша */}
      {fromCache && !running && (
        <div className="benchmark-info">
          📋 Результаты загружены из кеша. Нажмите "Запустить" для обновления.
        </div>
      )}

      {/* Прогресс задачи */}
      {running && taskId && (
        <div className="benchmark-progress">
          <div className="progress-info">
            <span>⏳ {taskStatus || 'Обработка...'}</span>
            <span>{taskProgress}%</span>
          </div>
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ width: `${taskProgress}%` }}
            />
          </div>
        </div>
      )}

      {/* Система занята */}
      {systemBusy && (
        <div className="benchmark-warning">
          ⏳ Система занята другой операцией. Подождите завершения...
        </div>
      )}

      {/* Ошибка */}
      {error && (
        <div className="benchmark-error">
          ❌ {error}
          {running && (
            <button
              className="cancel-button"
              onClick={() => {
                setRunning(false)
                setError(null)
                setTaskId(null)
              }}
            >
              Отменить
            </button>
          )}
        </div>
      )}

      {/* Результаты */}
      {result && (
        <div className="benchmark-results">
          <div className="results-header">
            <h3>Результаты: {result.dataset_name}</h3>
            <span className="execution-time">
              Время: {result.execution_time_sec.toFixed(1)}с
            </span>
          </div>

          {/* Таблица результатов */}
          <table className="results-table">
            <thead>
              <tr>
                <th>Метод</th>
                <th>Спирмен ⬆</th>
                <th>Пирсон</th>
                <th>MSE ⬇</th>
                <th>Пропусков</th>
              </tr>
            </thead>
            <tbody>
              {result.results.map((r) => (
                <tr
                  key={r.method}
                  className={bestMethod?.method === r.method ? 'best-method' : ''}
                >
                  <td>
                    <span
                      className="method-indicator"
                      style={{ backgroundColor: METHOD_COLORS[r.method] || '#6B7280' }}
                    />
                    {r.method}
                  </td>
                  <td className="metric-spearman">
                    {(r.spearman * 100).toFixed(1)}%
                    {bestMethod?.method === r.method && <span className="best-badge">🏆</span>}
                  </td>
                  <td>{(r.pearson * 100).toFixed(1)}%</td>
                  <td>{r.mse.toFixed(4)}</td>
                  <td>{r.missing} / {result.dataset_size}</td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* Bar chart для Спирмена */}
          <div className="chart-container">
            <h4>Корреляция Спирмена по методам</h4>
            <div className="bar-chart">
              {result.results.map((r) => (
                <div key={r.method} className="bar-row">
                  <span className="bar-label">{r.method.replace('SBERT', 'SB')}</span>
                  <div className="bar-track">
                    <div
                      className="bar-fill"
                      style={{
                        width: `${Math.max(r.spearman * 100, 5)}%`,
                        backgroundColor: METHOD_COLORS[r.method] || '#6B7280',
                      }}
                    />
                  </div>
                  <span className="bar-value">{(r.spearman * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>

          {/* Лучший метод */}
          {bestMethod && (
            <div className="best-method-summary">
              🏆 Лучший метод: <strong>{bestMethod.method}</strong>
              {' '}с корреляцией Спирмена <strong>{(bestMethod.spearman * 100).toFixed(1)}%</strong>
            </div>
          )}
        </div>
      )}

      {/* Загрузка */}
      {(loading || (running && !taskId)) && (
        <div className="benchmark-loading">
          {loading ? (
            '📡 Загрузка датасетов...'
          ) : (
            <>
              <span className="loading-spinner">⚙️</span>
              {' '}Выполнение бенчмарка (это может занять 1-2 минуты)...
            </>
          )}
        </div>
      )}

      {/* Пустое состояние */}
      {!loading && !running && !result && !error && (
        <div className="benchmark-empty">
          <p>Выберите датасет и нажмите "Запустить бенчмарк" для сравнения методов.</p>
        </div>
      )}
    </div>
  )
}