// frontend/src/hooks/useServiceState.ts
// Единый хук для управления состоянием сервиса
//
// Версия: 1.0
// Обновлено: 2026-04-09

import { API_BASE_URL } from '@/services/api'
import { useState, useEffect, useCallback, useRef } from 'react'

/**
 * Типы данных
 */
export interface MethodResult {
  method: string
  spearman: number
  pearson: number
  mse: number
  missing: number
  predictions_count: number
}

export interface DatasetStatus {
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  results: Record<string, MethodResult> | null
  saved_at: string | null
  error: string | null
}

export interface ActiveTask {
  id: string
  dataset: string
  method: string
  progress: number
  status: string
}

export interface SystemState {
  system_status: 'ready' | 'busy' | 'error'
  busy_reason: string | null
  domains_loaded: boolean
  domains_count: number
  datasets: Record<string, DatasetStatus>
  active_tasks: ActiveTask[]
}

export interface BenchmarkRunRequest {
  dataset: string
  method: string
  force_recalculate?: boolean
}

export interface BenchmarkRunResponse {
  success: boolean
  task_id: string | null
  error: string | null
  active_task: ActiveTask | null
}

export interface SSEEvent {
  event: string
  task_id: string
  dataset: string
  method: string
  progress?: number
  status?: string
  error?: string
}

/**
 * Хук для управления состоянием сервиса
 */
export function useServiceState() {
  // Состояние
  const [state, setState] = useState<SystemState | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [subscribed, setSubscribed] = useState(false)
  // Глобальная блокировка интерфейса при загрузке
  const [globalLoading, setGlobalLoading] = useState(false)

  // Ref для EventSource
  const eventSourceRef = useRef<EventSource | null>(null)
  const abortControllerRef = useRef<AbortController | null>(null)

  /**
   * Загрузка состояния системы
   */
  const fetchState = useCallback(async () => {
    setLoading(true)
    setError(null)

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/state`, {
        signal: abortControllerRef.current?.signal,
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const data: SystemState = await response.json()
      setState(data)

      // Если система занята - подписываемся на SSE
      if (data.system_status === 'busy' && !subscribed) {
        subscribeToEvents()
      }
    } catch (err) {
      if (err instanceof Error && err.name !== 'AbortError') {
        setError(err.message)
      }
    } finally {
      setLoading(false)
    }
  }, [subscribed])

  /**
   * Подписка на SSE события
   */
  const subscribeToEvents = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
    }

    const eventSource = new EventSource(`${API_BASE_URL}/api/v1/tasks/stream`)
    eventSourceRef.current = eventSource
    setSubscribed(true)

    eventSource.onmessage = (event) => {
      try {
        const data: SSEEvent = JSON.parse(event.data)

        // Обновляем состояние на основе события
        setState((prev) => {
          if (!prev) return prev

          if (data.event === 'completed' || data.event === 'failed') {
            // Завершена задача - обновляем состояние
            return {
              ...prev,
              datasets: {
                ...prev.datasets,
                [data.dataset]: {
                  ...prev.datasets[data.dataset],
                  status: data.event === 'completed' ? 'completed' : 'failed',
                  progress: data.event === 'completed' ? 1.0 : 0,
                  error: data.error || null,
                },
              },
            }
          }

          if (data.event === 'progress') {
            // Обновляем прогресс
            return {
              ...prev,
              datasets: {
                ...prev.datasets,
                [data.dataset]: {
                  ...prev.datasets[data.dataset],
                  status: 'running',
                  progress: data.progress || 0,
                },
              },
            }
          }

          return prev
        })
      } catch (err) {
        console.error('Ошибка парсинга SSE события:', err)
      }
    }

    eventSource.onerror = () => {
      console.error('SSE ошибка, переподключение через 5 секунд...')
      setSubscribed(false)
      setTimeout(subscribeToEvents, 5000)
    }
  }, [])

  /**
   * Запуск бенчмарка
   */
  const runBenchmark = useCallback(async (request: BenchmarkRunRequest): Promise<BenchmarkRunResponse> => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/benchmark/run-safe`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      })

      const data: BenchmarkRunResponse = await response.json()

      if (data.success) {
        // Если task_id отсутствует — результаты из кэша, НЕ меняем статус
        if (data.task_id) {
          // Подписываемся на события если ещё не подписаны
          if (!subscribed) {
            subscribeToEvents()
          }

          // Обновляем состояние — запускается реальная задача
          setState((prev) => {
            if (!prev) return prev
            return {
              ...prev,
              datasets: {
                ...prev.datasets,
                [request.dataset]: {
                  ...prev.datasets[request.dataset],
                  status: 'running',
                  progress: 0,
                },
              },
            }
          })
        } else {
          console.log(`[useServiceState] Результаты для ${request.dataset} из кэша, не запускаем задачу`)
        }
      }

      return data
    } catch (err) {
      return {
        success: false,
        task_id: null,
        error: err instanceof Error ? err.message : 'Unknown error',
        active_task: null,
      }
    }
  }, [subscribed, subscribeToEvents])

  /**
   * Перезагрузка состояния
   */
  const refresh = useCallback(() => {
    fetchState()
  }, [fetchState])

  /**
   * Начальная загрузка
   */
  useEffect(() => {
    fetchState()

    return () => {
      // Cleanup
      if (eventSourceRef.current) {
        eventSourceRef.current.close()
      }
      if (abortControllerRef.current) {
        abortControllerRef.current.abort()
      }
    }
  }, []) // Пустой массив - только при mount

  return {
    state,
    loading,
    error,
    subscribed,
    globalLoading,
    setGlobalLoading,
    runBenchmark,
    refresh,
    subscribeToEvents,
  }
}

/**
 * Вспомогательные функции
 */
export function getBestByAccuracy(datasets: Record<string, DatasetStatus>): { method: string; avgSpearman: number }[] {
  const methodScores: Record<string, number[]> = {}

  for (const dataset of Object.values(datasets)) {
    if (dataset.status === 'completed' && dataset.results) {
      for (const [method, result] of Object.entries(dataset.results)) {
        if (!methodScores[method]) {
          methodScores[method] = []
        }
        methodScores[method].push(result.spearman)
      }
    }
  }

  return Object.entries(methodScores)
    .map(([method, scores]) => ({
      method,
      avgSpearman: scores.reduce((a, b) => a + b, 0) / scores.length,
    }))
    .sort((a, b) => b.avgSpearman - a.avgSpearman)
    .slice(0, 3)
}

export function getBestBySpeed(_datasets: Record<string, DatasetStatus>): { method: string; avgTime: number }[] {
  // TODO: Implement when we store execution times
  return []
}
