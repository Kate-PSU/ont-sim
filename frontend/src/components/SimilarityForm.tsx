// frontend/src/components/SimilarityForm.tsx
// Форма для расчёта близости между доменами
//
// Версия: 1.0
// Обновлено: 2026-04-06

import { API_BASE_URL } from '@/services/api'
import { useState, useEffect } from 'react'

interface SimilarityResult {
  domain1: string
  domain2: string
  score: number
  metric: string
}

interface SimilarityFormProps {
  domains: string[]
}

export function SimilarityForm({ domains }: SimilarityFormProps) {
  const [domain1, setDomain1] = useState('')
  const [domain2, setDomain2] = useState('')
  const [metric, setMetric] = useState<'cosine' | 'euclidean'>('cosine')
  const [result, setResult] = useState<SimilarityResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Очистка результата при изменении выбора
  useEffect(() => {
    setResult(null)
    setError(null)
  }, [domain1, domain2, metric])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!domain1 || !domain2) {
      setError('Выберите оба домена')
      return
    }

    if (domain1 === domain2) {
      setError('Домены должны быть разными')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/v1/domains/${encodeURIComponent(domain1)}/similarity/${encodeURIComponent(domain2)}?metric=${metric}`
      )

      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.detail || 'Ошибка расчёта')
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Неизвестная ошибка')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="similarity-form">
      <h2>Расчёт близости доменов</h2>

      <form onSubmit={handleSubmit}>
        <div className="form-row">
          <label>
            Домен 1:
            <select
              value={domain1}
              onChange={(e) => setDomain1(e.target.value)}
              disabled={loading}
            >
              <option value="">-- Выберите домен --</option>
              {domains.map((d) => (
                <option key={d} value={d} disabled={d === domain2}>
                  {d}
                </option>
              ))}
            </select>
          </label>

          <span className="vs">vs</span>

          <label>
            Домен 2:
            <select
              value={domain2}
              onChange={(e) => setDomain2(e.target.value)}
              disabled={loading}
            >
              <option value="">-- Выберите домен --</option>
              {domains.map((d) => (
                <option key={d} value={d} disabled={d === domain1}>
                  {d}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="form-row">
          <label>
            Метрика:
            <select
              value={metric}
              onChange={(e) => setMetric(e.target.value as 'cosine' | 'euclidean')}
              disabled={loading}
            >
              <option value="cosine">Косинусная</option>
              <option value="euclidean">Евклидова</option>
            </select>
          </label>

          <button type="submit" disabled={loading || !domain1 || !domain2}>
            {loading ? 'Расчёт...' : 'Рассчитать'}
          </button>
        </div>
      </form>

      {error && (
        <div className="error-message">
          ❌ {error}
        </div>
      )}

      {result && (
        <div className="result-card">
          <h3>Результат</h3>
          <div className="result-content">
            <div className="domain-names">
              <span className="domain">{result.domain1}</span>
              <span className="connector">↔</span>
              <span className="domain">{result.domain2}</span>
            </div>
            <div className="score-display">
              <span className="score-label">Близость:</span>
              <span className="score-value">
                {(result.score * 100).toFixed(2)}%
              </span>
            </div>
            <div className="metric-badge">
              {result.metric === 'cosine' ? 'Косинусная' : 'Евклидова'}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
