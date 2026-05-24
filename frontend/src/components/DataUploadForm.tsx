// frontend/src/components/DataUploadForm.tsx
// Форма для загрузки данных
//
// Версия: 1.0
// Обновлено: 2026-04-06

import { API_BASE_URL } from '@/services/api'
import { useState } from 'react'

// Демо-данные для загрузки
const DEMO_DATA = {
  domains: [
    {
      name: "ML",
      description: "Machine Learning and Artificial Intelligence",
      terms: [
        { "name": "машинное обучение", "frequency": 150 },
        { "name": "нейронная сеть", "frequency": 120 },
        { "name": "обучение с учителем", "frequency": 95 },
        { "name": "глубокое обучение", "frequency": 110 },
        { "name": "обратное распространение", "frequency": 78 },
        { "name": "стохастический градиент", "frequency": 62 },
        { "name": "свёрточная сеть", "frequency": 88 },
        { "name": "рекуррентная сеть", "frequency": 74 },
        { "name": "трансформер", "frequency": 91 },
        { "name": "внимание (attention)", "frequency": 67 }
      ]
    },
    {
      name: "Biology",
      description: "Molecular and Cell Biology",
      terms: [
        { "name": "белок", "frequency": 80 },
        { "name": "ген", "frequency": 95 },
        { "name": "клетка", "frequency": 70 },
        { "name": "ДНК", "frequency": 88 },
        { "name": "РНК", "frequency": 62 },
        { "name": "фермент", "frequency": 55 },
        { "name": "рецептор", "frequency": 48 },
        { "name": "антитело", "frequency": 52 },
        { "name": "метаболизм", "frequency": 60 },
        { "name": "транскрипция", "frequency": 44 }
      ]
    },
    {
      name: "Mathematics",
      description: "Pure and Applied Mathematics",
      terms: [
        { "name": "интеграл", "frequency": 65 },
        { "name": "производная", "frequency": 72 },
        { "name": "матрица", "frequency": 58 },
        { "name": "вектор", "frequency": 63 },
        { "name": "собственное значение", "frequency": 40 },
        { "name": "дифференциал", "frequency": 38 },
        { "name": "топология", "frequency": 35 },
        { "name": "алгебра", "frequency": 50 },
        { "name": "вероятность", "frequency": 45 },
        { "name": "статистика", "frequency": 55 }
      ]
    },
    {
      name: "Physics",
      description: "Classical and Modern Physics",
      terms: [
        { "name": "энергия", "frequency": 90 },
        { "name": "сила", "frequency": 85 },
        { "name": "поле", "frequency": 78 },
        { "name": "квант", "frequency": 68 },
        { "name": "фотон", "frequency": 55 },
        { "name": "электрон", "frequency": 60 },
        { "name": "протон", "frequency": 48 },
        { "name": "нейтрон", "frequency": 45 },
        { "name": "гравитация", "frequency": 40 },
        { "name": "теория струн", "frequency": 30 }
      ]
    },
    {
      name: "Chemistry",
      description: "General and Organic Chemistry",
      terms: [
        { "name": "молекула", "frequency": 75 },
        { "name": "реакция", "frequency": 82 },
        { "name": "валентность", "frequency": 50 },
        { "name": "катализатор", "frequency": 55 },
        { "name": "pH", "frequency": 42 },
        { "name": "раствор", "frequency": 38 },
        { "name": "осаждение", "frequency": 35 },
        { "name": "полимер", "frequency": 48 },
        { "name": "ион", "frequency": 52 },
        { "name": "стехиометрия", "frequency": 30 }
      ]
    }
  ]
}

interface DataUploadFormProps {
  onDataLoaded: () => void
}

export function DataUploadForm({ onDataLoaded }: DataUploadFormProps) {
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null)

  const loadDemoData = async () => {
    setLoading(true)
    setMessage(null)

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/upload/json`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(DEMO_DATA),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Ошибка загрузки данных')
      }

      setMessage({
        type: 'success',
        text: `Загружено ${data.terms_loaded} терминов из ${data.domains_loaded} доменов`
      })
      onDataLoaded()
    } catch (err) {
      setMessage({
        type: 'error',
        text: err instanceof Error ? err.message : 'Неизвестная ошибка'
      })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="data-upload-form">
      <div className="upload-header">
        <h3>📊 Управление данными</h3>
      </div>

      <div className="upload-content">
        <p className="upload-description">
          Загрузите демо-данные для демонстрации работы сервиса.
          Демо-данные включают 5 предметных областей с терминами.
        </p>

        <button
          className="load-demo-button"
          onClick={loadDemoData}
          disabled={loading}
        >
          {loading ? '⏳ Загрузка...' : '📥 Загрузить демо-данные'}
        </button>

        {message && (
          <div className={`message ${message.type}`}>
            {message.type === 'success' ? '✅' : '❌'} {message.text}
          </div>
        )}
      </div>

      <div className="data-info">
        <h4>📋 Формат данных:</h4>
        <div className="format-section">
          <div className="format-block">
            <strong>CSV:</strong>
            <code>term,domain,frequency</code>
            <code>машинное обучение,ML,150</code>
          </div>
          <div className="format-block">
            <strong>JSON:</strong>
            <code>{'{"domains": [{"name": "ML", "terms": [...]}]}'}</code>
          </div>
        </div>
        <div className="domains-list">
          <strong>Домены:</strong> ML, Biology, Mathematics, Physics, Chemistry
        </div>
      </div>
    </div>
  )
}
