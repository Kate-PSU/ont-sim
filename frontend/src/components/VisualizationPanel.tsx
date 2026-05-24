// frontend/src/components/VisualizationPanel.tsx
// Панель визуализации с несколькими вкладками
//
// Версия: 1.0
// Обновлено: 2026-04-08

import { useState } from 'react'
import * as d3 from 'd3'
import { API_BASE_URL } from '@/services/api'

// Типы данных
interface BenchmarkResult {
  method: string
  spearman: number
  pearson: number
  mse: number
  missing: number
  predictions_count: number
}

interface HypernymsResponse {
  term: string
  hypernyms: string[]
}

type TabType = 'benchmark' | 'hypernyms' | 'matrix'

export function VisualizationPanel() {
  const [activeTab, setActiveTab] = useState<TabType>('benchmark')
  const [benchmarkData, setBenchmarkData] = useState<BenchmarkResult[]>([])
  const [benchmarkLoading, setBenchmarkLoading] = useState(false)
  const [hypernymTerm, setHypernymTerm] = useState('')
  const [hypernyms, setHypernyms] = useState<HypernymsResponse | null>(null)
  const [hypernymsLoading, setHypernymsLoading] = useState(false)

  // Загрузка бенчмарка
  const loadBenchmark = async (dataset: string) => {
    setBenchmarkLoading(true)
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/benchmark/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset })
      })
      const data = await response.json()
      if (data.success && data.comparison) {
        setBenchmarkData(data.comparison.results)
      }
    } catch (error) {
      console.error('Ошибка загрузки бенчмарка:', error)
    } finally {
      setBenchmarkLoading(false)
    }
  }

  // Поиск гиперонимов
  const searchHypernyms = async () => {
    if (!hypernymTerm.trim()) return

    setHypernymsLoading(true)
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/wordnet/hypernyms/${encodeURIComponent(hypernymTerm)}`)
      const data = await response.json()
      setHypernyms(data)
    } catch (error) {
      console.error('Ошибка поиска гиперонимов:', error)
    } finally {
      setHypernymsLoading(false)
    }
  }

  // Рендер вкладки бенчмарка
  const renderBenchmarkTab = () => (
    <div className="benchmark-tab">
      <div className="benchmark-controls">
        <button
          className="run-button"
          onClick={() => loadBenchmark('hj')}
        >
          Запустить бенчмарк (HJ)
        </button>
      </div>

      {benchmarkLoading ? (
        <div className="benchmark-loading">Загрузка...</div>
      ) : benchmarkData.length > 0 ? (
        <div className="benchmark-results">
          <h3>Результаты методов</h3>

          {/* Bar Chart */}
          <div className="bar-chart">
            {benchmarkData.map((result) => (
              <div key={result.method} className="bar-row">
                <span className="bar-label">{result.method}</span>
                <div className="bar-track">
                  <div
                    className="bar-fill"
                    style={{
                      width: `${Math.abs(result.spearman) * 100}%`,
                      background: getBarColor(result.spearman)
                    }}
                  />
                </div>
                <span className="bar-value">{result.spearman.toFixed(3)}</span>
              </div>
            ))}
          </div>

          {/* Таблица */}
          <table className="results-table">
            <thead>
              <tr>
                <th>Метод</th>
                <th>Spearman</th>
                <th>Pearson</th>
                <th>MSE</th>
              </tr>
            </thead>
            <tbody>
              {benchmarkData.map((result) => (
                <tr key={result.method}>
                  <td>{result.method}</td>
                  <td className="metric-spearman">{result.spearman.toFixed(4)}</td>
                  <td>{result.pearson.toFixed(4)}</td>
                  <td>{result.mse.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="benchmark-empty">
          Нажмите кнопку для запуска бенчмарка
        </div>
      )}
    </div>
  )

  // Рендер вкладки гиперонимов
  const renderHypernymsTab = () => (
    <div className="hypernyms-tab">
      <div className="hypernyms-search">
        <input
          type="text"
          placeholder="Введите термин..."
          value={hypernymTerm}
          onChange={(e) => setHypernymTerm(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && searchHypernyms()}
        />
        <button className="run-button" onClick={searchHypernyms}>
          Найти
        </button>
      </div>

      {hypernymsLoading ? (
        <div className="benchmark-loading">Загрузка...</div>
      ) : hypernyms ? (
        <div className="hypernyms-result">
          <h3>Гиперонимы для "{hypernyms.term}"</h3>

          {/* Tree visualization */}
          <div className="tree-container">
            <svg ref={(svg) => {
              if (svg && hypernyms.hypernyms.length > 0) {
                renderTree(svg, hypernyms.term, hypernyms.hypernyms)
              }
            }} className="tree-svg" />
          </div>

          {/* Список */}
          <div className="hypernyms-list">
            {hypernyms.hypernyms.map((h, i) => (
              <div key={i} className="hypernym-item" style={{ marginLeft: `${i * 20}px` }}>
                {i === 0 ? '└─ ' : '└─ '}{h}
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="benchmark-empty">
          Введите термин для поиска гиперонимов
        </div>
      )}
    </div>
  )

  return (
    <div className="visualization-panel">
      <h2>Панель визуализации</h2>

      {/* Tabs */}
      <div className="panel-tabs">
        <button
          className={activeTab === 'benchmark' ? 'active' : ''}
          onClick={() => setActiveTab('benchmark')}
        >
          Бенчмарк
        </button>
        <button
          className={activeTab === 'hypernyms' ? 'active' : ''}
          onClick={() => setActiveTab('hypernyms')}
        >
          Гиперонимы
        </button>
      </div>

      {/* Tab Content */}
      <div className="panel-content">
        {activeTab === 'benchmark' && renderBenchmarkTab()}
        {activeTab === 'hypernyms' && renderHypernymsTab()}
      </div>
    </div>
  )
}

// Вспомогательные функции
function getBarColor(value: number): string {
  if (value >= 0.5) return '#10B981' // green
  if (value >= 0.3) return '#3B82F6' // blue
  if (value >= 0) return '#F59E0B'   // yellow
  return '#EF4444'                   // red
}

// Рендер дерева D3
function renderTree(svg: SVGSVGElement, term: string, hypernyms: string[]) {
  const width = 600
  const height = Math.max(300, hypernyms.length * 40)

  d3.select(svg).selectAll('*').remove()

  const svgSelection = d3.select(svg)
    .attr('width', width)
    .attr('height', height)

  const g = svgSelection.append('g')
    .attr('transform', `translate(50, 30)`)

  // Узел термина
  g.append('circle')
    .attr('cx', 0)
    .attr('cy', 0)
    .attr('r', 20)
    .attr('fill', '#3B82F6')

  g.append('text')
    .attr('x', 0)
    .attr('y', 5)
    .attr('text-anchor', 'middle')
    .attr('fill', 'white')
    .attr('font-size', '10px')
    .text(term.substring(0, 10))

  // Линии к гипернимам
  hypernyms.forEach((h, i) => {
    const y = (i + 1) * 40

    g.append('line')
      .attr('x1', 0)
      .attr('y1', 20)
      .attr('x2', 0)
      .attr('y2', y - 15)
      .attr('stroke', '#6b7280')
      .attr('stroke-width', 2)

    // Узел гипернима
    g.append('circle')
      .attr('cx', 0)
      .attr('cy', y)
      .attr('r', 15)
      .attr('fill', '#f59e0b')

    g.append('text')
      .attr('x', 25)
      .attr('y', y + 4)
      .attr('fill', '#fff')
      .attr('font-size', '12px')
      .text(h.length > 20 ? h.substring(0, 20) + '...' : h)
  })
}
