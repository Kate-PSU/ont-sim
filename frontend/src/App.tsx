// frontend/src/App.tsx
// Главный компонент приложения - версия 10.0
//
// Версия: 10.0
// Обновлено: 2026-04-20
// Особенности: Двухколоночная раскладка с InfoPanel

import { useState, useCallback, useRef, useEffect } from 'react'
import { useServiceState, getBestByAccuracy } from './hooks/useServiceState'
import { GraphVisualization } from './components/GraphVisualization'
import { Heatmap } from './components/Heatmap'
import { InfoPanel } from './components/InfoPanel'
import { WikipediaSimilarityResponse, getWikipediaSimilarity, getEnsembleSimilarity, EnsembleResult, getDomains, API_BASE_URL } from './services/api'

const METHOD_NAMES: Record<string, string> = {
  'sbert': 'SBERT',
  'sbert_tfidf': 'SBERT+TF-IDF',
  'sbert_zscore': 'SBERT+Z-score',
  'wordnet_lin': 'RuWordNet Lin',
  'wordnet_wup': 'RuWordNet Wu-Palmer',
  'wordnet_lin_en': 'EnWordNet Lin',
  'wordnet_wup_en': 'EnWordNet Wu-Palmer',
  'hybrid': 'Hybrid',
  'bertopic': 'BERTopic',
  'doc2vec': 'Doc2Vec',
  'lda': 'LDA',
}

// Методы расчёта графа
type GraphMethod = 'sbert' | 'rag' | 'tfidf' | 'ensemble'

// Вкладки главного вида
type MainTab = 'graph' | 'wikipedia' | 'ensemble'

// Режимы для InfoPanel
type InfoMode = 'graph' | 'matrix' | 'domains'

const MAIN_TABS: { id: MainTab; label: string; icon: string }[] = [
  { id: 'graph', label: 'Граф', icon: '📊' },
  { id: 'wikipedia', label: 'Wikipedia', icon: '🌐' },
  { id: 'ensemble', label: 'Ensemble', icon: '⚖️' },
]

const GRAPH_METHODS: { id: GraphMethod; label: string; icon: string }[] = [
  { id: 'sbert', label: 'SBERT', icon: '🎯' },
  { id: 'rag', label: 'RAG', icon: '🔍' },
  { id: 'tfidf', label: 'TF-IDF', icon: '📊' },
  { id: 'ensemble', label: 'Ensemble', icon: '⚖️' },
]

/**
 * Обновляет GET-параметры в URL без перезагрузки страницы.
 */
function updateUrlParams(params: Record<string, string | number>) {
  const url = new URL(window.location.href)
  Object.entries(params).forEach(([key, value]) => {
    url.searchParams.set(key, String(value))
  })
  window.history.pushState({}, '', url.toString())
}

/**
 * Читает параметры из URL при загрузке.
 */
function getInitialParams(): { threshold: number; showTerms: boolean; method: GraphMethod; tab: MainTab } {
  const url = new URL(window.location.href)
  return {
    threshold: parseFloat(url.searchParams.get('threshold') || '0.5'),
    showTerms: url.searchParams.get('show_terms') !== 'false',
    method: (url.searchParams.get('method') as GraphMethod) || 'sbert',
    tab: (url.searchParams.get('tab') as MainTab) || 'graph',
  }
}

function App() {
  const { state, loading, error, subscribed, refresh } = useServiceState()

  // Читаем начальные параметры из URL
  const initialParams = getInitialParams()

  // Вкладка главного вида
  const [activeTab, setActiveTab] = useState<MainTab>(initialParams.tab)

  // Режим для InfoPanel
  const [activeMode, setActiveMode] = useState<InfoMode>('graph')

  // Состояние для графа
  const [threshold, setThreshold] = useState(initialParams.threshold)
  const [showTerms, setShowTerms] = useState(initialParams.showTerms)
  const [graphData, setGraphData] = useState<{ nodes: any[]; edges: any[] }>({ nodes: [], edges: [] })
  const [graphLoading, setGraphLoading] = useState(false)

  // Метод расчёта графа (SBERT, RAG, WordNet, BERT)
  const [graphMethod, setGraphMethod] = useState<GraphMethod>(initialParams.method)

  // Ref для ID текущего запроса (защита от дублирования)
  const currentRequestIdRef = useRef<string | null>(null)

  // RAG статус
  const [ragStatus, setRagStatus] = useState<{ built: boolean; message?: string }>({ built: false })

  // Wikipedia данные
  const [wikipediaData, setWikipediaData] = useState<WikipediaSimilarityResponse | null>(null)
  const [wikipediaLoading, setWikipediaLoading] = useState(false)
  const [wikipediaError, setWikipediaError] = useState<string | null>(null)

  // Ensemble данные
  const [ensembleWeights, setEnsembleWeights] = useState({ sbert: 0.7, tfidf: 0.3 })
  const [ensembleDomain1, setEnsembleDomain1] = useState('')
  const [ensembleDomain2, setEnsembleDomain2] = useState('')
  const [ensembleResult, setEnsembleResult] = useState<EnsembleResult | null>(null)
  const [ensembleLoading, setEnsembleLoading] = useState(false)
  const [ensembleError, setEnsembleError] = useState<string | null>(null)
  const [availableDomains, setAvailableDomains] = useState<string[]>([])
  const [domainsLoading, setDomainsLoading] = useState(false)

  // Проверка статуса RAG индексов
  const checkRagStatus = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/rag/status`)
      const data = await response.json()
      setRagStatus({ built: data.built, message: data.message })
    } catch (err) {
      console.error('Ошибка проверки RAG статуса:', err)
      setRagStatus({ built: false, message: 'Ошибка проверки статуса' })
    }
  }, [])

  // Загрузка Wikipedia similarity данных
  const loadWikipediaData = useCallback(async () => {
    setWikipediaLoading(true)
    setWikipediaError(null)
    try {
      const data = await getWikipediaSimilarity()
      setWikipediaData(data)
    } catch (err) {
      console.error('Ошибка загрузки Wikipedia данных:', err)
      setWikipediaError(err instanceof Error ? err.message : 'Неизвестная ошибка')
    } finally {
      setWikipediaLoading(false)
    }
  }, [])

  // Расчёт ensemble similarity
  const calculateEnsembleSimilarity = useCallback(async () => {
    if (!ensembleDomain1 || !ensembleDomain2) {
      setEnsembleError('Выберите оба домена')
      return
    }

    setEnsembleLoading(true)
    setEnsembleError(null)
    try {
      const result = await getEnsembleSimilarity(ensembleDomain1, ensembleDomain2, ensembleWeights)
      setEnsembleResult(result)
    } catch (err) {
      console.error('Ошибка расчёта ensemble similarity:', err)
      setEnsembleError(err instanceof Error ? err.message : 'Неизвестная ошибка')
    } finally {
      setEnsembleLoading(false)
    }
  }, [ensembleDomain1, ensembleDomain2, ensembleWeights])

  // Автоматический пересчёт при изменении весов
  useEffect(() => {
    if (ensembleDomain1 && ensembleDomain2 && ensembleResult) {
      calculateEnsembleSimilarity()
    }
  }, [ensembleWeights])

  // Проверяем RAG статус при загрузке
  useEffect(() => {
    checkRagStatus()
  }, [checkRagStatus])

  // Загрузка списка доменов для dropdown
  useEffect(() => {
    const loadDomains = async () => {
      setDomainsLoading(true)
      try {
        const domains = await getDomains()
        setAvailableDomains(domains)
      } catch (err) {
        console.error('Ошибка загрузки доменов:', err)
      } finally {
        setDomainsLoading(false)
      }
    }
    loadDomains()
  }, [])

  // Загружаем Wikipedia данные при переключении на вкладку
  useEffect(() => {
    if (activeTab === 'wikipedia' && !wikipediaData && !wikipediaLoading) {
      loadWikipediaData()
    }
  }, [activeTab, wikipediaData, wikipediaLoading, loadWikipediaData])

  // Обновляем mode для InfoPanel при смене вкладки
  useEffect(() => {
    if (activeTab === 'graph') {
      setActiveMode('graph')
    } else if (activeTab === 'wikipedia') {
      setActiveMode('matrix')
    } else if (activeTab === 'ensemble') {
      setActiveMode('domains')
    }
  }, [activeTab])

  // Обработчик изменения порога (без автозагрузки)
  const handleThresholdChange = useCallback((newThreshold: number) => {
    setThreshold(newThreshold)
    updateUrlParams({ threshold: newThreshold })
  }, [])

  // Обработчик изменения метода
  const handleMethodChange = useCallback((newMethod: GraphMethod) => {
    setGraphMethod(newMethod)
    updateUrlParams({ method: newMethod })
  }, [])

  // Обработчик изменения showTerms
  const handleShowTermsChange = useCallback((newShowTerms: boolean) => {
    setShowTerms(newShowTerms)
    updateUrlParams({ show_terms: `${newShowTerms}` })
  }, [])

  // Обработчик изменения вкладки
  const handleTabChange = useCallback((newTab: MainTab) => {
    setActiveTab(newTab)
    updateUrlParams({ tab: newTab })
  }, [])

  // Загрузка графа с выбранным методом
  const loadGraph = useCallback(async () => {
    // Генерируем уникальный ID запроса
    const requestId = `${threshold}-${showTerms}-${graphMethod}-${Date.now()}`

    // Проверяем, не выполняется ли уже такой запрос
    if (currentRequestIdRef.current === requestId) {
      console.log('Запрос уже выполняется, пропускаем')
      return
    }
    currentRequestIdRef.current = requestId

    setGraphLoading(true)
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/v1/graph/detailed?threshold=${threshold}&show_terms=${showTerms}&method=${graphMethod}`
      )
      const data = await response.json()

      // Проверяем, что это всё ещё наш запрос
      if (currentRequestIdRef.current === requestId) {
        setGraphData(data)
      }
    } catch (err) {
      console.error('Ошибка загрузки графа:', err)
    } finally {
      // Сбрасываем флаг только если это наш запрос
      if (currentRequestIdRef.current === requestId) {
        setGraphLoading(false)
        currentRequestIdRef.current = null
      }
    }
  }, [threshold, showTerms, graphMethod])

  // Лучшие методы
  const bestByAccuracy = state?.datasets ? getBestByAccuracy(state.datasets) : []

  // Рендер строки состояния
  const renderStatusBar = () => {
    if (loading) {
      return (
        <div className="status-bar status-loading">
          <span className="status-icon">⏳</span>
          <span>Загрузка состояния системы...</span>
        </div>
      )
    }

    if (error) {
      return (
        <div className="status-bar status-error">
          <span className="status-icon">❌</span>
          <span>Ошибка: {error}</span>
          <button onClick={refresh} className="retry-btn">Повторить</button>
        </div>
      )
    }

    if (!state) return null

    const statusIcon = state.system_status === 'ready' ? '🟢' :
      state.system_status === 'busy' ? '🟡' : '🔴'
    const statusText = state.system_status === 'ready' ? 'Система готова' :
      state.system_status === 'busy' ? `Занята: ${state.busy_reason}` :
        'Ошибка системы'

    return (
      <div className={`status-bar status-${state.system_status}`}>
        <span className="status-icon">{statusIcon}</span>
        <span className="status-text">{statusText}</span>
        <span className="status-sse">
          SSE: {subscribed ? '●' : '○'}
        </span>
        <span className="status-domains">
          {state.domains_count > 0 ? `${state.domains_count} доменов` : 'Нет данных'}
        </span>
        <button onClick={refresh} className="refresh-btn">🔄</button>
      </div>
    )
  }

  // Рендер секции графа
  const renderGraphSection = () => (
    <section className="section graph-section">

      {/* Вкладки выбора метода */}
      <div className="method-tabs" style={{ display: "flex", flexDirection: "column", minHeight: '85vh' }}>
        <div style={{ display: "flex" }}>
          <span className="method-tabs-label">Метод расчёта:</span>
          {GRAPH_METHODS.map(m => {
            // RAG кнопка disabled если индексы не построены ИЛИ идёт загрузка
            const isRagDisabled = m.id === 'rag' && !ragStatus.built
            const disabled = graphLoading || isRagDisabled

            return (
              <button
                key={m.id}
                data-method={m.id}
                onClick={() => handleMethodChange(m.id)}
                className={`method-tab ${graphMethod === m.id ? 'active' : ''}`}
                title={isRagDisabled && ragStatus.message ? ragStatus.message : `Метод: ${m.label}`}
                disabled={disabled}
                style={isRagDisabled ? { opacity: 0.5, cursor: 'not-allowed' } : {}}
              >
                {m.icon} {m.label}
                {isRagDisabled && ' ⚠️'}
              </button>
            )
          })}

          <div className="threshold-control">
            <label>
              Порог когнитивной близости:
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={threshold}
                onChange={(e) => handleThresholdChange(parseFloat(e.target.value))}
                disabled={graphLoading}
              />
              <span>{threshold.toFixed(2)}</span>
            </label>
          </div>

          <label className="toggle-terms">
            <input
              type="checkbox"
              checked={showTerms}
              onChange={(e) => handleShowTermsChange(e.target.checked)}
              disabled={graphLoading}
            />
            Показать термины
          </label>

          <button onClick={loadGraph} disabled={graphLoading || !state?.domains_loaded}>
            {graphLoading ? '⏳ Загрузка...' : '🔄 Показать граф'}
          </button>
        </div>

        {graphLoading ? (
          <div className="loading-placeholder">Загрузка графа...</div>
        ) : (
          <GraphVisualization
            nodes={graphData.nodes}
            edges={graphData.edges}
            showTerms={showTerms}
            currentMethod={graphMethod}
            graphLoading={graphLoading}
          />
        )}
      </div>
    </section>
  )

  // Рендер секции Wikipedia similarity
  const renderWikipediaSection = () => (
    <section className="section wikipedia-section">
      <h2>🌐 Wikipedia Domain Similarity</h2>

      <div className="wikipedia-controls">
        <button
          onClick={loadWikipediaData}
          disabled={wikipediaLoading}
          className="refresh-wikipedia-btn"
        >
          {wikipediaLoading ? '⏳ Загрузка...' : '🔄 Обновить данные'}
        </button>
      </div>

      {wikipediaError && (
        <div className="error-message wikipedia-error">
          ❌ {wikipediaError}
        </div>
      )}

      {wikipediaLoading && !wikipediaData && (
        <div className="loading-placeholder">Загрузка данных Wikipedia similarity...</div>
      )}

      {wikipediaData && (
        <div className="wikipedia-viz">
          <div className="heatmap-wrapper">
            <h3>Матрица близости доменов</h3>
            <Heatmap
              data={wikipediaData.matrix}
              labels={wikipediaData.domains}
            />
          </div>

          <div className="top-pairs">
            <h3>🔥 Top 5 пар доменов</h3>
            <div className="top-pairs-list">
              {wikipediaData.top_pairs.map((pair, index) => (
                <div key={`${pair.d1}-${pair.d2}`} className="top-pair-item">
                  <span className="pair-rank">#{index + 1}</span>
                  <span className="pair-domains">
                    {pair.d1} ↔ {pair.d2}
                  </span>
                  <span className="pair-score">
                    {pair.score.toFixed(4)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </section>
  )

  // Рендер секции Ensemble similarity
  const renderEnsembleSection = () => (
    <section className="section ensemble-section">
      <h2>⚖️ Ensemble Similarity</h2>

      <div className="ensemble-controls">
        <div className="domain-inputs">
          <div className="domain-input-group">
            <label htmlFor="domain1">Домен 1:</label>
            <select
              id="domain1"
              value={ensembleDomain1}
              onChange={(e) => setEnsembleDomain1(e.target.value)}
              disabled={ensembleLoading || domainsLoading}
            >
              <option value="">{domainsLoading ? 'Загрузка...' : 'Выберите домен'}</option>
              {availableDomains.map(d => (
                <option key={d} value={d}>{d}</option>
              ))}
            </select>
          </div>
          <span className="vs-label">vs</span>
          <div className="domain-input-group">
            <label htmlFor="domain2">Домен 2:</label>
            <select
              id="domain2"
              value={ensembleDomain2}
              onChange={(e) => setEnsembleDomain2(e.target.value)}
              disabled={ensembleLoading || domainsLoading}
            >
              <option value="">{domainsLoading ? 'Загрузка...' : 'Выберите домен'}</option>
              {availableDomains.map(d => (
                <option key={d} value={d}>{d}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="weights-controls">
          <h3>Настройка весов</h3>

          <div className="weight-slider">
            <label htmlFor="sbert-weight">
              SBERT вес: <span className="weight-value">{ensembleWeights.sbert.toFixed(2)}</span>
            </label>
            <input
              id="sbert-weight"
              type="range"
              min="0.5"
              max="1.0"
              step="0.05"
              value={ensembleWeights.sbert}
              onChange={(e) => {
                const newSbert = parseFloat(e.target.value)
                setEnsembleWeights(_ => ({
                  sbert: newSbert,
                  tfidf: parseFloat((1 - newSbert + 0.5).toFixed(2))
                }))
              }}
              disabled={ensembleLoading}
            />
            <div className="weight-range-labels">
              <span>0.5</span>
              <span>1.0</span>
            </div>
          </div>

          <div className="weight-slider">
            <label htmlFor="tfidf-weight">
              TF-IDF вес: <span className="weight-value">{ensembleWeights.tfidf.toFixed(2)}</span>
            </label>
            <input
              id="tfidf-weight"
              type="range"
              min="0.0"
              max="0.5"
              step="0.05"
              value={ensembleWeights.tfidf}
              onChange={(e) => {
                const newTfidf = parseFloat(e.target.value)
                setEnsembleWeights(_ => ({
                  tfidf: newTfidf,
                  sbert: parseFloat((1 - newTfidf + 0.5).toFixed(2))
                }))
              }}
              disabled={ensembleLoading}
            />
            <div className="weight-range-labels">
              <span>0.0</span>
              <span>0.5</span>
            </div>
          </div>

          <div className="weights-summary">
            <span className="summary-label">Сумма весов:</span>
            <span className="summary-value">{(ensembleWeights.sbert + ensembleWeights.tfidf).toFixed(2)}</span>
          </div>
        </div>

        <button
          onClick={calculateEnsembleSimilarity}
          disabled={ensembleLoading || !ensembleDomain1 || !ensembleDomain2}
          className="calculate-ensemble-btn"
        >
          {ensembleLoading ? '⏳ Расчёт...' : '🔄 Рассчитать'}
        </button>
      </div>

      {ensembleError && (
        <div className="error-message ensemble-error">
          ❌ {ensembleError}
        </div>
      )}

      {ensembleResult && (
        <div className="ensemble-results">
          <h3>📊 Результаты</h3>

          <div className="result-card main-similarity">
            <div className="result-label">Ensemble Similarity</div>
            <div className="result-value">{ensembleResult.similarity.toFixed(4)}</div>
          </div>

          <div className="result-cards">
            <div className="result-card sbert-score">
              <div className="result-label">SBERT Score</div>
              <div className="result-value">{ensembleResult.sbert_score.toFixed(4)}</div>
            </div>

            <div className="result-card tfidf-score">
              <div className="result-label">TF-IDF Score</div>
              <div className="result-value">{ensembleResult.tfidf_score.toFixed(4)}</div>
            </div>
          </div>

          <div className="result-card weights-info">
            <div className="result-label">Используемые веса</div>
            <div className="weights-display">
              <span className="sbert-weight">SBERT: {ensembleResult.weights.sbert.toFixed(2)}</span>
              <span className="tfidf-weight">TF-IDF: {ensembleResult.weights.tfidf.toFixed(2)}</span>
            </div>
          </div>
        </div>
      )}
    </section>
  )

  // Рендер топ-3
  const renderTopSection = () => {
    if (bestByAccuracy.length === 0) return null

    return (
      <section className="section top-section">
        <h2>🏆 Лучшие методы</h2>

        <div className="top-grid">
          <div className="top-card">
            <h3>По точности (Spearman)</h3>
            <ol>
              {bestByAccuracy.slice(0, 3).map((item, i) => (
                <li key={item.method}>
                  {i === 0 ? '🥇' : i === 1 ? '🥈' : '🥉'}
                  {METHOD_NAMES[item.method] || item.method}
                  <span className="score">{(item.avgSpearman * 100).toFixed(1)}%</span>
                  <button
                    className="select-btn"
                    onClick={() => {
                      handleMethodChange(item.method as GraphMethod)
                    }}
                    disabled={graphLoading}
                  >
                    Выбрать
                  </button>
                </li>
              ))}
            </ol>
          </div>

          <div className="top-card">
            <h3>По скорости</h3>
            <p className="coming-soon">В разработке...</p>
          </div>
        </div>
      </section>
    )
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>🔬 Онтологический граф-сервис oценки когнитивной близости предметных областей</h1>
        {renderStatusBar()}
      </header>

      {/* Вкладки главного вида */}
      <nav className="main-tabs">
        {MAIN_TABS.map(tab => (
          <button
            key={tab.id}
            className={`main-tab ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => handleTabChange(tab.id)}
          >
            {tab.icon} {tab.label}
          </button>
        ))}
      </nav>

      <main className="app-main" style={{ display: 'flex', flexDirection: 'row' }}>
        <div className="app-content-wrapper">
          {activeTab === 'graph' && renderGraphSection()}
          {activeTab === 'wikipedia' && renderWikipediaSection()}
          {activeTab === 'ensemble' && renderEnsembleSection()}
          {activeTab === 'graph' && renderTopSection()}
        </div>
        <InfoPanel mode={activeMode} />
      </main>
    </div>
  )
}

export default App
