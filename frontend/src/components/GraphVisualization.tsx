// frontend/src/components/GraphVisualization.tsx
// Компонент визуализации графа (D3.js)
//
// Версия: 7.0
// Обновлено: 2026-04-19
// Особенности: рамка вокруг контейнера D3, placeholder с border

import { useEffect, useRef, useState, useCallback } from 'react'
import * as d3 from 'd3'

// Типы узлов
interface Node extends d3.SimulationNodeDatum {
  id: string
  label: string
  type: 'domain' | 'term'
  parent?: string
}

interface Edge {
  source: string | Node
  target: string | Node
  weight: number
  type: 'similarity' | 'belongs_to'
}

interface GraphVisualizationProps {
  nodes: Node[]
  edges: Edge[]
  showTerms?: boolean
  onMethodChange?: (method: string) => void
  currentMethod?: string
  graphLoading?: boolean
}

// Параметры визуализации (адаптивные)
const CHARGE_STRENGTH = -400
const LINK_DISTANCE_BASE = 120

// Базовые размеры (используются по умолчанию)
const BASE_WIDTH = 900
const BASE_HEIGHT = 650

// Размеры узлов
const DOMAIN_RADIUS = 25
const TERM_RADIUS = 8

// Базовые цвета для доменов (彩虹 палитра)
const DOMAIN_COLORS = [
  '#e53935',  // Красный
  '#1e88e5',  // Синий
  '#43a047',  // Зелёный
  '#fb8c00',  // Оранжевый
  '#8e24aa',  // Фиолетовый
  '#00acc1',  // Бирюзовый
  '#fdd835',  // Жёлтый
  '#6d4c41',  // Коричневый
  '#546e7a',  // Серо-синий
  '#d81b60',  // Розовый
  '#00897b',  // Тёмно-бирюзовый
  '#5e35b1',  // Тёмно-фиолетовый
]

// Цвета связей
const SIMILARITY_COLOR = '#6b7280'  // Серый для связей близости
const BELONGS_COLOR = '#9ca3af'     // Более контрастный серый для принадлежности

/**
 * Генерирует палитру цветов для заданного количества доменов.
 * Использует HSL для равномерного распределения оттенков.
 */
function generateDomainPalette(domainCount: number): string[] {
  if (domainCount <= DOMAIN_COLORS.length) {
    return DOMAIN_COLORS.slice(0, domainCount)
  }
  
  // Если доменов больше чем базовых цветов — генерируем через HSL
  const colors: string[] = []
  for (let i = 0; i < domainCount; i++) {
    const hue = (i * 360) / domainCount
    const saturation = 65 + (i % 2) * 10  // 65-75%
    const lightness = 45 + (i % 3) * 5    // 45-55%
    colors.push(`hsl(${hue}, ${saturation}%, ${lightness}%)`)
  }
  return colors
}

/**
 * Осветляет цвет для терминов (наследуют цвет домена).
 * Использует d3 для работы с цветовым пространством.
 */
function lightenColor(color: string, amount: number = 0.4): string {
  const parsed = d3.color(color)
  if (!parsed) return color
  
  const hsl = d3.hsl(parsed)
  hsl.l = Math.min(hsl.l + amount, 0.95)
  hsl.s = Math.max(hsl.s - amount * 0.5, 0.3)
  
  return hsl.formatHex()
}

/**
 * Затемняет цвет для обводки.
 */
function darkenColor(color: string, amount: number = 0.2): string {
  const parsed = d3.color(color)
  if (!parsed) return color
  
  const hsl = d3.hsl(parsed)
  hsl.l = Math.max(hsl.l - amount, 0.2)
  
  return hsl.formatHex()
}

export function GraphVisualization({ 
  nodes, 
  edges, 
  showTerms = true,
}: GraphVisualizationProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null)
  const [hoveredNode, setHoveredNode] = useState<Node | null>(null)
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 })
  const [dimensions, setDimensions] = useState({ width: BASE_WIDTH, height: BASE_HEIGHT })
  
  // Функция для получения размеров контейнера
  const updateDimensions = useCallback(() => {
    if (containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect()
      setDimensions({
        width: rect.width > 0 ? rect.width : BASE_WIDTH,
        height: rect.height > 0 ? rect.height : BASE_HEIGHT
      })
    }
  }, [])

  // Следим за изменениями размеров окна и контейнера
  useEffect(() => {
    updateDimensions()
    window.addEventListener('resize', updateDimensions)
    return () => window.removeEventListener('resize', updateDimensions)
  }, [updateDimensions])

  // Функция сброса зума
  const resetZoom = () => {
    if (svgRef.current && zoomRef.current) {
      d3.select(svgRef.current)
        .transition()
        .duration(300)
        .call(zoomRef.current.transform, d3.zoomIdentity)
    }
  }

  useEffect(() => {
    if (!svgRef.current || !nodes || nodes.length === 0) return

    // Используем актуальные размеры
    const WIDTH = dimensions.width
    const HEIGHT = dimensions.height

    // Очистка предыдущего графа
    d3.select(svgRef.current).selectAll('*').remove()

    // Создание SVG
    const svg = d3
      .select(svgRef.current)
      .attr('width', '100%')
      .attr('height', '100%')
      .attr('viewBox', `0 0 ${WIDTH} ${HEIGHT}`)

    // Определение маркеров (стрелок)
    const defs = svg.append('defs')
    
    // Стрелка для belongs_to связей
    defs.append('marker')
      .attr('id', 'arrowhead')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 8)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', BELONGS_COLOR)

    // Создание группы для графа
    const g = svg.append('g')

    // Зум и панорамирование
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.05, 4])  // Минимум 0.05 позволяет сильнее уменьшать
      .on('zoom', (event) => {
        g.attr('transform', event.transform)
      })
    
    // Сохраняем ссылку на zoom для кнопки Reset
    zoomRef.current = zoom

    svg.call(zoom)

    // Подготовка данных
    const nodeData = nodes.map((n) => ({ ...n }))
    const linkData = edges.map((e) => ({ ...e }))

    // Разделяем узлы по типу
    const domainNodes = nodeData.filter((n) => n.type === 'domain')
    const termNodes = nodeData.filter((n) => n.type === 'term')
    
    // Генерируем палитру для доменов
    const domainColors = generateDomainPalette(domainNodes.length)
    
    // Создаём маппинг: domain id -> color
    const domainColorMap = new Map<string, string>()
    domainNodes.forEach((domain, index) => {
      domainColorMap.set(domain.id, domainColors[index])
    })
    
    // Все ноды для симуляции
    const allNodes = [...domainNodes, ...termNodes]

    // Симуляция силы
    const validLinks: typeof linkData = []
    for (const edge of linkData) {
      const sourceId = typeof edge.source === 'string' ? edge.source : edge.source.id
      const targetId = typeof edge.target === 'string' ? edge.target : edge.target.id
      const sourceNode = allNodes.find(n => n.id === sourceId)
      const targetNode = allNodes.find(n => n.id === targetId)
      if (sourceNode && targetNode) {
        validLinks.push(edge)
      }
    }
    
    const simulation = d3
      .forceSimulation(allNodes)
      .force(
        'link',
        d3
          .forceLink(validLinks)
          .id((d) => (d as Node).id)
          .distance((d) => {
            const edge = d as Edge
            return edge.type === 'belongs_to' ? LINK_DISTANCE_BASE * 0.6 : LINK_DISTANCE_BASE
          }),
      )
      .force('charge', d3.forceManyBody().strength(CHARGE_STRENGTH))
      .force('center', d3.forceCenter(WIDTH / 2, HEIGHT / 2))
      .force('collision', d3.forceCollide().radius(30))

    // Отрисовка связей принадлежности (только если showTerms=true)
    const belongsLinksGroup = g
      .append('g')
      .attr('class', 'belongs-links')
      .style('display', showTerms ? 'block' : 'none')
    
    const belongsLinks = belongsLinksGroup
      .selectAll('line')
      .data(linkData.filter((e) => e.type === 'belongs_to'))
      .join('line')
      .attr('stroke', BELONGS_COLOR)
      .attr('stroke-opacity', 0.8)
      .attr('stroke-width', 3)
      .attr('stroke-dasharray', '6,3')
      .attr('marker-end', 'url(#arrowhead)')

    // Отрисовка связей близости (толстые сплошные с цветом по степени близости)
    const similarityLinks = g
      .append('g')
      .attr('class', 'similarity-links')
      .selectAll('line')
      .data(linkData.filter((e) => e.type === 'similarity'))
      .join('line')
      .attr('stroke', SIMILARITY_COLOR)
      .attr('stroke-opacity', (d) => 0.4 + (d as Edge).weight * 0.6)
      // Усиленная зависимость толщины от веса: от 1px до 8px
      .attr('stroke-width', (d) => 1 + (d as Edge).weight * 7)

    // Отрисовка узлов доменов с индивидуальными цветами
    const domainGroups = g
      .append('g')
      .attr('class', 'domain-nodes')
      .selectAll<SVGGElement, Node>('g')
      .data(domainNodes)
      .join('g')
      .attr('class', 'domain-node')
      .call(
        d3
          .drag<SVGGElement, Node>()
          .on('start', (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart()
            d.fx = d.x
            d.fy = d.y
          })
          .on('drag', (event, d) => {
            d.fx = event.x
            d.fy = event.y
          })
          .on('end', (event, d) => {
            if (!event.active) simulation.alphaTarget(0)
            d.fx = null
            d.fy = null
          }),
      )

    // Круг для домена с индивидуальным цветом
    domainGroups
      .append('circle')
      .attr('r', DOMAIN_RADIUS)
      .attr('fill', (d) => domainColorMap.get(d.id) || DOMAIN_COLORS[0])
      .attr('stroke', (d) => darkenColor(domainColorMap.get(d.id) || DOMAIN_COLORS[0]))
      .attr('stroke-width', 2)

    // Подсветка при наведении на домен
    domainGroups
      .style('cursor', 'pointer')
      .on('mouseenter', (event, d) => {
        setHoveredNode(d)
        setTooltipPos({ x: event.pageX, y: event.pageY })
        highlightGroup(d.id, true)
        
        // Увеличиваем круг домена
        d3.select(event.currentTarget)
          .select('circle')
          .transition()
          .duration(150)
          .attr('r', DOMAIN_RADIUS * 1.3)
          .attr('stroke-width', 3)
      })
      .on('mouseleave', (event) => {
        setHoveredNode(null)
        highlightGroup(null, false)
        
        // Возвращаем обычный размер
        d3.select(event.currentTarget)
          .select('circle')
          .transition()
          .duration(150)
          .attr('r', DOMAIN_RADIUS)
          .attr('stroke-width', 2)
      })

    // Текстовая метка домена
    domainGroups
      .append('text')
      .text((d) => d.label)
      .attr('x', 0)
      .attr('y', DOMAIN_RADIUS + 15)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('font-weight', 'bold')
      .attr('fill', '#1f2937')

    // Отрисовка терминов с цветом, унаследованным от родительского домена
    if (showTerms) {
      const termGroups = g
        .append('g')
        .attr('class', 'term-nodes')
        .selectAll<SVGGElement, Node>('g')
        .data(termNodes)
        .join('g')
        .attr('class', 'term-node')

      // Круг для термина с осветлённым цветом родителя
      termGroups
        .append('circle')
        .attr('r', TERM_RADIUS)
        .attr('fill', (d) => {
          const parentColor = domainColorMap.get(d.parent || '')
          return parentColor ? lightenColor(parentColor, 0.35) : '#f59e0b'
        })
        .attr('stroke', (d) => {
          const parentColor = domainColorMap.get(d.parent || '')
          return parentColor ? darkenColor(parentColor, 0.15) : '#d97706'
        })
        .attr('stroke-width', 1)

      // Текстовая метка термина
      termGroups
        .append('text')
        .text((d) => d.label)
        .attr('x', TERM_RADIUS + 4)
        .attr('y', 4)
        .attr('font-size', '10px')
        .attr('fill', '#374151')

      // Интерактивность для терминов
      termGroups
        .style('cursor', 'pointer')
        .on('mouseenter', (event, d) => {
          setHoveredNode(d)
          setTooltipPos({ x: event.pageX, y: event.pageY })
          highlightGroup(d.parent || null, true)
          
          d3.select(event.currentTarget)
            .select('circle')
            .transition()
            .duration(150)
            .attr('r', TERM_RADIUS * 1.5)
        })
        .on('mouseleave', (event) => {
          setHoveredNode(null)
          highlightGroup(null, false)
          
          d3.select(event.currentTarget)
            .select('circle')
            .transition()
            .duration(150)
            .attr('r', TERM_RADIUS)
        })
    }

      // Функция для подсветки связанных элементов
      const highlightGroup = (domainId: string | null, highlight: boolean) => {
        if (!showTerms) return
        
        // Подсветка связей принадлежности
        belongsLinks
          .attr('stroke-opacity', (d) => {
            if (!highlight) return 0.8
            const edge = d as Edge
            const sourceId = typeof edge.source === 'string' ? edge.source : edge.source.id
            return sourceId.includes(domainId || '') ? 0.9 : 0.2
          })
          .attr('stroke-width', (d) => {
            if (!highlight) return 3
            const edge = d as Edge
            const sourceId = typeof edge.source === 'string' ? edge.source : edge.source.id
            return sourceId.includes(domainId || '') ? 5 : 2
          })
        
        // Подсветка терминов
        if (g.selectAll('.term-node').nodes().length > 0) {
          g.selectAll('.term-node circle')
            .attr('opacity', (d) => {
              if (!highlight) return 1
              const node = d as Node
              return node.parent === domainId ? 1 : 0.3
            })
          g.selectAll('.term-node text')
            .attr('opacity', (d) => {
              if (!highlight) return 1
              const node = d as Node
              return node.parent === domainId ? 1 : 0.3
            })
        }
      }

      // Отрисовка эллипсов групп (домен + термины) - только если showTerms=true
      if (showTerms) {
        g.append('g')
          .attr('class', 'group-ellipses')
          .selectAll('ellipse')
          .data(domainNodes)
          .join('ellipse')
          .attr('class', 'domain-group')
          .attr('fill', (d) => {
            const color = domainColorMap.get(d.id) || DOMAIN_COLORS[0]
            const parsed = d3.color(color)
            if (!parsed) return color
            const hsl = d3.hsl(parsed)
            hsl.l = Math.min(hsl.l + 0.35, 0.9)
            hsl.s = Math.max(hsl.s - 0.3, 0.2)
            return hsl.formatHex()
          })
          .attr('fill-opacity', 0.15)
          .attr('stroke', (d) => domainColorMap.get(d.id) || DOMAIN_COLORS[0])
          .attr('stroke-width', 2)
          .attr('stroke-opacity', 0.5)
          .attr('stroke-dasharray', '5,5')
      }

      // Обновление позиций при симуляции
      simulation.on('tick', () => {
        belongsLinks
          .attr('x1', (d) => ((d.source as Node).x || 0))
          .attr('y1', (d) => ((d.source as Node).y || 0))
          .attr('x2', (d) => ((d.target as Node).x || 0))
          .attr('y2', (d) => ((d.target as Node).y || 0))

        similarityLinks
          .attr('x1', (d) => ((d.source as Node).x || 0))
          .attr('y1', (d) => ((d.source as Node).y || 0))
          .attr('x2', (d) => ((d.target as Node).x || 0))
          .attr('y2', (d) => ((d.target as Node).y || 0))

        g.selectAll('.domain-node').attr(
          'transform',
          (d) => `translate(${((d as Node).x || 0)},${((d as Node).y || 0)})`,
        )

        // Проверяем существование term-node перед обновлением
        if (g.selectAll('.term-node').nodes().length > 0) {
          g.selectAll('.term-node').attr(
            'transform',
            (d) => `translate(${((d as Node).x || 0)},${((d as Node).y || 0)})`,
          )
        }
        
        // Обновление эллипсов групп
        if (showTerms) {
          domainNodes.forEach((domain) => {
            const domainTerms = termNodes.filter(t => t.parent === domain.id)
            const allGroupNodes = [domain, ...domainTerms].filter(n => n.x !== undefined && n.y !== undefined)
            
            if (allGroupNodes.length === 0) return
            
            const xPositions = allGroupNodes.map(n => n.x || 0)
            const yPositions = allGroupNodes.map(n => n.y || 0)
            
            const minX = Math.min(...xPositions)
            const maxX = Math.max(...xPositions)
            const minY = Math.min(...yPositions)
            const maxY = Math.max(...yPositions)
            
            const centerX = (minX + maxX) / 2
            const centerY = (minY + maxY) / 2
            const radiusX = Math.max((maxX - minX) / 2 + 25, 50)
            const radiusY = Math.max((maxY - minY) / 2 + 20, 40)
            
            g.selectAll(`.domain-group`)
              .filter((d) => (d as Node).id === domain.id)
              .attr('cx', centerX)
              .attr('cy', centerY)
              .attr('rx', radiusX)
              .attr('ry', radiusY)
          })
        }
      })

      // Центрирование графа после завершения симуляции
      simulation.on('end', () => {
        // Рассчитываем bounding box всех нодов
        const padding = 50 // Отступ от краёв
        const nodesWithPos = allNodes.filter(n => n.x !== undefined && n.y !== undefined)
        
        if (nodesWithPos.length === 0) return
        
        const xPositions = nodesWithPos.map(n => n.x || 0)
        const yPositions = nodesWithPos.map(n => n.y || 0)
        
        const minX = Math.min(...xPositions)
        const maxX = Math.max(...xPositions)
        const minY = Math.min(...yPositions)
        const maxY = Math.max(...yPositions)
        
        // Центр графа
        const graphCenterX = (minX + maxX) / 2
        const graphCenterY = (minY + maxY) / 2
        
        // Размеры графа
        const graphWidth = maxX - minX + DOMAIN_RADIUS * 2 + padding * 2
        const graphHeight = maxY - minY + DOMAIN_RADIUS * 2 + padding * 2
        
        // Коэффициент для подгонки под viewport
        const scaleX = WIDTH / graphWidth
        const scaleY = HEIGHT / graphHeight
        const scale = Math.min(Math.max(Math.min(scaleX, scaleY), 0.3), 1.5)
        
        // Применяем transform для центрирования
        const translateX = WIDTH / 2 - graphCenterX * scale
        const translateY = HEIGHT / 2 - graphCenterY * scale
        
        // Анимированный переход к центру
        svg.transition()
          .duration(500)
          .call(
            zoom.transform,
            d3.zoomIdentity.translate(translateX, translateY).scale(scale)
          )
      })

      return () => {
        simulation.stop()
      }
  }, [nodes, edges, showTerms, dimensions])

  // Статистика
  const domainCount = nodes.filter((n) => n.type === 'domain').length
  const termCount = nodes.filter((n) => n.type === 'term').length
  const similarityCount = edges.filter((e) => e.type === 'similarity').length

  // Проверяем, есть ли данные для отображения
  const hasData = nodes.length > 0

  return (
    <div className="graph-visualization" ref={containerRef} style={{height: '85vh'}}>
      {/* Статистика */}
      {hasData && (
        <div className="graph-stats">
          <span className="stat domain">
            Домены: {domainCount}
          </span>
          <span className="stat term">
            Термины: {termCount}
          </span>
          <span className="stat edge">
            Когнитивные связи: {similarityCount}
          </span>
        </div>
      )}

      {/* SVG граф с рамкой */}
      <div className="graph-container-wrapper">
        <svg ref={svgRef} className="graph-svg" />
      </div>
      
      {/* Кнопка сброса зума (только когда есть данные) */}
      {hasData && (
        <button className="reset-zoom-btn" onClick={resetZoom} title="Сбросить масштаб">
          ↺ Сброс
        </button>
      )}

      {/* Tooltip */}
      {hoveredNode && (
        <div
          className="graph-tooltip"
          style={{
            left: tooltipPos.x + 10,
            top: tooltipPos.y + 10,
          }}
        >
          <div className="tooltip-label">{hoveredNode.label}</div>
          <div className="tooltip-type">
            {hoveredNode.type === 'domain' ? 'Домен' : `Термин (${hoveredNode.parent})`}
          </div>
        </div>
      )}
    </div>
  )
}