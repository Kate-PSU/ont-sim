// frontend/src/components/Heatmap.tsx
// Компонент для визуализации heatmap матрицы similarity

interface HeatmapProps {
  data: number[][]
  labels: string[]
}

export function Heatmap({ data, labels }: HeatmapProps) {
  const getColor = (value: number): string => {
    // Цветовая схема от синего (низкая близость) к красному (высокая)
    const hue = (1 - value) * 240 // 240 = blue, 0 = red
    return `hsl(${hue}, 70%, 50%)`
  }

  if (!data.length || !labels.length) {
    return <div className="heatmap-empty">Нет данных для отображения</div>
  }

  return (
    <div className="heatmap-container">
      <table className="heatmap-table">
        <thead>
          <tr>
            <th className="heatmap-corner"></th>
            {labels.map((label, i) => (
              <th key={i} className="heatmap-label heatmap-label-col">
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
            <tr key={i}>
              <th className="heatmap-label heatmap-label-row">
                {labels[i]}
              </th>
              {row.map((value, j) => (
                <td
                  key={j}
                  className="heatmap-cell"
                  style={{ backgroundColor: getColor(value) }}
                  title={`${labels[i]} ↔ ${labels[j]}: ${value.toFixed(3)}`}
                >
                  {value.toFixed(2)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      
      {/* Color scale legend */}
      <div className="heatmap-legend">
        <span>0 (низкая)</span>
        <div className="heatmap-scale" />
        <span>1 (высокая)</span>
      </div>
    </div>
  )
}
