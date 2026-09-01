import React from 'react'
import type { EquityPoint } from '../../hooks/useEquityCurve'

interface EquityCurveChartProps {
  data?: EquityPoint[]
}

const WIDTH = 560
const HEIGHT = 160
const PAD_X = 4
const PAD_Y = 10

export const EquityCurveChart: React.FC<EquityCurveChartProps> = ({ data = [] }) => {
  if (data.length === 0) {
    return (
      <div className="equity-curve-chart">
        <h2 className="h2 mb-3">Equity curve</h2>
        <p className="text-muted mono">
          No equity data available yet.
        </p>
      </div>
    )
  }

  const values = data.map((d) => d.equity)
  const max = Math.max(...values)
  const min = Math.min(...values)
  const range = max - min || 1

  const first = values[0]
  const last = values[values.length - 1]
  const isUp = last >= first
  const changePct = first !== 0 ? ((last - first) / first) * 100 : 0
  const lineColor = isUp ? 'var(--green)' : 'var(--red)'
  const fillColor = isUp ? 'rgba(57, 255, 106, 0.12)' : 'rgba(255, 59, 59, 0.12)'

  const points = data.map((d, i) => {
    const x = PAD_X + (i / (data.length - 1 || 1)) * (WIDTH - PAD_X * 2)
    const y = PAD_Y + (1 - (d.equity - min) / range) * (HEIGHT - PAD_Y * 2)
    return [x, y] as const
  })

  const linePath = points
    .map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`)
    .join(' ')

  const areaPath =
    `M${points[0][0].toFixed(1)},${(HEIGHT - PAD_Y).toFixed(1)} ` +
    points.map(([x, y]) => `L${x.toFixed(1)},${y.toFixed(1)}`).join(' ') +
    ` L${points[points.length - 1][0].toFixed(1)},${(HEIGHT - PAD_Y).toFixed(1)} Z`

  return (
    <div className="equity-curve-chart">
      <h2 className="h2 mb-3">Equity curve</h2>

      <div className="flex items-baseline gap-3 mb-2 mono">
        <span style={{ fontSize: 20, color: 'var(--fg)' }}>
          ${last.toLocaleString(undefined, { maximumFractionDigits: 0 })}
        </span>
        <span className={isUp ? 'positive' : 'negative'}>
          {isUp ? '+' : ''}
          {changePct.toFixed(2)}%
        </span>
      </div>

      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        width="100%"
        height={HEIGHT}
        preserveAspectRatio="none"
        role="img"
        aria-label="Equity curve over time"
      >
        <title>Equity curve over time</title>
        <path d={areaPath} fill={fillColor} stroke="none" />
        <path d={linePath} fill="none" stroke={lineColor} strokeWidth={1.5} />
      </svg>

      <div className="flex justify-between mono text-xs text-muted mt-1">
        <span>{data[0].date}</span>
        <span>{data[data.length - 1].date}</span>
      </div>
    </div>
  )
}
