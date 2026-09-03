import React from 'react'
import type { PositionSnapshot } from '../../types/models'
import { Card } from '../common/Card'

export type { PositionSnapshot }

const PLByPositionChart: React.FC<{ positions: PositionSnapshot[] }> = ({ positions }) => {
  const maxAbs = Math.max(...positions.map((p) => Math.abs(p.unrealized_pl)), 1)

  return (
    <div className="mt-5">
      <div className="text-xs text-muted mono mb-2">Unrealized P/L by position</div>

      <div className="flex items-end gap-2" style={{ height: 90 }}>
        {positions.map((pos) => {
          const heightPct = Math.max((Math.abs(pos.unrealized_pl) / maxAbs) * 100, 4)
          const isPositive = pos.unrealized_pl >= 0

          return (
            <div
              key={pos.symbol}
              className="flex flex-col items-center justify-end flex-1 h-full min-w-0"
            >
              <div
                className="mono text-xs mb-0.5 whitespace-nowrap"
                style={{ color: isPositive ? 'var(--color-green)' : 'var(--color-red)' }}
              >
                {isPositive ? '+' : ''}
                {pos.unrealized_pl.toFixed(0)}
              </div>
              <div
                className="w-full"
                style={{
                  maxWidth: 28,
                  height: `${heightPct}%`,
                  backgroundColor: isPositive ? 'var(--color-green)' : 'var(--color-red)',
                  opacity: 0.85,
                }}
              />
              <div className="mono text-xs text-muted mt-1 truncate w-full text-center">
                {pos.symbol}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

interface PositionsGridProps {
  positions: PositionSnapshot[]
  isDemo?: boolean
}

export const PositionsGrid: React.FC<PositionsGridProps> = ({ positions, isDemo = false }) => (
  <Card title="Open Positions" isDemo={isDemo} action={<span className="mono text-xs text-muted">{positions.length} open</span>}>
    {positions.length === 0 && <p className="text-muted mono text-sm">No open positions</p>}

    {positions.length > 0 && (
      <>
        {/* min-w-0 on the scroll wrapper is what actually fixes the table
            "bleeding" into neighboring cards: a grid/flex item defaults to
            min-width: auto, so a wide table forces its whole column wider
            instead of scrolling inside its own card. The Card component
            sets min-width: 0 on .card and .card-body for the same reason. */}
        <div className="table-scroll min-w-0">
          <table className="table">
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Side</th>
                <th>Qty</th>
                <th>Entry</th>
                <th>Market</th>
                <th>P/L</th>
                <th>P/L %</th>
              </tr>
            </thead>
            <tbody>
              {positions.map((pos) => (
                <tr key={pos.symbol}>
                  <td className="font-medium">{pos.symbol}</td>
                  <td className="text-muted">{pos.side}</td>
                  <td>{pos.qty.toFixed(4)}</td>
                  <td>${pos.avg_entry_price.toFixed(2)}</td>
                  <td>${pos.market_value.toFixed(2)}</td>
                  <td className={pos.unrealized_pl >= 0 ? 'positive' : 'negative'}>
                    ${pos.unrealized_pl.toFixed(2)}
                  </td>
                  <td className={pos.unrealized_plpc >= 0 ? 'positive' : 'negative'}>
                    {pos.unrealized_plpc.toFixed(2)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <PLByPositionChart positions={positions} />
      </>
    )}
  </Card>
)
