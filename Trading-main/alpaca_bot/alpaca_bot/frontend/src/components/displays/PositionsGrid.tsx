import React from 'react'

export interface PositionSnapshot {
  symbol: string
  qty: number
  market_value: number
  avg_entry_price: number
  unrealized_pl: number
  unrealized_plpc: number
  side: string
}

const PLByPositionChart: React.FC<{ positions: PositionSnapshot[] }> = ({ positions }) => {
  const maxAbs = Math.max(...positions.map((p) => Math.abs(p.unrealized_pl)), 1)

  return (
    <div className="mt-4">
      <div className="text-xs text-muted mono mb-2">Unrealized P/L by position</div>

      <div style={{ display: 'flex', alignItems: 'flex-end', gap: 8, height: 90 }}>
        {positions.map((pos) => {
          const heightPct = Math.max((Math.abs(pos.unrealized_pl) / maxAbs) * 100, 4)
          const isPositive = pos.unrealized_pl >= 0

          return (
            <div
              key={pos.symbol}
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'flex-end',
                flex: 1,
                height: '100%',
              }}
            >
              <div
                className="mono text-xs"
                style={{ color: isPositive ? 'var(--green)' : 'var(--red)', marginBottom: 2 }}
              >
                {isPositive ? '+' : ''}
                {pos.unrealized_pl.toFixed(0)}
              </div>
              <div
                style={{
                  width: '100%',
                  maxWidth: 28,
                  height: `${heightPct}%`,
                  backgroundColor: isPositive ? 'var(--green)' : 'var(--red)',
                  opacity: 0.85,
                }}
              />
              <div className="mono text-xs text-muted mt-1">{pos.symbol}</div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export const PositionsGrid: React.FC<{ positions: PositionSnapshot[] }> = ({ positions }) => (
  <div className="positions-grid">
    <h2 className="h2 mb-4">Open Positions</h2>

    {positions.length === 0 && (
      <p className="text-muted">No open positions</p>
    )}

    {positions.length > 0 && (
      <>
        <div className="table-scroll">
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
                  <td className="text-muted">
                    {pos.side}
                  </td>
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
  </div>
)
