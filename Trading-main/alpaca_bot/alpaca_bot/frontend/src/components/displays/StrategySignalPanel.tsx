import React from 'react'

export interface SignalData {
  [symbol: string]: 'BUY' | 'SELL' | 'HOLD'
}

const SignalBar: React.FC<{ label: string; count: number; total: number; color: string }> = ({
  label,
  count,
  total,
  color,
}) => {
  const pct = total > 0 ? (count / total) * 100 : 0

  return (
    <div className="flex items-center gap-3 mb-2 mono text-xs">
      <div style={{ width: 40, color: 'var(--muted)' }}>{label}</div>
      <div style={{ flex: 1, backgroundColor: 'var(--border)', height: 8 }}>
        <div style={{ width: `${pct}%`, height: '100%', backgroundColor: color }} />
      </div>
      <div style={{ width: 24, textAlign: 'right' }}>{count}</div>
    </div>
  )
}

export const StrategySignalPanel: React.FC<{ signals: SignalData }> = ({ signals }) => {
  const buyCount = Object.values(signals).filter(s => s === 'BUY').length
  const sellCount = Object.values(signals).filter(s => s === 'SELL').length
  const holdCount = Object.values(signals).filter(s => s === 'HOLD').length
  const total = Object.keys(signals).length

  return (
    <div className="signal-panel">
      <h2 className="h2 mb-3">Strategy Signals</h2>

      <div className="grid grid-cols-3 gap-2 text-xs mb-4">
        <div className="status-indicator buy">
          <span>BUY</span>
          <span>{buyCount}</span>
        </div>
        <div className="status-indicator sell">
          <span>SELL</span>
          <span>{sellCount}</span>
        </div>
        <div className="status-indicator neutral">
          <span>HOLD</span>
          <span>{holdCount}</span>
        </div>
      </div>

      <SignalBar label="Buy" count={buyCount} total={total} color="var(--green)" />
      <SignalBar label="Sell" count={sellCount} total={total} color="var(--red)" />
      <SignalBar label="Hold" count={holdCount} total={total} color="var(--muted)" />

      <div className="divider" />

      <p className="mt-2 text-sm text-muted mono">
        Total: {total} signals
      </p>
    </div>
  )
}
