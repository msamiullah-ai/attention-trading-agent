import React from 'react'

export interface AccountSnapshot {
  equity?: number
  daily_pnl_pct?: number
  mode?: string
}

export interface ExpectancyStats {
  n: number
  win_rate: number
  loss_rate: number
  expectancy: number
  breakeven_win_rate: number
  margin_of_safety: number
}

interface RiskMetricsProps {
  account?: Partial<AccountSnapshot> | null
  expectancy?: ExpectancyStats
}

export const RiskMetrics: React.FC<RiskMetricsProps> = ({ account, expectancy }) => {
  const dailyPnl = account?.daily_pnl_pct ?? 0
  const mode = account?.mode ?? 'UNKNOWN'

  return (
    <div className="risk-metrics">
      <h2 className="h2 mb-3">Risk Metrics</h2>

      <div className="grid grid-cols-2 gap-2 text-xs mono">
        <div>
          Daily P/L:{' '}
          <span className={dailyPnl >= 0 ? 'positive' : 'negative'}>
            {(dailyPnl >= 0 ? '+' : '') + dailyPnl.toFixed(2)}%
          </span>
        </div>
        <div>
          Expectancy:{' '}
          <span className={(expectancy?.expectancy ?? 0) >= 0 ? 'positive' : 'negative'}>
            {((expectancy?.expectancy ?? 0) * 100).toFixed(2)}%
          </span>
        </div>
        <div>Breakeven WR: {((expectancy?.breakeven_win_rate ?? 0) * 100).toFixed(2)}%</div>
        <div>Margin of Safety: {((expectancy?.margin_of_safety ?? 0) * 100).toFixed(2)}%</div>
      </div>

      <p className="mt-2 text-sm text-muted">
        Mode: {mode}
      </p>
    </div>
  )
}
