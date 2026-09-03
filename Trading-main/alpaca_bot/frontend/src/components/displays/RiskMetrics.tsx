import React from 'react'
import type { RiskMetricsData } from '../../hooks/useRiskMetrics'

export interface AccountSnapshot {
  equity?: number
  daily_pnl_pct?: number
  mode?: string
}

interface RiskMetricsProps {
  account?: Partial<AccountSnapshot> | null
  risk?: RiskMetricsData | null
}

const pct = (value: number | undefined): string =>
  value === undefined ? '--' : `${value.toFixed(2)}%`

export const RiskMetrics: React.FC<RiskMetricsProps> = ({ account, risk }) => {
  const dailyPnl = account?.daily_pnl_pct ?? 0
  const mode = account?.mode ?? 'UNKNOWN'

  // Every *_pct field from /api/risk-metrics is already in percent
  // (api_types.py), so these render as-is. `expectancy` is an R-multiple —
  // "+0.35R per trade" — and is not a percentage at all.
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
          <span className={(risk?.expectancy ?? 0) >= 0 ? 'positive' : 'negative'}>
            {risk ? `${risk.expectancy >= 0 ? '+' : ''}${risk.expectancy.toFixed(3)}R` : '--'}
          </span>
        </div>
        <div>Win Rate: {pct(risk?.win_rate)}</div>
        <div>Breakeven WR: {pct(risk?.breakeven_win_rate)}</div>
        <div>
          Margin of Safety:{' '}
          <span className={(risk?.margin_of_safety ?? 0) >= 0 ? 'positive' : 'negative'}>
            {pct(risk?.margin_of_safety)}
          </span>
        </div>
        <div>Kelly Size: {pct(risk?.kelly_position_pct)}</div>
      </div>

      <p className="mt-2 text-sm text-muted">
        Mode: {mode}
        {risk ? ` | ${risk.strategy} | ${risk.n} closed trades` : ''}
      </p>
    </div>
  )
}
