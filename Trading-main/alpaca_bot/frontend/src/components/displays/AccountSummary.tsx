import React from 'react'
import type { AccountSnapshot } from '../../types/models'
import { Card } from '../common/Card'

export type { AccountSnapshot }

interface AccountSummaryProps {
  account?: Partial<AccountSnapshot> | null
  isDemo?: boolean
}

const formatCurrency = (value: unknown): string => {
  const number = typeof value === 'number' && Number.isFinite(value) ? value : 0
  return number.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}

// `daily_pnl_pct` is a fraction (e.g. 0.0142 === +1.42%), matching the
// Python backend's convention in dashboard.py. Multiply by 100 here at the
// render boundary rather than storing/passing around a pre-multiplied value.
const formatPercent = (value: unknown): string => {
  const number = typeof value === 'number' && Number.isFinite(value) ? value : 0
  return (number * 100).toFixed(2)
}

export const AccountSummary: React.FC<AccountSummaryProps> = ({ account, isDemo = false }) => {
  const equity = account?.equity ?? 0
  const cash = account?.cash ?? 0
  const buyingPower = account?.buying_power ?? 0
  const dailyPnl = account?.daily_pnl_pct ?? 0
  const blocked = account?.blocked ?? false

  return (
    <Card title="Account Summary" isDemo={isDemo}>
      <div className="kpi-strip">
        <div>
          <div className="kpi-label">Equity</div>
          <div className="kpi-value">${formatCurrency(equity)}</div>
        </div>

        <div>
          <div className="kpi-label">Cash</div>
          <div className="kpi-value">${formatCurrency(cash)}</div>
        </div>

        <div>
          <div className="kpi-label">Buying Power</div>
          <div className="kpi-value">${formatCurrency(buyingPower)}</div>
        </div>

        <div>
          <div className="kpi-label">Day P/L</div>
          <div className={`kpi-value ${dailyPnl >= 0 ? 'positive' : 'negative'}`}>
            {dailyPnl >= 0 ? '+' : ''}
            {formatPercent(dailyPnl)}%
          </div>
        </div>
      </div>

      {blocked && (
        <p className="mono negative mt-4 text-xs">
          Account is blocked — no new orders will be placed
        </p>
      )}
    </Card>
  )
}
