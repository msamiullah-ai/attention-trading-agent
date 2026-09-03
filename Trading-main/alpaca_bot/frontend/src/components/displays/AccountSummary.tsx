import React from 'react'

export interface AccountSnapshot {
  equity?: number
  cash?: number
  buying_power?: number
  daily_pnl_pct?: number
  mode?: string
  blocked?: boolean
}

interface AccountSummaryProps {
  account?: Partial<AccountSnapshot> | null
}

const formatNumber = (value: unknown): string => {
  const number = typeof value === 'number' && Number.isFinite(value)
    ? value
    : 0

  return number.toLocaleString()
}

const formatPercent = (value: unknown): string => {
  const number = typeof value === 'number' && Number.isFinite(value)
    ? value
    : 0

  return number.toFixed(2)
}

export const AccountSummary: React.FC<AccountSummaryProps> = ({
  account,
}) => {
  const equity = account?.equity ?? 0
  const cash = account?.cash ?? 0
  const buyingPower = account?.buying_power ?? 0
  const dailyPnl = account?.daily_pnl_pct ?? 0
  const mode = account?.mode ?? 'UNKNOWN'
  const blocked = account?.blocked ?? false

  return (
    <div
      className="account-summary"
      style={{
        borderColor: 'var(--border)',
        backgroundColor: 'var(--surface)',
        color: 'var(--fg)',
        borderWidth: '1px',
        borderStyle: 'solid',
        padding: '16px',
      }}
    >
      <h2 className="h2 mb-4">
        Account Summary
      </h2>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: '12px',
        }}
      >
        <div style={{ color: 'var(--fg)' }}>
          <div style={{ color: 'var(--muted)', marginBottom: '4px' }}>
            Equity
          </div>
          <div className="mono">
            ${formatNumber(equity)}
          </div>
        </div>

        <div style={{ color: 'var(--fg)' }}>
          <div style={{ color: 'var(--muted)', marginBottom: '4px' }}>
            Cash
          </div>
          <div className="mono">
            ${formatNumber(cash)}
          </div>
        </div>

        <div style={{ color: 'var(--fg)' }}>
          <div style={{ color: 'var(--muted)', marginBottom: '4px' }}>
            Buying Power
          </div>
          <div className="mono">
            ${formatNumber(buyingPower)}
          </div>
        </div>

        <div>
          <div style={{ color: 'var(--muted)', marginBottom: '4px' }}>
            Day P/L
          </div>

          <div
            className="mono"
            style={{
              color: dailyPnl >= 0 ? 'var(--green)' : 'var(--red)',
            }}
          >
            {dailyPnl >= 0 ? '+' : ''}
            {formatPercent(dailyPnl)}%
          </div>
        </div>
      </div>

      <p style={{ marginTop: '12px' }}>
        <span style={{ color: 'var(--muted)' }}>Mode: </span>
        {mode}
      </p>

      {blocked && (
        <p
          style={{
            color: 'var(--red)',
            marginTop: '4px',
          }}
        >
          Blocked - no trading
        </p>
      )}
    </div>
  )
}
