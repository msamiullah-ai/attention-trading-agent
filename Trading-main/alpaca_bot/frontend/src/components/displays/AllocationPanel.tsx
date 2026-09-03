import React from 'react'
import type { Allocation } from '../../hooks/useAgent'

interface Props {
  allocation?: Allocation
}

const money = (v: number) =>
  `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`

/** How the account is split between the equity and options sides.
 *
 *  Rendered as a single bar because the one thing worth seeing at a glance is
 *  that the parts sum to the whole — that was the actual bug this replaced,
 *  two processes each sizing against the full account and committing past
 *  100% between them. */
export const AllocationPanel: React.FC<Props> = ({ allocation }) => {
  if (!allocation) {
    return (
      <div className="allocation-panel">
        <h2 className="h2 mb-3">Allocation</h2>
        <p className="text-muted mono">Not available.</p>
      </div>
    )
  }

  const total =
    allocation.equity_budget + allocation.options_budget + allocation.reserve
  const pct = (v: number) => (total > 0 ? (v / total) * 100 : 0)

  const segments = [
    { label: 'equities', value: allocation.equity_budget, color: 'var(--green)' },
    { label: 'options', value: allocation.options_budget, color: 'var(--accent)' },
    { label: 'reserve', value: allocation.reserve, color: 'var(--border)' },
  ]

  return (
    <div className="allocation-panel">
      <div className="flex items-baseline justify-between mb-3">
        <h2 className="h2">Allocation</h2>
        <span className="mono text-xs text-muted">
          {allocation.enabled ? `regime ${allocation.regime}` : 'not enforced'}
        </span>
      </div>

      {!allocation.enabled && (
        <p className="mb-3 text-xs negative">
          Shown but not applied: execution.allocator is “none”, so each side
          still sizes against the full account independently.
        </p>
      )}

      <div
        className="flex h-6 w-full overflow-hidden border"
        style={{ borderColor: 'var(--border)' }}
      >
        {segments.map((s) =>
          s.value > 0 ? (
            <div
              key={s.label}
              style={{ width: `${pct(s.value)}%`, backgroundColor: s.color }}
              title={`${s.label} ${money(s.value)}`}
            />
          ) : null,
        )}
      </div>

      <div className="mt-3 flex flex-col gap-1 text-xs mono">
        {segments.map((s) => (
          <div key={s.label} className="flex justify-between">
            <span className="text-muted">
              <span
                className="inline-block w-2 h-2 mr-2"
                style={{ backgroundColor: s.color }}
              />
              {s.label}
            </span>
            <span className={s.value > 0 ? 'text-fg' : 'text-muted'}>
              {money(s.value)}{' '}
              <span className="text-muted">({pct(s.value).toFixed(0)}%)</span>
            </span>
          </div>
        ))}
      </div>

      {/* The reason carries the whole derivation — anchor, then each input's
          signed contribution. An allocation nobody can reconstruct is one
          nobody should trust. */}
      <p className="mt-3 text-xs text-muted">{allocation.reason}</p>

      {allocation.options_budget === 0 && (
        <p className="mt-2 text-xs text-muted">
          A cash-secured put ties up strike × 100, so a small account cannot
          reach one at any split. Reserving for it would only starve the side
          that can trade.
        </p>
      )}
    </div>
  )
}
