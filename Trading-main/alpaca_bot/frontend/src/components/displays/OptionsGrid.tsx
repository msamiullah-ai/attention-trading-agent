import React from 'react'
import type { OptionPosition } from '../../hooks/useOptions'

interface OptionsGridProps {
  positions?: OptionPosition[]
  collateralPosted?: number
  equity?: number
}

const money = (v: number) =>
  `${v < 0 ? '-' : ''}$${Math.abs(v).toLocaleString(undefined, { maximumFractionDigits: 0 })}`

/** Held option contracts.
 *
 *  Sorted by cushion, thinnest first, so whatever is closest to trouble reads
 *  at the top rather than wherever the broker happened to return it.
 *
 *  Collateral is given its own column deliberately. A short put's market value
 *  is the premium — a hundred dollars or so — while the obligation behind it is
 *  strike x 100. Showing only market value makes a $9,200 commitment look like
 *  a $120 one. */
export const OptionsGrid: React.FC<OptionsGridProps> = ({
  positions = [], collateralPosted = 0, equity = 0,
}) => {
  if (positions.length === 0) {
    return (
      <div className="options-grid">
        <h2 className="h2 mb-3">Options</h2>
        <p className="text-muted mono">No option positions open.</p>
      </div>
    )
  }

  const deployed = equity > 0 ? (collateralPosted / equity) * 100 : 0

  return (
    <div className="options-grid">
      <div className="flex items-baseline justify-between mb-3">
        <h2 className="h2">Options</h2>
        <span className="mono text-xs text-muted">
          {positions.length} open · {money(collateralPosted)} collateral
          {equity > 0 && ` (${deployed.toFixed(0)}% of equity)`}
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full mono text-xs">
          <thead>
            <tr className="text-muted text-left">
              <th className="pr-3 pb-1">contract</th>
              <th className="pr-3 pb-1 text-right">strike</th>
              <th className="pr-3 pb-1 text-right">dte</th>
              <th className="pr-3 pb-1 text-right">qty</th>
              <th className="pr-3 pb-1 text-right">cushion</th>
              <th className="pr-3 pb-1 text-right">collateral</th>
              <th className="pb-1 text-right">P/L</th>
            </tr>
          </thead>
          <tbody>
            {positions.map((p) => {
              // null cushion means the underlying was not quoted this cycle.
              // Rendered as unknown rather than as zero: "we could not tell"
              // and "it is at the strike" are different claims, and only one
              // of them is safe to act on.
              const unknown = p.moneyness_pct === null
              const danger = p.itm === true
              const near = !unknown && !danger && (p.moneyness_pct as number) < 1
              const tone = danger || unknown ? 'negative' : near ? '' : 'positive'

              return (
                <tr key={p.symbol} className="border-t" style={{ borderColor: 'var(--border)' }}>
                  <td className="pr-3 py-1">
                    <span className="text-fg">{p.underlying}</span>{' '}
                    <span className="text-muted">
                      {p.right === 'put' ? 'P' : 'C'} {p.expiry.slice(5)}
                    </span>
                    {p.side === 'short' && <span className="text-muted"> · short</span>}
                  </td>
                  <td className="pr-3 py-1 text-right">{p.strike.toFixed(2)}</td>
                  <td className="pr-3 py-1 text-right">{p.dte}</td>
                  <td className="pr-3 py-1 text-right">{p.qty}</td>
                  <td className={`pr-3 py-1 text-right ${tone}`}>
                    {unknown ? 'unquoted' : `${(p.moneyness_pct as number).toFixed(2)}%`}
                    {danger && ' ITM'}
                  </td>
                  <td className="pr-3 py-1 text-right">
                    {p.collateral > 0 ? money(p.collateral) : '—'}
                  </td>
                  <td className={`py-1 text-right ${p.unrealized_pl >= 0 ? 'positive' : 'negative'}`}>
                    {p.unrealized_pl >= 0 ? '+' : ''}{p.unrealized_pl.toFixed(0)}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <p className="mt-2 text-xs text-muted">
        Cushion is the distance to the strike. A short put is collateralised at
        strike × 100; a short call is secured by shares, so it posts no cash and
        shows “—”.
      </p>
    </div>
  )
}
