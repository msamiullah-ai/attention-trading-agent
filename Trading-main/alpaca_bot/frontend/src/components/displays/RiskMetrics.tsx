import React from 'react'
import type { ExpectancyStats } from '../../types/models'

export type { ExpectancyStats }

interface RiskMetricsProps {
  expectancy?: ExpectancyStats
}

// There is currently no backend endpoint that supplies expectancy stats, so
// this panel shows an honest "no data" state rather than fabricated
// figures. Wire this up to a real data source once one exists.
//
// Field units matter here (see risk.ExpectancyStats / types/models.ts):
// `expectancy` is DOLLARS per trade and `expectancy_r` is an R-multiple —
// neither is a percentage. Only win_rate/loss_rate/breakeven_win_rate/
// margin_of_safety are fractions that get rendered as %.
export const RiskMetrics: React.FC<RiskMetricsProps> = ({ expectancy }) => {
  if (!expectancy) {
    return (
      <div className="risk-metrics">
        <h2 className="h2 mb-3">Risk Metrics</h2>
        <p className="text-muted mono text-sm">
          No expectancy data available yet.
        </p>
      </div>
    )
  }

  return (
    <div className="risk-metrics">
      <h2 className="h2 mb-3">Risk Metrics</h2>

      <div className="grid grid-cols-2 gap-2 text-xs mono">
        <div>
          Expectancy:{' '}
          <span className={expectancy.expectancy >= 0 ? 'positive' : 'negative'}>
            {expectancy.expectancy >= 0 ? '+' : ''}
            ${expectancy.expectancy.toFixed(2)}/trade
          </span>
        </div>
        <div>
          Expectancy (R):{' '}
          <span className={expectancy.expectancy_r >= 0 ? 'positive' : 'negative'}>
            {expectancy.expectancy_r >= 0 ? '+' : ''}
            {expectancy.expectancy_r.toFixed(2)}R
          </span>
        </div>
        <div>Sample size: {expectancy.n}</div>
        <div>Reward:Risk: {expectancy.reward_risk_ratio.toFixed(2)}</div>
        <div>Win rate: {(expectancy.win_rate * 100).toFixed(1)}%</div>
        <div>Loss rate: {(expectancy.loss_rate * 100).toFixed(1)}%</div>
        <div>Avg win: ${expectancy.avg_win.toFixed(2)}</div>
        <div>Avg loss: ${expectancy.avg_loss.toFixed(2)}</div>
        <div>Breakeven WR: {(expectancy.breakeven_win_rate * 100).toFixed(1)}%</div>
        <div>Margin of Safety: {(expectancy.margin_of_safety * 100).toFixed(1)}%</div>
      </div>
    </div>
  )
}
