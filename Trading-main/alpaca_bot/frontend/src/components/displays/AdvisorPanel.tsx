import React from 'react'
import type { AdvisorStats, AdvisorVerdict } from '../../hooks/useAgent'

interface Props {
  verdicts?: AdvisorVerdict[]
  stats?: AdvisorStats
}

const ACTION_TONE: Record<string, string> = {
  veto: 'var(--red)',
  shrink: 'var(--yellow, #d8a200)',
  confirm: 'var(--green)',
}

/** What the LLM reviewer decided, in its own words, and whether it was right.
 *
 *  The reasons are shown in full rather than summarised. A risk layer whose
 *  reasoning is hidden is one nobody can overrule, and the single most useful
 *  thing on this panel is catching the reviewer vetoing on something vague —
 *  which is only visible if you can read what it actually said. */
export const AdvisorPanel: React.FC<Props> = ({ verdicts = [], stats }) => {
  if (!stats && verdicts.length === 0) {
    return (
      <div className="advisor-panel">
        <h2 className="h2 mb-3">Reviewer</h2>
        <p className="text-muted mono">
          No advisor configured, or no trades reviewed yet.
        </p>
      </div>
    )
  }

  const settled = stats ? stats.overall_n : 0

  return (
    <div className="advisor-panel">
      <div className="flex items-baseline justify-between mb-3">
        <h2 className="h2">Reviewer</h2>
        <span className="mono text-xs text-muted">{stats?.model}</span>
      </div>

      {stats && (
        <div className="mb-3 text-xs mono flex flex-wrap gap-x-5 gap-y-1">
          {/* Accuracy is meaningless without its sample size, so the count is
              never rendered separately from the rate. */}
          <span className="text-muted">
            veto accuracy{' '}
            <span className={stats.veto_n ? 'text-fg' : 'text-muted'}>
              {stats.veto_n ? `${(stats.veto_rate * 100).toFixed(0)}%` : '—'}
            </span>{' '}
            ({stats.veto_n})
          </span>
          <span className="text-muted">
            overall{' '}
            <span className={settled ? 'text-fg' : 'text-muted'}>
              {settled ? `${(stats.overall_rate * 100).toFixed(0)}%` : '—'}
            </span>{' '}
            ({settled})
          </span>
          <span className="text-muted">{stats.pending} awaiting a verdict</span>
        </div>
      )}

      {settled === 0 && stats && stats.pending > 0 && (
        <p className="mb-3 text-xs text-muted">
          Nothing scored yet: a call is only judged once the underlying has
          moved enough to answer the question, which keeps drift out of the
          record.
        </p>
      )}

      <div className="flex flex-col gap-2 max-h-96 overflow-y-auto">
        {verdicts.map((v, i) => (
          <div
            key={`${v.ts}-${i}`}
            className="border p-2 text-xs mono"
            style={{ borderColor: 'var(--border)' }}
          >
            <div className="flex justify-between gap-3 items-baseline">
              <span>
                <span className="text-fg">{v.symbol}</span>{' '}
                <span className="text-muted">{v.side}</span>{' '}
                <span style={{ color: ACTION_TONE[v.action] ?? 'var(--fg)' }}>
                  {v.action.toUpperCase()}
                </span>
              </span>
              <span className="text-muted">{v.ts.slice(11, 16)}</span>
            </div>

            {v.reason && <div className="mt-1 text-muted">{v.reason}</div>}

            <div className="mt-1">
              {v.correct === null ? (
                <span className="text-muted">
                  {v.move_pct === null
                    ? 'not yet settled'
                    : `moved ${v.move_pct.toFixed(2)}% — too small to judge`}
                </span>
              ) : (
                <span className={v.correct ? 'positive' : 'negative'}>
                  {v.correct ? 'RIGHT' : 'WRONG'}
                  {v.move_pct !== null &&
                    ` — underlying moved ${v.move_pct > 0 ? '+' : ''}${v.move_pct.toFixed(2)}%`}
                </span>
              )}
            </div>
          </div>
        ))}
      </div>

      <p className="mt-2 text-xs text-muted">
        The reviewer can only veto a trade or shrink it. It never proposes one,
        so a mistaken confirm costs nothing the rule layer had not already
        allowed.
      </p>
    </div>
  )
}
