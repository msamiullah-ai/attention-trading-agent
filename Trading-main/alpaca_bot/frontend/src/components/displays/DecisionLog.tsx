import React from 'react'
import type { OptionCycle } from '../../hooks/useOptions'

interface DecisionLogProps {
  cycles?: OptionCycle[]
  limit?: number
}

/** What the overlay decided, cycle by cycle — including the cycles that traded
 *  nothing.
 *
 *  That inclusion is the whole point. A panel showing only fills cannot show
 *  the agent declining for a stated reason, and declining is most of what it
 *  does: "bearish signal with no shares; level 1 has no bearish option
 *  expression" is a decision, not an absence of one. A dashboard that hides it
 *  makes a working agent look idle. */
export const DecisionLog: React.FC<DecisionLogProps> = ({ cycles = [], limit = 12 }) => {
  const recent = [...cycles].reverse().slice(0, limit)

  if (recent.length === 0) {
    return (
      <div className="decision-log">
        <h2 className="h2 mb-3">Decisions</h2>
        <p className="text-muted mono">
          No cycles recorded yet. The overlay writes one entry per cycle to
          state/options_cycles.jsonl once it runs.
        </p>
      </div>
    )
  }

  return (
    <div className="decision-log">
      <div className="flex items-baseline justify-between mb-3">
        <h2 className="h2">Decisions</h2>
        <span className="mono text-xs text-muted">last {recent.length} cycles</span>
      </div>

      <div className="flex flex-col gap-2">
        {recent.map((c, i) => {
          const traded = c.orders.length > 0
          const reasons: string[] = [
            ...c.skipped.map((s) => String(s.reason ?? '')),
            ...c.plans
              .filter((p) => p.action === 'none')
              .map((p) => String(p.reason ?? '')),
          ].filter(Boolean)

          return (
            <div
              key={`${c.ts}-${i}`}
              className="border p-2 text-xs mono"
              style={{
                borderColor: traded ? 'var(--green)' : 'var(--border)',
                backgroundColor: traded ? 'rgba(57, 255, 106, 0.06)' : 'transparent',
              }}
            >
              <div className="flex justify-between gap-3">
                <span className={traded ? 'positive' : 'text-muted'}>
                  {traded ? '● traded' : '○ no trade'}
                </span>
                <span className="text-muted">{c.ts.slice(11, 19)}</span>
              </div>

              <div className="mt-1 text-fg">{c.summary}</div>

              {reasons.length > 0 && (
                <ul className="mt-1 text-muted">
                  {/* De-duplicated: one repeated reason across twenty symbols is
                      one fact, not twenty. */}
                  {Array.from(new Set(reasons)).slice(0, 3).map((r, j) => (
                    <li key={j}>— {r}</li>
                  ))}
                </ul>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
