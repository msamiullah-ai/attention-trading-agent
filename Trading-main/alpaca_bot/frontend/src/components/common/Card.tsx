import React from 'react'
import { DemoBadge } from './DemoBadge'

interface CardProps {
  /** Section title, rendered as the card's uppercase eyebrow heading. */
  title: string
  /** Shows the persistent "DEMO DATA" badge in the header when true. */
  isDemo?: boolean
  /** Optional right-aligned header content (e.g. a summary stat). */
  action?: React.ReactNode
  className?: string
  children: React.ReactNode
}

/**
 * Shared card chrome used by every dashboard widget: border, background,
 * header row, and body. Centralizing this in one place is what keeps every
 * widget visually consistent — previously each display component hand-rolled
 * its own inline styles, which is how the account summary card ended up
 * looking structurally different from the rest.
 */
export const Card: React.FC<CardProps> = ({
  title,
  isDemo = false,
  action,
  className = '',
  children,
}) => (
  <div className={`card ${className}`.trim()}>
    <div className="card-header">
      <h2 className="h2">{title}</h2>

      <div className="flex items-center gap-2">
        {action}
        {isDemo && <DemoBadge />}
      </div>
    </div>

    <div className="card-body">{children}</div>
  </div>
)
