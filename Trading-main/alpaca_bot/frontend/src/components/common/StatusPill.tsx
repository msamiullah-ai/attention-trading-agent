import React from 'react'

interface StatusPillProps {
  positive: boolean
  positiveLabel: string
  negativeLabel: string
}

/**
 * Small dot + label pill used for binary status (backend connected/
 * disconnected, market open/closed, etc.) — one visual language reused
 * anywhere the dashboard needs to show "is this thing okay right now".
 */
export const StatusPill: React.FC<StatusPillProps> = ({
  positive,
  positiveLabel,
  negativeLabel,
}) => (
  <div
    className={`mono flex items-center gap-2 px-3 py-1.5 border text-xs ${
      positive ? 'positive' : 'negative'
    }`}
    style={{
      borderColor: positive ? 'var(--color-green)' : 'var(--color-red)',
      backgroundColor: positive
        ? 'rgba(57, 255, 106, 0.08)'
        : 'rgba(255, 59, 59, 0.08)',
    }}
  >
    <span className="dot" />
    {positive ? positiveLabel : negativeLabel}
  </div>
)
