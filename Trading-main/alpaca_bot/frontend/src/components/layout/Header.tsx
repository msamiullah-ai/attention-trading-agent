import React from 'react'
import { StatusPill } from '../common/StatusPill'

interface HeaderProps {
  mode: string
  backendOffline: boolean
}

/**
 * Top-of-page identity + live status. Pulled out of Dashboard.tsx so the
 * page component is left composing layout pieces rather than hand-building
 * markup — the same pattern PositionsGrid/EquityCurveChart/etc. follow.
 */
export const Header: React.FC<HeaderProps> = ({ mode, backendOffline }) => (
  <header className="mb-6 pb-6" style={{ borderBottom: '1px solid var(--color-border)' }}>
    <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
      <div>
        <h1 className="h1 text-accent">Alpaca Trading Bot</h1>
        <p className="text-muted mono mt-1">Mode: {mode}</p>
      </div>

      <StatusPill
        positive={!backendOffline}
        positiveLabel="Backend connected"
        negativeLabel="Backend disconnected"
      />
    </div>
  </header>
)
