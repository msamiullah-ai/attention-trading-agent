import React from 'react'
import { useQuery } from '@tanstack/react-query'
import { AccountSummary, type AccountSnapshot } from '../components/displays/AccountSummary'
import { PositionsGrid } from '../components/displays/PositionsGrid'
import { EquityCurveChart } from '../components/displays/EquityCurveChart'
import { RiskMetrics } from '../components/displays/RiskMetrics'
import { StrategySignalPanel } from '../components/displays/StrategySignalPanel'
import { useSignals } from '../hooks/useSignals'
import { usePositions } from '../hooks/usePositions'
import { useEquityCurve } from '../hooks/useEquityCurve'
import { useAuthStore } from '../stores/useAuthStore'
import { mockAccount, mockPositions, mockSignals, mockEquityCurve } from '../api/mockData'

export const Dashboard: React.FC = () => {
  const {
    data: account,
    isLoading: loadingAccount,
    isError: errorAccount,
  } = useQuery({
    queryKey: ['account'],
    queryFn: async (): Promise<AccountSnapshot> => {
      const res = await fetch('/api/account')

      if (!res.ok) {
        throw new Error(`Failed to fetch account: ${res.status}`)
      }

      return res.json()
    },
    refetchInterval: 180_000,
    retry: 0, // Don't retry — we fall back to demo data on error
  })

  const {
    data: signals,
    isLoading: loadingSignals,
    isError: errorSignals,
  } = useSignals()

  const {
    data: positions,
    isLoading: loadingPositions,
    isError: errorPositions,
  } = usePositions()

  const {
    data: equityCurve,
    isLoading: loadingEquityCurve,
    isError: errorEquityCurve,
  } = useEquityCurve()

  const { paper } = useAuthStore()

  const backendOffline =
    errorAccount || errorSignals || errorPositions || errorEquityCurve

  // Fill in demo data whenever the backend is unreachable, so the
  // dashboard is always fully explorable instead of showing an empty state.
  const displayAccount = backendOffline ? mockAccount : (account ?? mockAccount)
  const displayPositions = backendOffline ? mockPositions : (positions ?? mockPositions)
  const displaySignals = backendOffline ? mockSignals : (signals ?? mockSignals)
  const displayEquityCurve = backendOffline ? mockEquityCurve : (equityCurve ?? mockEquityCurve)

  if (loadingAccount || loadingSignals || loadingPositions || loadingEquityCurve) {
    return (
      <div className="min-h-screen bg-bg text-fg p-6">
        <div className="max-w-7xl mx-auto">
          <h1 className="text-3xl font-bold text-accent mb-2">
            Alpaca Trading Bot
          </h1>

          <p className="text-muted mono">
            Loading dashboard...
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-bg text-fg p-4 md:p-6">
      <div className="max-w-7xl mx-auto">

        <header className="mb-8">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">

            <div>
              <h1 className="text-3xl font-bold text-accent tracking-tight">
                Alpaca Trading Bot
              </h1>

              <p className="text-muted mt-2 mono">
                Mode: {paper ? 'PAPER' : 'LIVE'} |{' '}
                {displayAccount?.mode || 'UNKNOWN'}
              </p>
            </div>

            <div
              className={`px-4 py-2 border text-sm mono ${
                backendOffline ? 'negative' : 'positive'
              }`}
              style={{
                borderColor: backendOffline ? 'var(--red)' : 'var(--green)',
                backgroundColor: backendOffline
                  ? 'rgba(255, 59, 59, 0.08)'
                  : 'rgba(57, 255, 106, 0.08)',
              }}
            >
              {backendOffline
                ? '● Backend disconnected'
                : '● Backend connected'}
            </div>

          </div>

          {!backendOffline && displayAccount?.blocked && (
            <p className="mt-3 negative">
              Account is blocked for trading
            </p>
          )}
        </header>

        {backendOffline && (
          <div
            className="mb-6 border p-4"
            style={{ borderColor: 'var(--red)', backgroundColor: 'rgba(255, 59, 59, 0.08)' }}
          >
            <p className="font-semibold negative">
              Backend API unavailable
            </p>

            <p className="mt-1 text-sm text-muted">
              Showing demo data below so you can preview the dashboard.
              Start the Python backend to display real account, position,
              signal, and risk data.
            </p>
          </div>
        )}

        <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <AccountSummary account={displayAccount} />

          <PositionsGrid positions={displayPositions} />

          <EquityCurveChart data={displayEquityCurve} />
        </section>

        <section className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-6">
          <StrategySignalPanel signals={displaySignals} />

          <RiskMetrics account={displayAccount} />
        </section>

      </div>
    </div>
  )
}
