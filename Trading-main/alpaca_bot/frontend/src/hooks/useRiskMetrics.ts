import { useQuery } from '@tanstack/react-query'

/** Mirrors RiskMetricsJson in api_types.py. Percent fields arrive already
 *  scaled (65.0 = 65%), so nothing here multiplies by 100. `expectancy` is
 *  R-multiples per trade, not a percent. */
export interface RiskMetricsData {
  n: number
  win_rate: number
  loss_rate: number
  expectancy: number
  breakeven_win_rate: number
  margin_of_safety: number
  kelly_position_pct: number
  strategy: string
  window: number
}

export function useRiskMetrics() {
  return useQuery({
    queryKey: ['risk-metrics'],
    queryFn: async (): Promise<RiskMetricsData> => {
      const res = await fetch('/api/risk-metrics')
      if (!res.ok) {
        throw new Error(`Failed to fetch risk metrics: ${res.status}`)
      }
      return res.json()
    },
    refetchInterval: 180_000,
    retry: 0, // Don't retry — the caller falls back to demo data on error
    staleTime: 30_000,
  })
}
