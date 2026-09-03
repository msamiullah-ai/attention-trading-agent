import { useQuery } from '@tanstack/react-query'

/** One LLM review and how the underlying then moved.
 *
 *  `correct` is nullable and means THREE things, not two: right, wrong, or
 *  "the market did not move enough to say". The third is the common case
 *  shortly after a verdict, and rendering it as a failure would make a working
 *  reviewer look broken. */
export interface AdvisorVerdict {
  ts: string
  symbol: string
  side: string
  action: 'confirm' | 'shrink' | 'veto'
  reason: string
  price_at: number
  move_pct: number | null
  correct: boolean | null
}

export interface AdvisorStats {
  overall_rate: number
  overall_n: number
  veto_rate: number
  veto_n: number
  pending: number
  model: string
}

/** The equity/options split as it stands right now. */
export interface Allocation {
  equity_budget: number
  options_budget: number
  reserve: number
  regime: string
  reason: string
  enabled: boolean
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path)
  if (!res.ok) throw new Error(`Failed to fetch ${path}: ${res.status}`)
  return res.json()
}

export function useAdvisorVerdicts(limit = 20) {
  return useQuery({
    queryKey: ['advisor-verdicts', limit],
    queryFn: () => get<AdvisorVerdict[]>(`/api/advisor/verdicts?limit=${limit}`),
    refetchInterval: 60_000,
    retry: 0,
  })
}

export function useAdvisorStats() {
  return useQuery({
    queryKey: ['advisor-stats'],
    queryFn: () => get<AdvisorStats>('/api/advisor/stats'),
    refetchInterval: 60_000,
    retry: 0,
  })
}

export function useAllocation() {
  return useQuery({
    queryKey: ['allocation'],
    queryFn: () => get<Allocation>('/api/allocation'),
    // Recomputed server-side from live equity, so it moves with the account
    // rather than with a log. Polled a little faster than the trade cycle.
    refetchInterval: 90_000,
    retry: 0,
  })
}
