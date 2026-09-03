import { useQuery } from '@tanstack/react-query'

/** Mirrors OptionPositionJson in api_types.py.
 *
 *  `collateral` is the field an equity-shaped view has no room for and the one
 *  that matters: a short $92 put obliges us to buy 100 shares at the strike, so
 *  it is a $9,200 commitment while `market_value` reads as $120. Rendering only
 *  market value understates the position by two orders of magnitude. */
export interface OptionPosition {
  symbol: string
  underlying: string
  right: 'call' | 'put'
  strike: number
  expiry: string
  dte: number
  qty: number
  side: 'long' | 'short'
  market_value: number
  unrealized_pl: number
  collateral: number
  moneyness_pct: number | null   // null = the underlying was not quoted; unknown, not safe
  itm: boolean | null
}

/** Mirrors OptionPlanJson. One entry per cycle, INCLUDING the ones that traded
 *  nothing — those carry the reason, and they are most of them. */
export interface OptionCycle {
  ts: string
  summary: string
  equity: number
  collateral_posted: number
  open_options: number
  plans: Array<Record<string, unknown>>
  orders: Array<Record<string, unknown>>
  skipped: Array<Record<string, unknown>>
}

export interface OptionsConfig {
  enabled: boolean
  target_delta: number
  delta_min: number
  delta_max: number
  max_collateral_pct: number   // already percent (50.0 = 50%)
  max_positions: number
  max_orders_per_day: number
  min_credit: number
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path)
  if (!res.ok) throw new Error(`Failed to fetch ${path}: ${res.status}`)
  return res.json()
}

export function useOptionPositions() {
  return useQuery({
    queryKey: ['option-positions'],
    queryFn: () => get<OptionPosition[]>('/api/options/positions'),
    refetchInterval: 180_000,
    retry: 0,
    staleTime: 30_000,
  })
}

export function useOptionCycles(limit = 25) {
  return useQuery({
    queryKey: ['option-cycles', limit],
    queryFn: () => get<OptionCycle[]>(`/api/options/cycles?limit=${limit}`),
    refetchInterval: 180_000,
    retry: 0,
    staleTime: 30_000,
  })
}

export function useOptionsConfig() {
  return useQuery({
    queryKey: ['options-config'],
    queryFn: () => get<OptionsConfig>('/api/options/config'),
    // Config changes only when someone edits options.yaml and restarts, so
    // polling it every three minutes would be noise.
    staleTime: Infinity,
    retry: 0,
  })
}
