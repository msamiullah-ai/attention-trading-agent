import { useQuery } from '@tanstack/react-query'

export interface PositionSnapshot {
  symbol: string
  qty: number
  market_value: number
  avg_entry_price: number
  unrealized_pl: number
  unrealized_plpc: number
  side: string
}

export function usePositions() {
  return useQuery({
    queryKey: ['positions'],
    queryFn: async (): Promise<PositionSnapshot[]> => {
      const res = await fetch('/api/positions')
      if (!res.ok) {
        throw new Error(`Failed to fetch positions: ${res.status}`)
      }
      return res.json()
    },
    refetchInterval: 180_000,
    retry: 0, // Don't retry — the caller falls back to demo data on error
    staleTime: 30_000,
  })
}
