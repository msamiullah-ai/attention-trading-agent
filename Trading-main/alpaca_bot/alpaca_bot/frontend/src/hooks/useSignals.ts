import { useQuery } from '@tanstack/react-query'

export interface SignalData {
  [symbol: string]: 'BUY' | 'SELL' | 'HOLD'
}

export function useSignals() {
  return useQuery({
    queryKey: ['signals'],
    queryFn: async (): Promise<SignalData> => {
      const res = await fetch('/api/signals')
      if (!res.ok) {
        throw new Error(`Failed to fetch signals: ${res.status}`)
      }
      return res.json()
    },
    refetchInterval: 180_000,
    retry: 0, // Don't retry — the caller falls back to demo data on error
    staleTime: 30_000,
  })
}
