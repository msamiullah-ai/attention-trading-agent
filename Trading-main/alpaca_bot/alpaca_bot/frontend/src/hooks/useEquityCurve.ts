import { useQuery } from '@tanstack/react-query'

export interface EquityPoint {
  date: string
  equity: number
}

export function useEquityCurve() {
  return useQuery({
    queryKey: ['equity-curve'],
    queryFn: async (): Promise<EquityPoint[]> => {
      const res = await fetch('/api/equity-curve')
      if (!res.ok) {
        throw new Error(`Failed to fetch equity curve: ${res.status}`)
      }
      return res.json()
    },
    refetchInterval: 180_000,
    retry: 0, // Don't retry — the caller falls back to demo data on error
    staleTime: 30_000,
  })
}
