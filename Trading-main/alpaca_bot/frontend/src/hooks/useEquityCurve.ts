import { useQuery } from '@tanstack/react-query'
import type { EquityPoint } from '../types/models'

export type { EquityPoint }

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
  })
}
