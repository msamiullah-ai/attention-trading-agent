import { useQuery } from '@tanstack/react-query'
import type { PositionSnapshot } from '../types/models'

export type { PositionSnapshot }

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
  })
}
