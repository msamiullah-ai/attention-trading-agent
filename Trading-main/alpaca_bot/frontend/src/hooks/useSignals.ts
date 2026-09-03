import { useQuery } from '@tanstack/react-query'
import type { SignalData } from '../types/models'

export type { SignalData }

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
  })
}
