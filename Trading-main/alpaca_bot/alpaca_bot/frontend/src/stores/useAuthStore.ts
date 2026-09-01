import { create } from 'zustand'

export type AuthState = {
  paper: boolean
  mode: 'PAPER' | 'LIVE'
  setPaper: (paper: boolean) => void
}

export const useAuthStore = create<AuthState>((set) => ({
  paper: true,
  mode: 'PAPER',
  setPaper: (paper: boolean) =>
    set(() => ({ paper, mode: paper ? 'PAPER' : 'LIVE' })),
}))