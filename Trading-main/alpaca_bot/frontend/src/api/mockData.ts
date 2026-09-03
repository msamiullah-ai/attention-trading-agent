import type { AccountSnapshot, PositionSnapshot, SignalData, EquityPoint } from '../types/models'

// Sample data shown when the Python backend isn't reachable, so the
// dashboard is fully explorable (design, layout, states) without it running.
// Every widget that renders this data also renders a persistent "DEMO DATA"
// badge (see Dashboard.tsx / DemoBadge.tsx) — this data should never be
// mistaken for a real account.

export const mockAccount: AccountSnapshot = {
  equity: 104250.32,
  cash: 18320.11,
  buying_power: 36640.22,
  // Fraction, not a percentage — matches the real backend's contract.
  // Renders as "+1.42%".
  daily_pnl_pct: 0.0142,
  mode: 'PAPER',
  blocked: false,
}

export const mockPositions: PositionSnapshot[] = [
  {
    symbol: 'AAPL',
    qty: 40,
    market_value: 9412.8,
    avg_entry_price: 221.15,
    unrealized_pl: 348.6,
    unrealized_plpc: 3.85,
    side: 'long',
  },
  {
    symbol: 'TSLA',
    qty: 15,
    market_value: 4890.75,
    avg_entry_price: 342.9,
    unrealized_pl: -172.25,
    unrealized_plpc: -3.4,
    side: 'long',
  },
  {
    symbol: 'NVDA',
    qty: 25,
    market_value: 3512.5,
    avg_entry_price: 132.4,
    unrealized_pl: 202.5,
    unrealized_plpc: 6.12,
    side: 'long',
  },
  {
    symbol: 'MSFT',
    qty: 10,
    market_value: 4231.0,
    avg_entry_price: 431.8,
    unrealized_pl: -87.0,
    unrealized_plpc: -2.02,
    side: 'short',
  },
]

export const mockSignals: SignalData = {
  AAPL: 'BUY',
  TSLA: 'HOLD',
  NVDA: 'BUY',
  MSFT: 'SELL',
  AMZN: 'HOLD',
  GOOGL: 'SELL',
}

export const mockEquityCurve: EquityPoint[] = [
  { date: '08/03', equity: 98500 },
  { date: '08/04', equity: 98720 },
  { date: '08/05', equity: 99010 },
  { date: '08/06', equity: 98860 },
  { date: '08/07', equity: 99340 },
  { date: '08/10', equity: 99800 },
  { date: '08/11', equity: 100120 },
  { date: '08/12', equity: 99950 },
  { date: '08/13', equity: 100430 },
  { date: '08/14', equity: 100890 },
  { date: '08/17', equity: 101200 },
  { date: '08/18', equity: 100980 },
  { date: '08/19', equity: 101540 },
  { date: '08/20', equity: 102010 },
  { date: '08/21', equity: 102460 },
  { date: '08/24', equity: 102190 },
  { date: '08/25', equity: 102980 },
  { date: '08/26', equity: 103410 },
  { date: '08/27', equity: 103890 },
  { date: '08/28', equity: 104250.32 },
]
