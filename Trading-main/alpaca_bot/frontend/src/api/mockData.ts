import type { AccountSnapshot } from '../components/displays/AccountSummary'
import type { PositionSnapshot } from '../components/displays/PositionsGrid'
import type { SignalData } from '../components/displays/StrategySignalPanel'
import type { EquityPoint } from '../hooks/useEquityCurve'
import type { RiskMetricsData } from '../hooks/useRiskMetrics'

// Sample data shown when the Python backend isn't reachable, so the
// dashboard is fully explorable (design, layout, states) without it running.

export const mockAccount: AccountSnapshot = {
  equity: 104250.32,
  cash: 18320.11,
  buying_power: 36640.22,
  daily_pnl_pct: 1.42,
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

// /api/equity-curve returns CUMULATIVE REALIZED P/L over closed trades
// starting from zero (api_types.py), not account equity. The mock matches
// that shape so switching the backend on doesn't rescale the chart.
export const mockEquityCurve: EquityPoint[] = [
  { date: '08/03', equity: 0 },
  { date: '08/04', equity: 220 },
  { date: '08/05', equity: 510 },
  { date: '08/06', equity: 360 },
  { date: '08/07', equity: 840 },
  { date: '08/10', equity: 1300 },
  { date: '08/11', equity: 1620 },
  { date: '08/12', equity: 1450 },
  { date: '08/13', equity: 1930 },
  { date: '08/14', equity: 2390 },
  { date: '08/17', equity: 2700 },
  { date: '08/18', equity: 2480 },
  { date: '08/19', equity: 3040 },
  { date: '08/20', equity: 3510 },
  { date: '08/21', equity: 3960 },
  { date: '08/24', equity: 3690 },
  { date: '08/25', equity: 4480 },
  { date: '08/26', equity: 4910 },
  { date: '08/27', equity: 5390 },
  { date: '08/28', equity: 5750.32 },
]

// Percent fields in percent, expectancy in R — same contract as the API.
export const mockRiskMetrics: RiskMetricsData = {
  n: 42,
  win_rate: 54.76,
  loss_rate: 45.24,
  expectancy: 0.184,
  breakeven_win_rate: 40.0,
  margin_of_safety: 14.76,
  kelly_position_pct: 3.2,
  strategy: 'ema_rsi',
  window: 50,
}
