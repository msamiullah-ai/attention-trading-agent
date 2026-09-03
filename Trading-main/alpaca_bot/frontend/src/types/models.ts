// Shared data contracts between the frontend and the (future) backend API.
// Single source of truth — do not redeclare these in component/hook files.
//
// AccountSnapshot and PositionSnapshot field names/shapes are checked
// against src/trading_bot/broker.py's dataclasses of the same name.
// ExpectancyStats is checked against src/trading_bot/risk.py.

export interface AccountSnapshot {
  equity?: number
  cash?: number
  buying_power?: number
  /**
   * Fraction, not a percentage — e.g. 0.0142 means +1.42%.
   * Matches broker.AccountSnapshot.daily_pnl_pct, a computed property
   * ((equity - last_equity) / last_equity), which dashboard.py formats
   * with a `%` format specifier. Multiply by 100 when rendering.
   */
  daily_pnl_pct?: number
  /**
   * Not present on broker.AccountSnapshot — that dataclass only has
   * equity/cash/buying_power/last_equity/blocked. "PAPER" vs "LIVE" lives
   * on Credentials.mode (config.py) instead. Whichever API layer gets
   * built will need to merge the two; this field assumes it does.
   */
  mode?: string
  blocked?: boolean
}

export interface PositionSnapshot {
  symbol: string
  qty: number
  market_value: number
  avg_entry_price: number
  unrealized_pl: number
  unrealized_plpc: number
  side: string
}

export interface SignalData {
  [symbol: string]: 'BUY' | 'SELL' | 'HOLD'
}

export interface EquityPoint {
  date: string
  equity: number
}

/**
 * Matches risk.ExpectancyStats exactly. Note the two different expectancy
 * fields have different units — neither is a percentage:
 *   - expectancy: dollars per trade
 *   - expectancy_r: R-multiples per trade
 * win_rate / loss_rate / breakeven_win_rate / margin_of_safety are
 * fractions (0-1); multiply by 100 to render as a percentage.
 */
export interface ExpectancyStats {
  n: number
  win_rate: number
  loss_rate: number
  avg_win: number // dollars
  avg_loss: number // dollars, positive magnitude
  avg_r_winner: number // R-multiple
  reward_risk_ratio: number
  expectancy: number // dollars per trade — NOT a percentage
  expectancy_r: number // R-multiples per trade — NOT a percentage
  breakeven_win_rate: number
  margin_of_safety: number // win_rate - breakeven_win_rate
}
