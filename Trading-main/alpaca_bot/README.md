# Alpaca Trading Bot

Automated buy/sell on Alpaca with an expectancy/Kelly risk framework. Defaults
to **paper trading** (fake money).

## Layout

```
TradingApp/
├─ main.py                    trading-loop entry point (loop + dashboard)
├─ run.py                     account CLI (status / trade / close / panic)
├─ config.yaml                symbols, strategy, risk limits, execution settings
├─ .env                       API keys (never committed)
├─ requirements.txt
├─ src/trading_bot/
│  ├─ config.py               loads .env + config.yaml into typed objects
│  ├─ indicators.py           SMA/EMA/RSI/ATR/VWAP/Bollinger/MACD from scratch
│  ├─ signals.py               Strategy interface + 5 strategies (see below)
│  ├─ risk.py                  trade log, expectancy tracker, Kelly sizing, hard limits
│  ├─ backtest.py               bar-by-bar backtest engine (Sharpe, drawdown, R-multiples)
│  ├─ broker.py                all Alpaca order/account/position calls
│  ├─ data.py                  historical bars + latest prices
│  ├─ trader.py                 live loop: bars -> signal -> risk -> bracket order
│  ├─ dashboard.py              terminal stats display
│  ├─ logger.py                 console + rotating file logs
│  │
│  │                            -- agent layer, all opt-in --
│  ├─ mcp_broker.py             same account over Alpaca's MCP server
│  ├─ attention.py              weighs every strategy's vote by regime fit
│  ├─ learning.py               per-strategy reliability from closed trades
│  ├─ retrieval.py              news + corporate actions for one ticker
│  ├─ agent.py                  LLM review: confirm / shrink / veto only
│  ├─ advisor_memory.py         scores the LLM's own past verdicts
│  ├─ scheduling.py             allocates position slots when they are scarce
│  ├─ allocation.py             splits capital between equities and options
│  ├─ jsonl.py                  shared append-only storage for the above
│  └─ options*.py               contracts, sizing and the options cycle
├─ scripts/
│  ├─ check_connection.py      smoke-test keys, trading API, data API
│  └─ backtest.py               CLI: backtest against real Alpaca history, save CSV+PNG
├─ backtests/                  backtest output (trade CSVs, equity curve PNGs)
├─ tests/                      pytest suite (no network needed)
└─ logs/, state/               rotating logs, trade history (state/trades.csv)
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env        # then paste your keys into .env
```

Get keys at <https://app.alpaca.markets> → **Home → API Keys → Generate**.
Paper and live accounts have **different** keys. The secret is shown only once —
if you lose it, regenerate the pair.

## Usage

```powershell
.venv\Scripts\python.exe scripts\check_connection.py       # always run this first
.venv\Scripts\python.exe scripts\backtest.py --days 180     # backtest the configured strategy
.venv\Scripts\python.exe main.py --once --dry-run           # one cycle, log signals only
.venv\Scripts\python.exe main.py --dry-run                  # continuous loop, no orders sent
.venv\Scripts\python.exe main.py                             # continuous loop, real paper orders

.venv\Scripts\python.exe run.py status                      # account, positions, orders
.venv\Scripts\python.exe run.py close AAPL                  # liquidate one position
.venv\Scripts\python.exe run.py panic                       # cancel all, close all
```

Run tests with `.venv\Scripts\python.exe -m pytest` (network-free, 597 tests, no keys).

## Strategies (`signals.py`)

| Strategy (`strategy:` in config.yaml) | Idea | Direction |
|---|---|---|
| `ema_rsi` | Oversold RSI reversal, filtered by EMA20 uptrend; exits on RSI>70 | long only |
| `vwap_mean_reversion` | Fade extreme moves away from intraday VWAP; mid-session only | long + short |
| `bollinger_squeeze` | Volatility contraction (new 20-bar bandwidth low) then a volume-confirmed breakout | long + short |
| `crypto_momentum` | Breakout momentum with a volatility filter, sized for crypto's fatter tails | long only |
| `opening_range_breakout` | Break of the first 15 minutes' range, volume-confirmed, stop at the far side of the range | long + short |

Every indicator (`indicators.py`) is causal — computed once over full history
and read bar-by-bar — so the backtest doesn't recompute indicators from
scratch on every bar (that would make a 6-month, 1-minute backtest O(n²) and
impractically slow).

## Risk framework (`risk.py`)

- **Expectancy** — tracked per strategy from closed trades in `state/trades.csv`:
  `Expectancy = win_rate*avg_win - loss_rate*avg_loss` (dollars), and in
  R-multiples: `Expectancy_R = win_rate*avg_R_winner - loss_rate*1.0`.
- **Kelly sizing** — `kelly = (win_rate*RR - loss_rate) / RR`, scaled by
  `kelly_multiplier` (0.5 = half-Kelly by default). Used only once
  `min_trades_for_kelly` trades exist; before that, a flat
  `fallback_position_pct` is used so sizing isn't guessing off zero data.
- **Breakeven win rate** — `1 / (1 + reward_risk_ratio)`; the dashboard shows
  the margin of safety (`win_rate - breakeven_win_rate`).
- **Hard limits** (always enforced, regardless of what Kelly suggests):
  `max_position_pct` (5% default), `max_open_positions` (3), `max_daily_loss_pct`
  (3%, halts new entries for the day), and an expectancy pause — a strategy
  stops taking new entries if its last `expectancy_window` trades are net
  negative in R.
- Stops/targets come from the strategy itself (ATR multiples, VWAP, or
  Bollinger band width) — there's no static stop-loss percentage in config.

## Backtesting (`backtest.py`, `scripts/backtest.py`)

Bar-by-bar replay, no lookahead (each decision uses only bars up to and
including that bar). Position sizing is Kelly-driven via the same
`RiskManager` the live trader uses, fed by the trade history the backtest
itself accumulates — so sizing starts flat and adapts exactly like it would
live. Slippage (0.02% default) and commission ($0, explicit so you can change
it) are modelled; spread and partial fills are not.

```powershell
.venv\Scripts\python.exe scripts\backtest.py --days 180
.venv\Scripts\python.exe scripts\backtest.py --symbols AAPL,MSFT --strategy vwap_mean_reversion
```

Saves `backtests/<strategy>_<symbol>_trades.csv` and `..._equity.png` per
symbol with at least one trade. Reports total return, Sharpe ratio, max
drawdown, win rate, expectancy (R), and the resulting Kelly fraction.

## Live trading (`trader.py`, `main.py`)

Each cycle: fetch bars → `strategy.generate_signal()` → `RiskManager.approve_entry()`
→ submit a **bracket order** (take-profit + stop-loss attached server-side at
Alpaca, so protection survives this process crashing). A strategy's own exit
signal (e.g. `ema_rsi`'s RSI>70 rule) closes the position manually; Alpaca
cancels the bracket's other leg automatically. `main.py`'s dashboard callback
prints account equity, live expectancy stats, and the last signal every cycle.
Ctrl+C triggers a graceful shutdown: cancel open orders, close all positions.

## Safety layers

| Guard | Where | Effect |
|---|---|---|
| Kelly sizing + hard `max_position_pct` cap | risk.py | caps size of any single position |
| `max_open_positions` | risk.py | caps concurrent holdings |
| `max_daily_loss_pct` | risk.py | halts new entries after a bad day |
| expectancy pause | risk.py | halts a strategy after `expectancy_window` net-negative trades |
| bracket orders | broker.py / trader.py | stop/target live server-side, survive a crash |
| duplicate-order check | trader.py | skips symbols with a pending order |
| `--dry-run` | CLI | logs intended orders, sends nothing |
| live confirmation | main.py / run.py | typing `I ACCEPT` required when `ALPACA_PAPER=false` |

## Writing your own strategy

1. Add a class to `src/trading_bot/signals.py` subclassing `Strategy`,
   implementing `precompute(df)`, `evaluate(df, ctx, i)`, `get_stop_loss(...)`,
   `get_take_profit(...)`. Set `allow_short = False` if it's long-only.
2. Register it in `signals.py`'s `REGISTRY`.
3. Point `strategy:` in config.yaml at its `name`, add any `strategy_params`.

Keep every indicator causal (rolling/ewm/cumulative, never referencing future
rows) — that's what lets `precompute()` be computed once and read bar-by-bar
in the backtest instead of recomputed per bar.

## Going live

Only after a strategy has run profitably on paper for a meaningful period:
set `ALPACA_PAPER=false` in `.env` **and** swap in your live keys. Start with
a small `max_position_pct` and a low `kelly_multiplier` (e.g. 0.25) until you
trust the live expectancy numbers. None of the three bundled strategies are a
proven edge — they're a template to replace, not trade as-is.
# Trading
