# Attention Trading Agent

An autonomous trading agent built for the Alpaca AI Trading Agents Hackathon.
Trades US equities and options through Alpaca's MCP server.

## Idea

Signals are weighted rather than chosen. Several strategies each read the same
bar and vote; an attention layer weighs those votes by how well each strategy
suits the current market regime and how it has actually been performing, then
retrieved news and corporate-action context is passed to an LLM for a final
review. Everything survives a hard-coded risk layer — position sizing,
correlation limits, exposure caps, daily loss limits — before an order is sent.

## What is implemented, and what it is not

The layers below all exist and are tested. Where a claim is weaker than it
sounds, it says so here rather than in a footnote.

| layer | file | what it does |
|---|---|---|
| strategies | `signals.py` | five: EMA/RSI, VWAP reversion, Bollinger squeeze, crypto momentum, opening-range breakout |
| attention | `attention.py` | softmax-weighted fusion of strategy votes |
| learning | `learning.py` | per-strategy reliability from closed trades |
| retrieval | `retrieval.py` | news + corporate actions per candidate |
| LLM review | `agent.py` | confirm / shrink / veto on an already-approved trade |
| advisor memory | `advisor_memory.py` | scores the LLM's own past verdicts |
| scheduling | `scheduling.py` | allocates scarce position slots fairly |
| allocation | `allocation.py` | splits capital between equities and options |
| risk | `risk.py` | Kelly sizing, expectancy pause, hard limits |
| options | `options*.py` | cash-secured puts and covered calls |

**The attention layer is not trained.** It is softmax-weighted signal fusion:
the query/key/value framing maps onto it honestly — regime as query, per-strategy
suitability as keys, votes as values — and the arithmetic is the same arithmetic.
But the weights are hand-derived from the regime table, not learned from data,
because there is no labelled dataset here to learn them from. "We weight signals
by a scoring function" is the accurate claim.

**The LLM can only subtract.** It cannot propose a trade, pick a symbol or
enlarge one — only veto or shrink something the rules already approved. The
failure mode of a language model is confident fabrication, and under veto-only
authority the worst a hallucination can do is decline a good trade rather than
open a bad one.

**RAG has no vector store.** The corpus is a handful of headlines about one
ticker, fetched fresh each cycle. Embedding six documents to rank six documents
would add a dependency and a failure mode to answer what the news endpoint
already answered by filtering on symbol and date.

**Earnings are not enforceable.** Alpaca's corporate-actions endpoint accepts
only dividend, merger, spinoff and split — passing `earnings` returns 422. There
is no earnings calendar in this API, so earnings reach the model as a news
headline judged in prose, not as a gate.

**No strategy here is a proven edge.** Backtested on 90 days of 1-minute bars,
four of the five showed negative expectancy over their first twenty trades and
were paused by the risk layer, which is the risk layer working. Opening-range
breakout was the only one with a positive Sharpe and a real sample, and its
margin was thin. Treat this as a well-tested machine for making decisions whose
quality is unproven.

## Layout

```
Trading-main/alpaca_bot/
├─ main.py, main_options.py   equity loop and options overlay
├─ run.py                     account CLI: status, close, panic
├─ api.py, api_types.py       read-only FastAPI layer the dashboard reads
├─ config.yaml, options.yaml  symbols, strategy, risk and options limits
├─ src/trading_bot/           indicators, signals, agent layer, risk, brokers
├─ frontend/                  React + Vite dashboard
├─ tools/                     preflight checks against a live account
├─ scripts/                   connection check, backtests
└─ tests/                     pytest suite, network-free
```

Two broker transports share one interface: `broker.py` talks to Alpaca's REST
API, `mcp_broker.py` reaches the same account through Alpaca's MCP server and is
a drop-in for it. MCP additionally exposes news and corporate actions, which the
REST client has no endpoint for — with `transport: rest` the LLM reviewer sees
the trade and nothing else.

## Running the whole stack

Three processes. The bot trades, the API exposes what it did, the dashboard
draws it. The API is read-only and never submits an order, so it is safe to
start and stop independently of the bot.

```powershell
cd Trading-main\alpaca_bot

python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env        # then paste your Alpaca paper keys into .env

.venv\Scripts\python.exe scripts\check_connection.py   # verify keys first
.venv\Scripts\python.exe main.py --dry-run             # terminal 1: the bot
.venv\Scripts\python.exe api.py                        # terminal 2: API on :8000
```

```powershell
cd Trading-main\alpaca_bot\frontend
npm install
npm run dev                   # terminal 3: dashboard on :3000
```

The dev server proxies `/api` and `/health` to `127.0.0.1:8000` (see
`vite.config.ts`). Without the API running the dashboard still loads — it badges
itself "Backend disconnected" and falls back to the demo data in
`src/api/mockData.ts`, so the UI stays explorable. If you see that badge while
the API *is* up, the proxy target is the first thing to check.

### Turning the agent layer on and off

Every layer above the original strategies is opt-in and defaults to off in
code, so setting all of them to `none` reproduces the original behaviour
exactly. Turn them on one at a time.

In `config.yaml`. The values checked in are the ones this account has been
running, not the conservative defaults — set them back to the first option in
each list to get the original behaviour:

```yaml
strategy: opening_range_breakout   # or ema_rsi, the original

execution:
  transport: mcp       # rest | mcp        mcp also unlocks news retrieval
  scheduler: drr       # none | drr        who gets a slot when they are scarce
  allocator: dynamic   # none | dynamic    equities vs options split
```

In `.env`:

```ini
TRADING_BOT_FUSION=off       # on = all strategies vote, weighted by regime
TRADING_BOT_ADVISOR=none     # none | openai | featherless | anthropic | ollama | ...
```

The reviewer works with any provider speaking OpenAI's `/v1/chat/completions`;
set `LLM_MODEL` and `LLM_API_KEY` and the endpoint is derived from the provider
name. A missing key or an unreachable model degrades to rules-only rather than
halting the bot — the reviewer holds no permission the rule layer had not
already granted, so its absence is recorded as absence and the cycle continues.

### Checking it before trusting it

```powershell
.venv\Scripts\python.exe tools\check_agent.py      # bars, model, no orders
.venv\Scripts\python.exe main_options.py --check   # MCP tools + options level
```

`check_agent.py` runs the real strategies on real bars and puts the real model
through five hypothetical trades, printing the reasoning rather than asserting
on it — "is this judgement any good" is a question for a human, not an assert.
One of the five is deliberately vague macro commentary the reviewer should
refuse to veto on.

## Tests

```powershell
cd Trading-main\alpaca_bot
.venv\Scripts\python.exe -m pytest -q     # 597 tests, no network, no keys

cd frontend
npm run typecheck
npm run build
```

Both run in CI on every push (`.github/workflows/ci.yml`).

The suite is hermetic: `tests/conftest.py` neutralises the agent configuration
and redirects `state/` to a temp directory, so a machine with a live `.env` runs
the same tests as a clean checkout and no test can write to the real ledgers.

## Safety

Paper trading by default (`ALPACA_PAPER=true`). Going live requires both
flipping that flag *and* swapping in live keys, and the CLI demands a typed
confirmation before it will trade real money.

Two things worth knowing before pointing this at an account:

**Fractional positions have no broker-side stop.** Alpaca rejects bracket orders
on fractional quantities, so those positions are protected by an in-process
stop/target watch — which only runs while the bot is running. Whole-share
positions keep their server-side bracket and are safe across a restart.

**The expectancy pause will stop you.** After twenty trades with negative
expectancy the risk layer refuses new entries. That is the intended behaviour,
not a fault, and on this data every strategy reached it.
