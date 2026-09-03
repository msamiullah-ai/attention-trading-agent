"""Live/paper trading loop: fetch bars -> signal -> risk check -> bracket order.

Stops and targets are attached as bracket orders at entry time, so protection
lives server-side at Alpaca and survives this process crashing. A strategy's
explicit exit signal (e.g. EmaRsiStrategy's RSI>70 rule) still closes the
position manually — the bracket's opposite leg is cancelled automatically by
Alpaca when `close_position` fills the other side.

Crypto symbols (e.g. "BTC/USD") are handled differently in two ways: Alpaca
doesn't support bracket orders for crypto at all, so stop/target are watched
and closed manually each cycle (see `_check_crypto_protection`); and crypto
trades 24/7, so it's never gated by `require_market_open` the way equities are.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

from .advisor_memory import AdvisorMemory
from .agent import build_advisor, review
from .attention import Verdict, attend
from .broker import AccountSnapshot, Broker, PositionSnapshot
from .config import AgentSettings, Config, Credentials, load_agent_settings
from .learning import StrategyMemory
from .correlation import compute_correlation_matrix, find_correlated_open_position
from .data import MarketData, is_crypto_symbol, lookback_days_for
from .logger import get_logger
from .regime import detect_regime, strategy_allows_regime
from .retrieval import Retriever
from .risk import RiskManager, TradeRecord
from .scheduling import Candidate, DeficitRoundRobin
from .signals import BUY, HOLD, REGISTRY, SELL, Strategy, build_strategy

log = get_logger(__name__)


@dataclass
class OpenTrade:
    symbol: str
    strategy: str
    side: str  # "long" | "short"
    qty: float
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_time: str
    # Which strategies voted, and how, at entry. Defaulted so a trade opened
    # before fusion existed still constructs; it simply teaches nothing.
    votes: dict[str, str] = field(default_factory=dict)


def _build_ensemble(primary: str, params: dict | None) -> dict[str, Strategy]:
    """One instance of every strategy, built once at startup.

    Only the configured strategy receives `strategy_params`. That dict is a flat
    set of tunings meant for one strategy, so handing an ema_rsi tuning to
    bollinger_squeeze is a TypeError if the names differ and, far worse, a
    silently mis-parameterised indicator if they happen to collide. The others
    run on their documented defaults.

    Built once rather than per cycle because construction derives min_bars and
    validates parameters, and repeating that per symbol per cycle is pure waste.
    """
    ensemble: dict[str, Strategy] = {}
    for name in REGISTRY:
        try:
            ensemble[name] = build_strategy(name, params if name == primary else None)
        except Exception as exc:  # noqa: BLE001 - one bad member must not stop the rest
            log.warning("ensemble: skipping %s (%s)", name, exc)
    return ensemble


class Trader:
    def __init__(
        self,
        config: Config,
        creds: Credentials,
        dry_run: bool = False,
        broker: Broker | None = None,
        data: MarketData | None = None,
        risk: RiskManager | None = None,
        strategy: Strategy | None = None,
        advisor=None,
        fusion: bool | None = None,
        fusion_temperature: float | None = None,
        settings: AgentSettings | None = None,
        memory: StrategyMemory | None = None,
        advisor_memory: "AdvisorMemory | None" = None,
        budget: float | None = None,
    ):
        # broker/data/risk/strategy are injectable so tests can pass fakes
        # instead of hitting the real Alpaca API.
        self.cfg = config
        self.dry_run = dry_run
        self.broker = broker if broker is not None else Broker(creds)
        self.data = data if data is not None else MarketData(creds, feed=config.data.feed)
        self.risk = risk if risk is not None else RiskManager(config.risk)
        # Agent layer settings come from one place (config.load_agent_settings),
        # not from os.environ reads scattered through this file. Explicit
        # constructor arguments still win, so tests inject rather than patch
        # the environment.
        self.settings = settings if settings is not None else load_agent_settings()
        # Advisory layer, off unless asked for. Kept out of config.yaml on
        # purpose: it lives in .env, so a deployment can turn it on without a
        # schema change and every existing config file keeps working unchanged.
        # Absent advisor == today's rules-only behaviour, exactly.
        self.advisor = advisor if advisor is not None else build_advisor(self.settings)
        # Retrieval needs the MCP transport; the REST Broker has no `_call`.
        self.retriever = Retriever(self.broker) if (
            self.advisor is not None and hasattr(self.broker, "_call")) else None
        self.strategy: Strategy = strategy or build_strategy(config.strategy, config.strategy_params)
        # Signal fusion, off unless asked for. When on, all applicable strategies
        # vote and `attention.attend` weighs them by regime fit; when off, the
        # configured strategy decides alone exactly as it does today.
        #
        # The primary strategy stays load-bearing either way: the ensemble
        # decides DIRECTION, the primary still sets stop and target. That split
        # is not arbitrary -- get_stop_loss is a method on a specific strategy
        # and EmaRsiStrategy asserts side == "long" inside it, so levels cannot
        # be sourced from a vote without picking whose levels they are.
        if fusion is None:
            fusion = self.settings.fusion
        self.fusion_temperature = (fusion_temperature if fusion_temperature is not None
                                   else self.settings.fusion_temperature)
        self.ensemble: dict[str, Strategy] = (
            _build_ensemble(self.strategy.name, config.strategy_params) if fusion else {})
        # Normalised once at startup: config gives lists, the check wants
        # frozensets, and rebuilding them per symbol per cycle is waste.
        self._extra_correlated_pairs = {
            frozenset(p) for p in (config.risk.correlated_pairs or []) if len(p) == 2}
        # Closed-trade outcomes per strategy, feeding back into the vote.
        # Injectable so tests never touch the on-disk record.
        self.memory = memory if memory is not None else StrategyMemory()
        # The advisor's own record. Only built when there is an advisor to
        # score -- an empty ledger for a layer that never runs is just a file.
        self.advisor_memory = (advisor_memory if advisor_memory is not None
                               else (AdvisorMemory() if self.advisor is not None else None))
        # Slot allocation. None = enter on sight in config order, which is the
        # behaviour every earlier version had.
        # Ceiling on capital this side may deploy, from `allocation.decide`.
        # None means the whole account, which is how every earlier version
        # behaved. It can only ever reduce what sizing would otherwise allow.
        self.budget = budget
        self.scheduler = (DeficitRoundRobin()
                          if config.execution.scheduler == "drr" else None)
        self.last_verdict: dict[str, Verdict] = {}
        self.open_trades: dict[str, OpenTrade] = {}
        self.last_signal: dict[str, str] = {}
        self.last_signal_symbol: str | None = None
        self.last_signal_time: str | None = None
        self._cached_account: AccountSnapshot | None = None
        self._cached_positions: dict[str, PositionSnapshot] = {}
        log.info(
            "Trader ready | strategy=%s symbols=%s dry_run=%s mode=%s",
            self.strategy.name, ",".join(config.symbols), dry_run, creds.mode,
        )

    def _collect_votes(self, symbol: str, df) -> dict[str, str]:
        """Ask every applicable strategy for its read on this bar.

        A strategy that cannot answer ABSTAINS -- it is left out of the dict
        entirely rather than recorded as HOLD. The difference is not cosmetic:
        HOLD is a vote with weight that drags the fused score toward the
        deadband, so logging "no opinion" as HOLD would let a strategy that
        never even ran suppress a genuine signal from the ones that did.
        """
        votes: dict[str, str] = {}
        for name, strat in self.ensemble.items():
            # crypto_momentum's volatility percentile thresholds are calibrated
            # for crypto's fatter tails (see its docstring). Firing it at an
            # equity does not error, it just applies the wrong thresholds
            # confidently, which is the worse failure of the two.
            if name == "crypto_momentum" and not is_crypto_symbol(symbol):
                continue
            if len(df) < strat.min_bars:
                continue
            try:
                votes[name] = strat.generate_signal(df)
            except Exception as exc:  # noqa: BLE001 - abstain, never crash the cycle
                log.debug("ensemble: %s abstained on %s (%s)", name, symbol, exc)
        return votes

    # ------------------------------------------------------------ single pass

    def run_cycle(self) -> None:
        self._cached_account = self.broker.get_account()
        gate = self.risk.check_account(self._cached_account)
        if not gate.approved:
            log.warning("RISK HALT: %s", gate.reason)
            return

        equity_market_open = self.broker.is_market_open()
        equity_gated = self.cfg.execution.require_market_open and not equity_market_open
        if equity_gated:
            log.info(
                "Equity market closed (next open: %s) - still evaluating any crypto symbols.",
                self.broker.next_market_open(),
            )

        self._cached_positions = self.broker.get_positions()
        positions = self._cached_positions

        # A trade we were tracking is no longer open -> the bracket's TP/SL
        # (or our own manual close) filled since the last cycle. Reconcile it.
        for symbol in list(self.open_trades):
            if symbol not in positions:
                self._finalize_trade(symbol)

        lookback_days = lookback_days_for(self.cfg.timeframe, self.cfg.lookback_bars)
        bars = self.data.get_bars(self.cfg.symbols, timeframe=self.cfg.timeframe, lookback_days=lookback_days)
        pending: list[Candidate] = []

        for symbol in self.cfg.symbols:
            crypto = is_crypto_symbol(symbol)
            if not crypto and equity_gated:
                continue  # equities only trade during regular market hours

            df = bars.get(symbol)
            if df is None or len(df) < self.strategy.min_bars:
                continue

            position = positions.get(symbol)

            # No bracket order exists for crypto -- check stop/target ourselves
            # before doing anything else with this symbol this cycle.
            # Crypto never has a bracket; a fractional equity does not either,
            # because Alpaca rejects an advanced order_class on both. Checking
            # the held quantity is what makes the protection follow the position
            # rather than the asset class -- a whole-share equity keeps its
            # server-side bracket and is correctly skipped here.
            unbracketed = crypto or (
                position is not None and float(position.qty) != int(float(position.qty)))
            if unbracketed and position is not None and self._check_crypto_protection(
                    symbol, df, position):
                continue

            if self.ensemble:
                # The learned half of the weighting: regime fit says who SHOULD
                # be right here, the record says who HAS been. A strategy with a
                # poor recent record is quieted, never silenced.
                verdict = attend(self._collect_votes(symbol, df), regime=detect_regime(df),
                                 temperature=self.fusion_temperature,
                                 reliability=self.memory.reliability())
                signal = verdict.action
                if len(self.last_verdict) > 1000:
                    self.last_verdict.clear()
                self.last_verdict[symbol] = verdict
                if signal != HOLD:
                    log.info("FUSED %s | %s", symbol, verdict.explain())
            else:
                signal = self.strategy.generate_signal(df)
            if len(self.last_signal) > 1000:
                self.last_signal.clear()
            self.last_signal[symbol] = signal
            self.last_signal_symbol = symbol
            self.last_signal_time = datetime.now(timezone.utc).isoformat()
            if signal != "HOLD":
                log.info("SIGNAL %s %s", symbol, signal)

            if position is not None:
                self._maybe_exit(symbol, signal, position)
            elif self.scheduler is None:
                self._maybe_enter(symbol, signal, df, self._cached_account, positions, bars)
            elif signal in (BUY, SELL):
                # Deferred rather than entered. Entering here hands the scarce
                # slots to whichever symbols appear earliest in config.yaml,
                # which is why a strong signal at index 950 could never beat a
                # weak one at index 100. Collect now, allocate once the whole
                # universe has been seen.
                verdict = self.last_verdict.get(symbol)
                pending.append(Candidate(
                    symbol=symbol, signal=signal,
                    # Under fusion, agreement is the share of weight behind the
                    # winning direction. Without it, ask the strategy how strong
                    # its own signal was -- otherwise every candidate ties at
                    # the neutral default, and "best of 124" collapses into
                    # alphabetical order, which is just file-order bias wearing
                    # a different hat.
                    score=(verdict.agreement if verdict is not None
                           else self.strategy.strength(df)),
                    payload={"df": df},
                ))

        # Settle any advice old enough to judge, using the closes already in
        # hand. The price kept trading whether or not we took the trade, so a
        # veto has a measurable outcome even though the position never existed.
        if self.advisor_memory is not None:
            closes = {sym: float(df["close"].iloc[-1])
                      for sym, df in bars.items()
                      if df is not None and len(df)}
            self.advisor_memory.settle(closes)

        if self.scheduler is not None and pending:
            free = max(0, self.cfg.risk.max_open_positions - len(positions))
            for winner in self.scheduler.select(pending, free):
                self._maybe_enter(winner.symbol, winner.signal, winner.payload["df"],
                                  self._cached_account, positions, bars)

    def _check_crypto_protection(self, symbol: str, df, position: PositionSnapshot) -> bool:
        """Watch stop/target for positions that have no broker-side bracket.

        Alpaca rejects an advanced order_class on crypto AND on any fractional
        quantity, so neither kind carries server-side protection. Both are
        watched here instead. Named for crypto because that was the original
        case; it now covers every unbracketed position, which is the set that
        actually needs it.

        Returns True if a close was submitted (caller skips this symbol)."""
        trade = self.open_trades.get(symbol)
        if trade is None:
            return False  # e.g. a position opened manually outside the bot

        price = float(df["close"].iloc[-1])
        is_long = trade.side == "long"
        hit_stop = (is_long and price <= trade.stop_loss) or (not is_long and price >= trade.stop_loss)
        hit_target = (is_long and price >= trade.take_profit) or (not is_long and price <= trade.take_profit)
        if not (hit_stop or hit_target):
            return False

        reason = "stop-loss" if hit_stop else "take-profit"
        if self.dry_run:
            log.info("[DRY RUN] would CLOSE %s x%s (%s hit @ %.2f)", symbol, position.qty, reason, price)
            return False

        log.info("CRYPTO %s hit for %s @ %.2f - closing manually", reason.upper(), symbol, price)
        self.broker.close_position(symbol)
        return True

    # ---------------------------------------------------------------- entries

    def _maybe_enter(
        self,
        symbol: str,
        signal: str,
        df,
        account: AccountSnapshot,
        positions: dict[str, PositionSnapshot],
        bars: dict,
    ) -> None:
        wants_long = signal == BUY
        # A strategy may allow shorting; the ACCOUNT may not. Checking here
        # rather than discovering it from a 403 at submission means the whole
        # pipeline below -- correlation, sizing, the LLM call -- is not spent
        # on a trade that was never placeable.
        wants_short = signal == SELL and self.strategy.allow_short
        if wants_short and not self.broker.shorting_enabled():
            log.info("SKIP %s SELL - account is not enabled for shorting", symbol)
            return
        if not (wants_long or wants_short):
            return
        if self.broker.has_open_order(symbol):
            log.info("SKIP %s - open order already pending", symbol)
            return

        # The boolean regime gate applies only when ONE strategy is deciding.
        #
        # Under fusion it is deliberately skipped, and skipping it is the point
        # rather than an oversight: `attention.suitability` is derived from this
        # same STRATEGY_REGIMES table and has already applied it continuously in
        # the weights. Running the boolean form afterwards would re-impose the
        # all-or-nothing cut that fusion exists to replace -- a BUY carried by
        # three strategies that suit the regime would be discarded because the
        # configured primary happens not to suit it, which is exactly backwards.
        regime = detect_regime(df)
        if not self.ensemble and not strategy_allows_regime(self.strategy.name, regime):
            log.info("SKIP %s %s - regime %s doesn't suit %s", symbol, signal, regime, self.strategy.name)
            return

        if positions:
            # Cheap on purpose: only correlate the candidate against symbols
            # actually open right now, not the full universe every cycle.
            relevant_bars = {symbol: df, **{s: bars[s] for s in positions if s in bars}}
            corr_matrix = compute_correlation_matrix(relevant_bars)
            blocker = find_correlated_open_position(
                symbol, set(positions), corr_matrix,
                threshold=self.cfg.risk.correlation_threshold,
                extra_pairs=self._extra_correlated_pairs)
            if blocker:
                log.info("SKIP %s %s - correlated with open position %s", symbol, signal, blocker)
                return

        price = self.data.get_last_price(symbol) or float(df["close"].iloc[-1])
        # Fractionability is an asset fact, cached in the broker -- a symbol
        # that cannot be split must still be sized in whole shares or skipped.
        fractionable = self.broker.is_fractionable(symbol)
        # Apply the allocation budget by capping buying power. Sizing already
        # takes min(equity * fraction, buying_power), so this needs no change
        # to the risk model -- and subtracting what is already deployed makes
        # the cap CUMULATIVE. Capping each position individually would let
        # three positions deploy three budgets.
        account = self._within_budget(account, positions)
        decision = self.risk.approve_entry(symbol, price, self.strategy.name, account,
                                           positions, fractionable=fractionable)
        if not decision.approved:
            log.info("REJECTED %s %s - %s", symbol, signal, decision.reason)
            return

        # Advisory review: the last look, and a subtractive one. Everything
        # above has already approved this trade; the advisor can only veto it or
        # cut its size, never enlarge it and never pick a different one. With no
        # advisor configured the whole block is skipped and the rules decision
        # stands -- see agent.py on why an outage here must not halt trading.
        qty = decision.qty
        if self.advisor is not None:
            context = self.retriever.context(symbol).to_prompt() if self.retriever else ""
            side_word = "buy" if wants_long else "sell"
            history = (self.advisor_memory.prompt_block(symbol)
                       if self.advisor_memory else "")
            advice = review(self.advisor, symbol, side_word,
                            qty, price, context, history=history)
            if self.advisor_memory is not None:
                self.advisor_memory.record(symbol, side_word, advice.action,
                                           advice.reason, price)
            # Logged on EVERY outcome, confirms included. review() only speaks up
            # on veto and shrink, which makes "the advisor approved this" and
            # "the advisor never ran" identical in the log -- and those are very
            # different facts when you are deciding whether to trust a fill.
            log.info("ADVISOR %s | %s", symbol, advice.summary())
            if advice.blocks:
                log.info("VETOED %s %s - %s", symbol, signal, advice.reason)
                return
            qty = advice.apply_to(qty)
            if qty <= 0:
                log.info("SKIP %s %s - advisor cut size to zero", symbol, signal)
                return

        side = "long" if wants_long else "short"
        stop = self.strategy.get_stop_loss(price, df, side)
        target = self.strategy.get_take_profit(price, df, side)

        if self.dry_run:
            log.info(
                "[DRY RUN] would %s %s x%s @ ~$%.2f (stop=%.2f target=%.2f)",
                "BUY" if wants_long else "SELL", symbol, qty, price, stop, target,
            )
            # Counted even in a dry run, so the run reports the trades that
            # would really happen rather than every one it merely considered.
            self._reserve_position(positions, symbol, qty, price, side)
            return

        order_side = "buy" if wants_long else "sell"
        try:
            self.broker.submit_order(
                symbol, qty, order_side,
                order_type=self.cfg.execution.order_type,
                time_in_force=self.cfg.execution.time_in_force,
                take_profit_price=target,
                stop_loss_price=stop,
            )
        except Exception as exc:  # noqa: BLE001
            # One refused order must not end the cycle. The broker can reject
            # for reasons no local check anticipates -- a halted symbol, a
            # borrow that vanished, a per-asset restriction -- and aborting
            # would abandon every remaining candidate and leave the positions
            # already opened this cycle untracked.
            log.warning("ORDER REJECTED %s %s x%s: %s", order_side, symbol, qty, exc)
            positions.pop(symbol, None)      # release the slot we reserved
            return
        # Claim the slot immediately. `positions` is fetched once per cycle and
        # never refreshed inside it, so without this every symbol evaluated
        # afterwards still sees the count from before any of them filled --
        # max_open_positions is silently ignored and one cycle can submit far
        # more orders than the limit allows. It stayed hidden while a single
        # strategy rarely produced two tradeable signals in the same cycle.
        self._reserve_position(positions, symbol, qty, price, side)

        entry_price = self._resolve_fill_price(symbol, fallback=price)
        self.open_trades[symbol] = OpenTrade(
            symbol=symbol, strategy=self.strategy.name, side=side, qty=qty,
            entry_price=entry_price, stop_loss=stop, take_profit=target,
            entry_time=datetime.now(timezone.utc).isoformat(),
            votes=dict(self.last_verdict[symbol].votes) if symbol in self.last_verdict else {},
        )

    def _within_budget(self, account: AccountSnapshot,
                       positions: dict[str, PositionSnapshot]) -> AccountSnapshot:
        """The account as this side is allowed to see it.

        Returns the original untouched when no budget is set, so the default
        path is byte-identical to before. Otherwise buying power is reduced to
        what remains of the budget after existing exposure -- including the
        placeholder positions reserved earlier this cycle, so a cycle cannot
        spend the budget several times over.
        """
        if self.budget is None:
            return account
        deployed = sum(abs(float(p.market_value)) for p in positions.values())
        remaining = max(0.0, self.budget - deployed)
        if remaining >= account.buying_power:
            return account
        return replace(account, buying_power=remaining)

    def _reserve_position(self, positions: dict[str, PositionSnapshot], symbol: str,
                          qty: float, price: float, side: str) -> None:
        """Record an in-flight entry against this cycle's position count.

        A placeholder rather than a fetched snapshot: the fill may not have
        settled yet, and re-fetching positions after every order would add a
        round trip per candidate. Only the COUNT and the symbol matter here --
        `approve_entry` checks membership and length, nothing else — and both
        are correct immediately. The real snapshot replaces it next cycle.
        """
        positions[symbol] = PositionSnapshot(
            symbol=symbol, qty=qty, market_value=qty * price,
            avg_entry_price=price, unrealized_pl=0.0, unrealized_plpc=0.0,
            side=side,
        )

    def _resolve_fill_price(self, symbol: str, fallback: float, timeout_s: float = 5.0) -> float:
        """Paper-trading market orders usually fill within ~1s; poll briefly
        for the real avg_entry_price rather than trusting the pre-fill quote.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            position = self.broker.get_position(symbol)
            if position is not None:
                return position.avg_entry_price
            time.sleep(0.5)
        log.warning("Could not confirm fill price for %s within %.0fs, using %.2f", symbol, timeout_s, fallback)
        return fallback

    # ----------------------------------------------------------------- exits

    def _maybe_exit(self, symbol: str, signal: str, position: PositionSnapshot) -> None:
        is_long = position.side == "long"
        exit_wanted = (is_long and signal == SELL) or (not is_long and signal == BUY)
        if not exit_wanted:
            return  # let the bracket order's TP/SL manage it

        if self.dry_run:
            log.info("[DRY RUN] would CLOSE %s x%s (strategy exit signal)", symbol, position.qty)
            return

        log.info("STRATEGY EXIT %s - closing position", symbol)
        self.broker.close_position(symbol)

    def _finalize_trade(self, symbol: str) -> None:
        trade = self.open_trades.pop(symbol)
        exit_price, exit_time = self._lookup_exit_fill(symbol, trade)

        if trade.side == "long":
            pnl = (exit_price - trade.entry_price) * trade.qty
            risk_per_share = trade.entry_price - trade.stop_loss
        else:
            pnl = (trade.entry_price - exit_price) * trade.qty
            risk_per_share = trade.stop_loss - trade.entry_price

        r_multiple = pnl / (risk_per_share * trade.qty) if risk_per_share > 0 else 0.0

        # Settle the vote before recording the trade: each strategy that called
        # this direction gets credited or debited, which is what makes the next
        # cycle's weighting different from this one's.
        self.memory.record(trade.votes, BUY if trade.side == "long" else SELL,
                           symbol, r_multiple)

        self.risk.record_trade(
            TradeRecord(
                symbol=symbol, strategy=trade.strategy, side=trade.side, qty=trade.qty,
                entry_price=trade.entry_price, exit_price=exit_price, stop_loss=trade.stop_loss,
                take_profit=trade.take_profit,
                entry_time=trade.entry_time, exit_time=exit_time, pnl=pnl, r_multiple=r_multiple,
            )
        )

    def _lookup_exit_fill(self, symbol: str, trade: OpenTrade) -> tuple[float, str]:
        """Best-effort exit price/time from Alpaca's own fill record."""
        closing_side = "sell" if trade.side == "long" else "buy"
        # Alpaca's filled_at is always tz-aware, so a naive entry_time (an
        # older state file, a hand-written one) would make the comparison
        # below raise instead of just skipping the order. Assume UTC.
        entry_dt = datetime.fromisoformat(trade.entry_time)
        if entry_dt.tzinfo is None:
            entry_dt = entry_dt.replace(tzinfo=timezone.utc)
        for order in self.broker.get_closed_orders(symbol, limit=20):
            side = order.side.value if hasattr(order.side, "value") else order.side
            if side != closing_side or order.filled_avg_price is None:
                continue
            if order.filled_at and order.filled_at < entry_dt:
                continue
            filled_at = order.filled_at.isoformat() if order.filled_at else datetime.now(timezone.utc).isoformat()
            return float(order.filled_avg_price), filled_at

        fallback_price = self.data.get_last_price(symbol) or trade.entry_price
        log.warning("Could not find closing fill for %s, using last price %.2f", symbol, fallback_price)
        return fallback_price, datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------- loop

    def run_forever(self, interval_seconds: int | None = None, on_cycle=None) -> None:
        """`on_cycle`, if given, runs after every cycle (e.g. dashboard.print_dashboard)."""
        interval = interval_seconds or self.cfg.engine.poll_interval_seconds
        log.info("Starting loop, interval %ds. Ctrl+C for a graceful shutdown.", interval)
        try:
            while True:
                try:
                    self.run_cycle()
                    if on_cycle is not None:
                        on_cycle()
                except (ConnectionError, TimeoutError, OSError) as e:
                    log.warning("Cycle failed (transient): %s", e)
                except Exception:
                    log.exception("Cycle failed; retrying after interval")
                time.sleep(interval)
        except KeyboardInterrupt:
            self.shutdown()

    def shutdown(self) -> None:
        log.info("Shutting down: cancelling open orders and closing all positions.")
        if self.dry_run:
            log.info("[DRY RUN] would cancel orders and close all positions")
            return
        try:
            self.broker.cancel_all_orders()
            self.broker.close_all_positions(cancel_orders=True)
        except Exception:  # noqa: BLE001 - shutdown must not raise past this point
            log.exception("Error while closing positions during shutdown")
