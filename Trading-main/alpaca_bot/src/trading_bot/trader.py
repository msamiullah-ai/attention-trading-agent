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
from dataclasses import dataclass
from datetime import datetime, timezone

from .broker import AccountSnapshot, Broker, PositionSnapshot
from .config import Config, Credentials
from .correlation import compute_correlation_matrix, find_correlated_open_position
from .data import MarketData, is_crypto_symbol, lookback_days_for
from .logger import get_logger
from .regime import detect_regime, strategy_allows_regime
from .risk import RiskManager, TradeRecord
from .signals import BUY, SELL, Strategy, build_strategy

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
    ):
        # broker/data/risk/strategy are injectable so tests can pass fakes
        # instead of hitting the real Alpaca API.
        self.cfg = config
        self.dry_run = dry_run
        self.broker = broker if broker is not None else Broker(creds)
        self.data = data if data is not None else MarketData(creds, feed=config.data.feed)
        self.risk = risk if risk is not None else RiskManager(config.risk)
        self.strategy: Strategy = strategy or build_strategy(config.strategy, config.strategy_params)
        self.open_trades: dict[str, OpenTrade] = {}
        self.last_signal: dict[str, str] = {}
        self.last_signal_symbol: str | None = None
        self.last_signal_time: str | None = None
        log.info(
            "Trader ready | strategy=%s symbols=%s dry_run=%s mode=%s",
            self.strategy.name, ",".join(config.symbols), dry_run, creds.mode,
        )

    # ------------------------------------------------------------ single pass

    def run_cycle(self) -> None:
        account = self.broker.get_account()
        gate = self.risk.check_account(account)
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

        positions = self.broker.get_positions()

        # A trade we were tracking is no longer open -> the bracket's TP/SL
        # (or our own manual close) filled since the last cycle. Reconcile it.
        for symbol in list(self.open_trades):
            if symbol not in positions:
                self._finalize_trade(symbol)

        lookback_days = lookback_days_for(self.cfg.timeframe, self.cfg.lookback_bars)
        bars = self.data.get_bars(self.cfg.symbols, timeframe=self.cfg.timeframe, lookback_days=lookback_days)

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
            if crypto and position is not None and self._check_crypto_protection(symbol, df, position):
                continue

            signal = self.strategy.generate_signal(df)
            self.last_signal[symbol] = signal
            self.last_signal_symbol = symbol
            self.last_signal_time = datetime.now(timezone.utc).isoformat()
            if signal != "HOLD":
                log.info("SIGNAL %s %s", symbol, signal)

            if position is not None:
                self._maybe_exit(symbol, signal, position)
            else:
                self._maybe_enter(symbol, signal, df, account, positions, bars)

    def _check_crypto_protection(self, symbol: str, df, position: PositionSnapshot) -> bool:
        """Alpaca has no bracket-order support for crypto, so stop/target are
        watched here instead of living server-side. Returns True if a close
        was submitted (caller should skip further processing this cycle)."""
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
        wants_short = signal == SELL and self.strategy.allow_short
        if not (wants_long or wants_short):
            return
        if self.broker.has_open_order(symbol):
            log.info("SKIP %s - open order already pending", symbol)
            return

        regime = detect_regime(df)
        if not strategy_allows_regime(self.strategy.name, regime):
            log.info("SKIP %s %s - regime %s doesn't suit %s", symbol, signal, regime, self.strategy.name)
            return

        if positions:
            # Cheap on purpose: only correlate the candidate against symbols
            # actually open right now, not the full universe every cycle.
            relevant_bars = {symbol: df, **{s: bars[s] for s in positions if s in bars}}
            corr_matrix = compute_correlation_matrix(relevant_bars)
            blocker = find_correlated_open_position(symbol, set(positions), corr_matrix)
            if blocker:
                log.info("SKIP %s %s - correlated with open position %s", symbol, signal, blocker)
                return

        price = self.data.get_last_price(symbol) or float(df["close"].iloc[-1])
        decision = self.risk.approve_entry(symbol, price, self.strategy.name, account, positions)
        if not decision.approved:
            log.info("REJECTED %s %s - %s", symbol, signal, decision.reason)
            return

        side = "long" if wants_long else "short"
        stop = self.strategy.get_stop_loss(price, df, side)
        target = self.strategy.get_take_profit(price, df, side)

        if self.dry_run:
            log.info(
                "[DRY RUN] would %s %s x%s @ ~$%.2f (stop=%.2f target=%.2f)",
                "BUY" if wants_long else "SELL", symbol, decision.qty, price, stop, target,
            )
            return

        order_side = "buy" if wants_long else "sell"
        self.broker.submit_order(
            symbol, decision.qty, order_side,
            order_type=self.cfg.execution.order_type,
            time_in_force=self.cfg.execution.time_in_force,
            take_profit_price=target,
            stop_loss_price=stop,
        )

        entry_price = self._resolve_fill_price(symbol, fallback=price)
        self.open_trades[symbol] = OpenTrade(
            symbol=symbol, strategy=self.strategy.name, side=side, qty=decision.qty,
            entry_price=entry_price, stop_loss=stop, take_profit=target,
            entry_time=datetime.now(timezone.utc).isoformat(),
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
        for order in self.broker.get_closed_orders(symbol, limit=5):
            side = order.side.value if hasattr(order.side, "value") else order.side
            if side == closing_side and order.filled_avg_price is not None:
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
                except Exception:  # noqa: BLE001 - a transient API error must not kill the bot
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
