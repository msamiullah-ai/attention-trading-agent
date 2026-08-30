"""Risk management: trade logging, expectancy tracking, Kelly sizing, hard limits.

Every entry passes through `RiskManager.approve_entry` before an order is
sent. Every closed trade is recorded via `RiskManager.record_trade`, which
updates the running expectancy stats that both the position sizer and the
dashboard read from.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .broker import AccountSnapshot, PositionSnapshot
from .config import PROJECT_ROOT, RiskConfig
from .data import is_crypto_symbol
from .logger import get_logger

log = get_logger(__name__)

CSV_FIELDS = [
    "symbol", "strategy", "side", "qty", "entry_price", "exit_price",
    "stop_loss", "take_profit", "entry_time", "exit_time", "pnl", "r_multiple",
]


@dataclass
class TradeRecord:
    symbol: str
    strategy: str
    side: str  # "long" | "short"
    qty: float
    entry_price: float
    exit_price: float
    stop_loss: float
    entry_time: str  # ISO 8601
    exit_time: str
    pnl: float
    r_multiple: float
    # Defaulted so old callers/CSV rows without a planned target still work;
    # 0.0 means "unknown", not "no target was planned".
    take_profit: float = 0.0

    @property
    def is_win(self) -> bool:
        return self.pnl > 0

    @property
    def planned_reward_risk(self) -> float | None:
        """(take_profit - entry) / (entry - stop_loss) for a long, mirrored
        for a short -- the R/R the trade was planned at. None if take_profit
        wasn't recorded (0.0) or the stop distance is degenerate."""
        if self.take_profit == 0.0:
            return None
        risk = abs(self.entry_price - self.stop_loss)
        if risk <= 0:
            return None
        reward = abs(self.take_profit - self.entry_price)
        return reward / risk

    def to_csv_row(self) -> dict:
        return asdict(self)


class TradeLog:
    """Append-only trade history, persisted to CSV so state survives restarts."""

    def __init__(self, path: Path | None = None, in_memory: bool = False):
        """`in_memory=True` skips loading/writing CSV entirely — used by
        backtests so they never read or pollute the live trade history."""
        self.in_memory = in_memory
        self.path = None if in_memory else (path or PROJECT_ROOT / "state" / "trades.csv")
        self.trades: list[TradeRecord] = []
        if not in_memory:
            self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                self.trades.append(
                    TradeRecord(
                        symbol=row["symbol"],
                        strategy=row["strategy"],
                        side=row["side"],
                        qty=float(row["qty"]),
                        entry_price=float(row["entry_price"]),
                        exit_price=float(row["exit_price"]),
                        stop_loss=float(row["stop_loss"]),
                        entry_time=row["entry_time"],
                        exit_time=row["exit_time"],
                        pnl=float(row["pnl"]),
                        r_multiple=float(row["r_multiple"]),
                        # .get so CSV rows written before this field existed still load.
                        take_profit=float(row.get("take_profit") or 0.0),
                    )
                )
        log.info("Loaded %d historical trades from %s", len(self.trades), self.path)

    def record(self, trade: TradeRecord) -> None:
        self.trades.append(trade)
        if self.in_memory:
            log.debug("TRADE CLOSED (in-memory) %s %s pnl=$%.2f", trade.symbol, trade.side, trade.pnl)
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.path.exists()
        with open(self.path, "a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            if is_new:
                writer.writeheader()
            writer.writerow(trade.to_csv_row())
        log.info(
            "TRADE CLOSED %s %s qty=%s pnl=$%.2f (%.2fR) strategy=%s",
            trade.symbol, trade.side, trade.qty, trade.pnl, trade.r_multiple, trade.strategy,
        )

    def recent(self, n: int | None = None) -> list[TradeRecord]:
        return self.trades if n is None else self.trades[-n:]

    def for_strategy(self, strategy: str, n: int | None = None) -> list[TradeRecord]:
        matches = [t for t in self.trades if t.strategy == strategy]
        return matches if n is None else matches[-n:]


@dataclass
class ExpectancyStats:
    n: int = 0
    win_rate: float = 0.0
    loss_rate: float = 0.0
    avg_win: float = 0.0          # dollars
    avg_loss: float = 0.0         # dollars, positive magnitude
    avg_r_winner: float = 0.0     # R-multiple of the average winning trade
    reward_risk_ratio: float = 0.0
    expectancy: float = 0.0       # dollars per trade
    expectancy_r: float = 0.0     # R-multiples per trade
    breakeven_win_rate: float = 0.0
    margin_of_safety: float = 0.0  # win_rate - breakeven_win_rate

    @property
    def has_data(self) -> bool:
        return self.n > 0


def compute_expectancy(trades: list[TradeRecord]) -> ExpectancyStats:
    """Expectancy = win_rate*avg_win - loss_rate*avg_loss (dollars).

    Expectancy_R = win_rate*avg_R_winner - loss_rate*1.0, treating the
    average loser as exactly 1R (that's the definition of R: risk per trade).
    reward_risk_ratio = avg_R_winner, so breakeven_win_rate = 1/(1+RR) lines
    up with the same R-multiple convention.
    """
    if not trades:
        return ExpectancyStats()

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    n = len(trades)

    win_rate = len(wins) / n
    loss_rate = len(losses) / n
    avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(t.pnl for t in losses) / len(losses)) if losses else 0.0
    avg_r_winner = sum(t.r_multiple for t in wins) / len(wins) if wins else 0.0

    reward_risk_ratio = avg_r_winner
    expectancy = win_rate * avg_win - loss_rate * avg_loss
    expectancy_r = win_rate * avg_r_winner - loss_rate * 1.0
    breakeven_win_rate = 1 / (1 + reward_risk_ratio) if reward_risk_ratio > 0 else 1.0
    margin_of_safety = win_rate - breakeven_win_rate

    return ExpectancyStats(
        n=n,
        win_rate=win_rate,
        loss_rate=loss_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        avg_r_winner=avg_r_winner,
        reward_risk_ratio=reward_risk_ratio,
        expectancy=expectancy,
        expectancy_r=expectancy_r,
        breakeven_win_rate=breakeven_win_rate,
        margin_of_safety=margin_of_safety,
    )


def kelly_fraction(win_rate: float, reward_risk_ratio: float) -> float:
    """kelly = (win_rate * RR - loss_rate) / RR. 0 if RR <= 0 (no edge data)."""
    if reward_risk_ratio <= 0:
        return 0.0
    loss_rate = 1 - win_rate
    return (win_rate * reward_risk_ratio - loss_rate) / reward_risk_ratio


@dataclass
class Decision:
    approved: bool
    qty: float = 0.0
    reason: str = ""


class RiskManager:
    def __init__(self, config: RiskConfig, trade_log: TradeLog | None = None):
        self.cfg = config
        self.trade_log = trade_log or TradeLog()

    # ---------------------------------------------------------- expectancy

    def expectancy(self, strategy: str | None = None, window: int | None = None) -> ExpectancyStats:
        trades = (
            self.trade_log.for_strategy(strategy, window)
            if strategy
            else self.trade_log.recent(window)
        )
        return compute_expectancy(trades)

    def record_trade(self, trade: TradeRecord) -> None:
        self.trade_log.record(trade)

    # ------------------------------------------------------------ kill switches

    def check_account(self, account: AccountSnapshot) -> Decision:
        """Account-level gate evaluated once per cycle before any trading."""
        if account.blocked:
            return Decision(False, reason="account is blocked for trading")

        loss = -account.daily_pnl_pct
        if loss >= self.cfg.max_daily_loss_pct:
            return Decision(
                False,
                reason=(
                    f"daily loss {loss:.2%} hit limit {self.cfg.max_daily_loss_pct:.2%} "
                    "- no new trades today"
                ),
            )
        return Decision(True, reason="account OK")

    def check_expectancy_pause(self, strategy: str) -> Decision:
        """Halt entries for `strategy` if it has gone net-negative recently."""
        stats = self.expectancy(strategy=strategy, window=self.cfg.expectancy_window)
        if stats.n < self.cfg.expectancy_window:
            return Decision(True, reason=f"only {stats.n} trades logged, pause check skipped")
        if stats.expectancy_r < 0:
            return Decision(
                False,
                reason=(
                    f"{strategy} expectancy {stats.expectancy_r:+.2f}R over last "
                    f"{stats.n} trades - paused"
                ),
            )
        return Decision(True, reason="expectancy OK")

    # ---------------------------------------------------------------- sizing

    def kelly_position_pct(self, strategy: str) -> float:
        """Fraction of equity to risk, before the hard cap is applied.

        Uses half-Kelly (or whatever `kelly_multiplier` is set to) once enough
        trade history exists; otherwise falls back to a small flat size so the
        bot isn't sizing blind off zero data.
        """
        stats = self.expectancy(strategy=strategy)
        if stats.n < self.cfg.min_trades_for_kelly:
            return self.cfg.fallback_position_pct

        raw = kelly_fraction(stats.win_rate, stats.reward_risk_ratio)
        return max(0.0, raw) * self.cfg.kelly_multiplier

    def size_position(self, account: AccountSnapshot, price: float, strategy: str, symbol: str) -> float:
        """Quantity for a new position, 0 if nothing affordable.

        Crypto sizes fractionally (e.g. 0.0031 BTC) since Alpaca trades it
        that way; equities size in whole shares. Flooring to a whole unit
        for crypto would round almost every position down to exactly 0 --
        a $60k+ BTC price against a typical few-percent equity slice is
        nowhere near a full coin.
        """
        if price <= 0:
            return 0.0
        fraction = min(self.kelly_position_pct(strategy), self.cfg.max_position_pct)
        target_dollars = min(account.equity * fraction, account.buying_power)
        if is_crypto_symbol(symbol):
            return round(target_dollars / price, 6)
        return float(int(target_dollars // price))

    # ----------------------------------------------------------- entry gate

    def approve_entry(
        self,
        symbol: str,
        price: float,
        strategy: str,
        account: AccountSnapshot,
        positions: dict[str, PositionSnapshot],
    ) -> Decision:
        if symbol in positions:
            return Decision(False, reason=f"already holding {symbol}")

        if len(positions) >= self.cfg.max_open_positions:
            return Decision(
                False, reason=f"at max open positions ({self.cfg.max_open_positions})"
            )

        pause = self.check_expectancy_pause(strategy)
        if not pause.approved:
            return pause

        qty = self.size_position(account, price, strategy, symbol)
        min_qty = 1e-6 if is_crypto_symbol(symbol) else 1.0
        if qty < min_qty:
            return Decision(
                False,
                reason=(
                    f"sized qty {qty} below minimum "
                    f"(price ${price:,.2f}, buying power ${account.buying_power:,.2f})"
                ),
            )

        order_value = qty * price
        if order_value > account.buying_power:
            return Decision(
                False,
                reason=(
                    f"order ${order_value:,.2f} exceeds buying power "
                    f"${account.buying_power:,.2f}"
                ),
            )

        return Decision(True, qty=qty, reason=f"approved {qty} shares (${order_value:,.2f})")
