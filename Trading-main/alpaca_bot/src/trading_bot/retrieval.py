"""Retrieval: news and corporate actions, fetched per candidate.

The retrieval half of what the README calls RAG. There is no vector store and
no embedding model, and that is a choice rather than a shortcut: the corpus is
a handful of headlines about one ticker from the last few days, fetched fresh
each cycle. Embedding six documents to rank six documents adds a dependency, a
model and a failure mode, and answers a question that `get_news` already
answered by filtering on symbol and date.

If the corpus grows to thousands of documents across hundreds of names, that
calculus changes and a real index earns its place. Saying so here is more
useful than implying a vector store exists.

WHAT THIS IS ACTUALLY FOR, beyond filling a promise:

Selling premium into a scheduled event is the classic way a short-option book
dies. Implied vol collapses the morning after -- which is what a seller wants
-- but the stock gaps through the strike, which is not. The delta band cannot
see it, the liquidity gate cannot see it, and price history cannot see it,
because the event has not happened yet. It is only visible in a calendar.

A CORRECTION WORTH READING BEFORE YOU TRUST THIS

An earlier version of this module claimed it enforced an earnings blackout via
`get_corporate_action_announcements`. It cannot. Alpaca's endpoint accepts only
`ca_types` of dividend, merger, spinoff and split -- passing `earnings` returns
422. There is no earnings calendar in this API.

So the split is:

  - Dividends, mergers, spinoffs and splits ARE checked here, and they matter:
    an ex-dividend date drives early assignment on a short call, and a merger
    pins the stock and kills the premium the strategy is selling.

  - Earnings are NOT. The only signal available is a news headline saying so,
    which reaches the model through `news` below and is judged in prose rather
    than enforced by a gate. That is genuinely weaker: headlines are not a
    calendar, and absence of one is not absence of a print.

Naming that gap is more useful than a gate that reports "checked" while looking
at the wrong thing entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

from .logger import get_logger

log = get_logger(__name__)


@dataclass
class NewsItem:
    headline: str
    summary: str
    source: str
    created_at: str
    symbols: list[str] = field(default_factory=list)

    def brief(self, chars: int = 180) -> str:
        text = self.headline.strip()
        if self.summary:
            text = f"{text} -- {self.summary.strip()}"
        return text[:chars]


@dataclass
class EventRisk:
    """What the calendar says is coming for one ticker."""

    ticker: str
    has_earnings: bool = False
    earnings_date: str | None = None
    days_until: int | None = None
    other_actions: list[str] = field(default_factory=list)
    checked: bool = False           # False means we could not look, not that it is clear

    @property
    def blackout(self) -> bool:
        """True when an event sits inside the window we would hold through."""
        return self.has_earnings and self.days_until is not None and self.days_until >= 0

    def reason(self) -> str:
        if not self.checked:
            return "event calendar unavailable; treated as unknown, not as clear"
        if self.blackout:
            return (f"earnings for {self.ticker} in {self.days_until}d "
                    f"({self.earnings_date}); premium collapses but the stock gaps")
        # Deliberately not "no earnings": this endpoint has no earnings data,
        # and a reviewer that reads "no earnings" as a cleared check is being
        # told something the API never said.
        return "no dividend, merger, spinoff or split scheduled in the window"


@dataclass
class Context:
    """Everything retrieved for one candidate, ready to be handed to a model."""

    ticker: str
    news: list[NewsItem] = field(default_factory=list)
    events: EventRisk | None = None

    def to_prompt(self, max_items: int = 5) -> str:
        """Compact, and capped on purpose.

        Twenty headlines about one ticker is not twenty facts, it is one story
        repeated by twenty outlets. Feeding all of them spends context and
        invites the model to read repetition as significance.
        """
        lines: list[str] = []
        if self.events:
            lines.append(f"EVENTS: {self.events.reason()}")
            if self.events.other_actions:
                lines.append(f"OTHER ACTIONS: {', '.join(self.events.other_actions[:3])}")
        if self.news:
            lines.append(f"NEWS ({len(self.news)} items, newest first):")
            for item in self.news[:max_items]:
                lines.append(f"  - [{item.created_at[:10]}] {item.brief()}")
        else:
            lines.append("NEWS: none returned for this ticker in the window")
        return "\n".join(lines)


class Retriever:
    """Fetches context over the MCP server. Never raises.

    Every method degrades to "unknown" rather than propagating, because a news
    endpoint being slow must not stop the bot trading. The distinction that
    matters is that unknown is recorded AS unknown -- an empty news list and a
    failed fetch are different facts, and only one of them means "nothing is
    happening".
    """

    def __init__(self, broker, lookback_days: int = 3):
        self.broker = broker
        self.lookback_days = lookback_days

    def news(self, ticker: str, limit: int = 10) -> list[NewsItem]:
        start = (datetime.now(timezone.utc)
                 - timedelta(days=self.lookback_days)).strftime("%Y-%m-%d")
        try:
            raw = self.broker._call("news", symbols=ticker, start=start, limit=limit)
        except Exception as exc:  # noqa: BLE001 - see class docstring
            log.debug("news unavailable for %s: %s", ticker, exc)
            return []
        rows = raw
        if isinstance(raw, dict):
            rows = raw.get("news") or raw.get("articles") or []
        if not isinstance(rows, list):
            return []

        out: list[NewsItem] = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            out.append(NewsItem(
                headline=str(r.get("headline") or r.get("title") or "").strip(),
                summary=str(r.get("summary") or "").strip(),
                source=str(r.get("source") or r.get("author") or "").strip(),
                created_at=str(r.get("created_at") or r.get("updated_at") or ""),
                symbols=[str(s) for s in (r.get("symbols") or [])],
            ))
        return [n for n in out if n.headline]

    # Every type Alpaca's announcements endpoint accepts. `earnings` is NOT one
    # of them -- it returns 422 -- see the module docstring.
    CA_TYPES = ("dividend", "merger", "spinoff", "split")

    def events(self, ticker: str, horizon_days: int = 7,
               today: date | None = None) -> EventRisk:
        """Scheduled corporate actions inside the holding window.

        Parameter names are the REST API's own (`ca_types`, `since`, `until`),
        not the plausible-looking `symbols`/`start`/`end`. The server silently
        drops unknown arguments with a warning and queries the whole market, so
        a wrong name here does not fail -- it quietly returns the wrong answer.

        `checked=False` on failure is load-bearing. A gate that cannot tell the
        difference between "nothing scheduled" and "could not look" will happily
        sell premium into an event and report that it checked.
        """
        today = today or date.today()
        risk = EventRisk(ticker=ticker)
        until = (today + timedelta(days=horizon_days)).isoformat()
        rows: list = []
        checked_any = False
        for ca_type in self.CA_TYPES:
            try:
                raw = self.broker._call("corporate_actions", ca_types=ca_type,
                                        since=today.isoformat(), until=until,
                                        symbol=ticker)
            except Exception as exc:  # noqa: BLE001
                log.debug("corporate actions (%s) unavailable for %s: %s",
                          ca_type, ticker, exc)
                continue
            checked_any = True
            batch = raw
            if isinstance(raw, dict):
                batch = (raw.get("corporate_actions") or raw.get("announcements")
                         or raw.get("results") or [])
            if isinstance(batch, list):
                rows.extend(batch)
        if not checked_any:
            return risk                      # checked stays False

        risk.checked = True
        for r in rows:
            if not isinstance(r, dict):
                continue
            kind = str(r.get("ca_type") or r.get("type") or r.get("corporate_action_type")
                       or "").lower()
            when = str(r.get("ex_date") or r.get("effective_date")
                       or r.get("declaration_date") or r.get("date") or "")
            if "earning" in kind:
                risk.has_earnings = True
                risk.earnings_date = when or None
                risk.days_until = _days_between(today, when)
            elif kind:
                risk.other_actions.append(f"{kind} {when}".strip())
        return risk

    def context(self, ticker: str, horizon_days: int = 7,
                today: date | None = None) -> Context:
        return Context(ticker=ticker, news=self.news(ticker),
                       events=self.events(ticker, horizon_days, today))


def _days_between(today: date, iso: str) -> int | None:
    try:
        y, m, d = (int(x) for x in iso[:10].split("-"))
    except (ValueError, AttributeError):
        return None
    return (date(y, m, d) - today).days
