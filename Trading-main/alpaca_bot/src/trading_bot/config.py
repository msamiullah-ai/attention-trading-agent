"""Configuration loading: credentials from .env, tunables from config.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Credentials:
    api_key: str
    secret_key: str
    paper: bool

    @property
    def mode(self) -> str:
        return "PAPER" if self.paper else "LIVE"


@dataclass
class RiskConfig:
    # Position sizing: Kelly-derived fraction of equity, scaled by this
    # multiplier (quarter=0.25, half=0.5, full=1.0) and always capped below.
    kelly_multiplier: float = 0.5
    # Hard ceiling on any single trade's size, regardless of what Kelly says.
    max_position_pct: float = 0.05
    max_open_positions: int = 3
    max_daily_loss_pct: float = 0.03
    # Trades required before trusting a live win-rate/R-ratio for Kelly sizing.
    min_trades_for_kelly: int = 20
    # Flat position size used before min_trades_for_kelly is reached.
    fallback_position_pct: float = 0.02
    # Window of recent trades checked for the negative-expectancy pause.
    expectancy_window: int = 20
    # Size equities fractionally when a whole share costs more than the risk
    # budget allows. Keeps the same risk percentages valid at any account size
    # -- without it a $1,000 account cannot open a position in most symbols and
    # simply never trades. Fractional positions cannot carry a broker-side
    # bracket, so they are protected by the in-process stop/target watch.
    allow_fractional_equity: bool = True
    # Measured correlation at/above this blocks a second, near-duplicate bet.
    # Was a function default in correlation.py that nothing ever passed, so it
    # could not be tuned without editing code -- which contradicts this file's
    # own promise that everything tunable lives here.
    correlation_threshold: float = 0.75
    # Extra known-correlated pairs, merged with correlation.HARDCODED_PAIRS.
    # Lets a universe declare its own relationships (KO/PEP, V/MA, DAL/UAL...)
    # without touching source.
    correlated_pairs: list[list[str]] = field(default_factory=list)


@dataclass
class ExecutionConfig:
    # How orders reach Alpaca. "rest" is alpaca-py's TradingClient; "mcp" is
    # Alpaca's MCP server, which the event requires. MCPBroker is a drop-in for
    # Broker, so this swaps the transport without touching the trading logic.
    # MCP additionally gives the agent layer news and corporate actions, which
    # the REST client cannot reach at all.
    transport: str = "rest"
    # Which candidate gets a position slot when more symbols qualify than
    # max_open_positions allows.
    #   none = first to qualify in config.yaml order (what earlier versions did)
    #   drr  = Deficit Round Robin: best signal wins, and a symbol repeatedly
    #          passed over accrues credit until it cannot be ignored
    scheduler: str = "none"
    # How capital is split between the equity and options sides.
    #   none    = each side sizes against full equity independently. Run both
    #             processes and they can commit past 100% between them.
    #   dynamic = allocation.decide() sets a ceiling for each, from the regime,
    #             trend strength, volatility and recent form.
    allocator: str = "none"
    order_type: str = "market"
    time_in_force: str = "day"
    limit_slippage_pct: float = 0.001
    require_market_open: bool = True


@dataclass
class EngineConfig:
    poll_interval_seconds: int = 60


@dataclass
class DataConfig:
    feed: str = "iex"


@dataclass
class Config:
    symbols: list[str]
    strategy: str
    strategy_params: dict[str, Any] = field(default_factory=dict)
    timeframe: str = "1Min"
    lookback_bars: int = 250
    data: DataConfig = field(default_factory=DataConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)


@dataclass
class AgentSettings:
    """Everything the agent layer reads from .env, resolved in one place.

    Kept here beside Credentials rather than scattered through the modules that
    use it. Reading os.environ at each use site means the answer to "what is
    this bot configured to do" is spread across three files, nothing can be
    tested without monkeypatching the environment, and a typo in a variable name
    silently becomes a default instead of an error.
    """

    advisor: str = "none"           # none | anthropic | openai | featherless | ...
    fusion: bool = False
    fusion_temperature: float = 0.5
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    timeout: float = 60.0
    # Reasoning models spend tokens thinking before they write anything, and a
    # budget that runs out mid-thought returns EMPTY content with
    # finish_reason=length -- which the parser correctly treats as a confirm, so
    # the advisor silently reviews nothing. Measured against Qwen3-27B: 300
    # always truncated, 1024 truncated on the ambiguous cases, 2048 finished
    # with ~1800 used. Non-reasoning models ignore the headroom, so the cost of
    # setting this high is only paid by models that need it.
    max_tokens: int = 2048

    @property
    def advisor_enabled(self) -> bool:
        return self.advisor not in ("none", "off", "disabled", "")

    def redacted(self) -> dict[str, Any]:
        """Safe to log or print. The key becomes a length, never a value --
        a truncated key in scrollback is still a key."""
        return {
            "advisor": self.advisor,
            "model": self.model or "(default)",
            "base_url": self.base_url or "(provider default)",
            "api_key": f"set ({len(self.api_key)} chars)" if self.api_key else "MISSING",
            "fusion": self.fusion,
            "fusion_temperature": self.fusion_temperature,
            "max_tokens": self.max_tokens,
        }


# Providers that speak OpenAI's /v1/chat/completions, mapped to their endpoint
# so nobody has to look one up. An empty string means "let the caller supply it".
PROVIDER_ENDPOINTS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "featherless": "https://api.featherless.ai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "groq": "https://api.groq.com/openai/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "ollama": "http://localhost:11434/v1",
    "lmstudio": "http://localhost:1234/v1",
}

_TRUE = {"1", "true", "yes", "on"}


def load_agent_settings(env_path: Path | None = None) -> AgentSettings:
    """Read the agent layer's configuration from .env. The only place that does.

    Anthropic keeps its own key variable because it is a different account with
    a different key format; every other provider shares LLM_* because they share
    an API. The endpoint is filled in from the provider name when it is one we
    know, so `TRADING_BOT_ADVISOR=featherless` is enough on its own.
    """
    load_dotenv(env_path or PROJECT_ROOT / ".env")

    advisor = os.getenv("TRADING_BOT_ADVISOR", "none").strip().lower()
    anthropic = advisor == "anthropic"

    base_url = os.getenv("LLM_BASE_URL", "").strip().rstrip("/")
    if not base_url and not anthropic:
        base_url = PROVIDER_ENDPOINTS.get(advisor, "")

    return AgentSettings(
        advisor=advisor,
        fusion=os.getenv("TRADING_BOT_FUSION", "").strip().lower() in _TRUE,
        fusion_temperature=_read_float("TRADING_BOT_FUSION_TEMPERATURE", 0.5),
        model=os.getenv("ANTHROPIC_MODEL" if anthropic else "LLM_MODEL", "").strip(),
        api_key=os.getenv("ANTHROPIC_API_KEY" if anthropic else "LLM_API_KEY", "").strip(),
        base_url=base_url,
        timeout=_read_float("LLM_TIMEOUT", 60.0),
        max_tokens=int(_read_float("LLM_MAX_TOKENS", 2048)),
    )


def _read_float(name: str, default: float) -> float:
    """A malformed number falls back to the default rather than crashing the
    bot at startup over a stray character in an optional tuning knob."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def load_credentials(env_path: Path | None = None) -> Credentials:
    """Read Alpaca credentials from .env / environment.

    Raises RuntimeError with actionable text rather than letting the SDK fail
    with an opaque 403 later on.
    """
    load_dotenv(env_path or PROJECT_ROOT / ".env")

    api_key = os.getenv("ALPACA_API_KEY", "").strip()
    secret_key = os.getenv("ALPACA_SECRET_KEY", "").strip()
    paper = os.getenv("ALPACA_PAPER", "true").strip().lower() not in {"false", "0", "no"}

    missing = [
        name
        for name, value in (("ALPACA_API_KEY", api_key), ("ALPACA_SECRET_KEY", secret_key))
        if not value or value.startswith("your_")
    ]
    if missing:
        raise RuntimeError(
            f"Missing credentials: {', '.join(missing)}. "
            "Copy .env.example to .env and fill in your Alpaca keys."
        )

    return Credentials(api_key=api_key, secret_key=secret_key, paper=paper)


VALID_TIMEFRAMES = {"1min", "5min", "15min", "1hour", "1day"}
VALID_STRATEGIES = {"ema_rsi", "vwap_mean_reversion", "bollinger_squeeze",
                    "crypto_momentum", "opening_range_breakout"}
VALID_TRANSPORTS = {"rest", "mcp"}
VALID_SCHEDULERS = {"none", "drr"}
VALID_ALLOCATORS = {"none", "dynamic"}


def load_config(path: Path | None = None) -> Config:
    """Load config.yaml into typed dataclasses."""
    path = path or PROJECT_ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    symbols = [s.strip().upper() for s in raw.get("symbols", []) if s.strip()]
    if not symbols:
        raise ValueError(f"No symbols configured in {path}")

    strategy = raw.get("strategy", "ema_rsi")
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}. Available: {sorted(VALID_STRATEGIES)}")

    timeframe = raw.get("timeframe", "1Min")
    if timeframe.lower() not in VALID_TIMEFRAMES:
        raise ValueError(f"Invalid timeframe {timeframe!r}. Use one of {sorted(VALID_TIMEFRAMES)}")

    lookback_bars = int(raw.get("lookback_bars", 250))
    min_bars = {"ema_rsi": 22, "vwap_mean_reversion": 22, "bollinger_squeeze": 42,
                "crypto_momentum": 122, "opening_range_breakout": 22}
    if lookback_bars < min_bars.get(strategy, 30):
        raise ValueError(
            f"lookback_bars={lookback_bars} too small for {strategy} (needs {min_bars.get(strategy, 30)}+)"
        )

    execution = ExecutionConfig(**(raw.get("execution") or {}))
    if execution.transport not in VALID_TRANSPORTS:
        raise ValueError(
            f"Invalid transport {execution.transport!r}. Use one of {sorted(VALID_TRANSPORTS)}"
        )
    if execution.scheduler not in VALID_SCHEDULERS:
        raise ValueError(
            f"Invalid scheduler {execution.scheduler!r}. Use one of {sorted(VALID_SCHEDULERS)}"
        )
    if execution.allocator not in VALID_ALLOCATORS:
        raise ValueError(
            f"Invalid allocator {execution.allocator!r}. Use one of {sorted(VALID_ALLOCATORS)}"
        )

    return Config(
        symbols=symbols,
        strategy=strategy,
        strategy_params=raw.get("strategy_params") or {},
        timeframe=timeframe,
        lookback_bars=lookback_bars,
        data=DataConfig(**(raw.get("data") or {})),
        risk=RiskConfig(**(raw.get("risk") or {})),
        execution=execution,
        engine=EngineConfig(**(raw.get("engine") or {})),
    )
