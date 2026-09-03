"""Shared test setup.

The suite must not depend on whatever happens to be in a developer's .env.
Before the agent layer existed nothing read it during tests, so this was moot;
now `Trader.__init__` calls `load_agent_settings()`, and without this fixture a
machine with TRADING_BOT_ADVISOR=featherless set would build a real advisor,
make live HTTP calls from a unit test, and fail assertions that have nothing to
do with the code under test.

`load_dotenv` does not override variables already present in the environment,
so setting them here is enough to shadow the file for the whole session.
"""

from __future__ import annotations

import pytest

# Every variable load_agent_settings reads, pinned to its off/default value.
_AGENT_ENV = {
    "TRADING_BOT_ADVISOR": "none",
    "TRADING_BOT_FUSION": "off",
    "TRADING_BOT_FUSION_TEMPERATURE": "0.5",
    "LLM_BASE_URL": "",
    "LLM_MODEL": "",
    "LLM_API_KEY": "",
    "LLM_TIMEOUT": "60",
    "LLM_MAX_TOKENS": "2048",
    "ANTHROPIC_API_KEY": "",
    "ANTHROPIC_MODEL": "",
}


@pytest.fixture(autouse=True)
def isolate_state_dir(tmp_path, monkeypatch):
    """Keep every test's on-disk state out of the real `state/` directory.

    `Trader.__init__` builds an `AdvisorMemory()` and a `StrategyMemory()` on
    their default paths whenever none is injected, so the wiring tests -- which
    pass a canned advisor and then run a cycle -- were appending their fixtures
    to the live ledger. The dashboard duly rendered them: AAPL vetoed for
    "halt", reason "v", alongside real verdicts.

    Autouse and directory-level rather than per-class, so a future component
    that writes to `state/` is isolated without anyone remembering to.
    """
    # One patch point. Every state file now resolves through
    # `jsonl.state_path`, which reads PROJECT_ROOT from config at call time, so
    # redirecting it here covers each of them -- and covers any module added
    # later without this list needing to know about it.
    import trading_bot.config as cfg
    import trading_bot.options_trader as ot
    import trading_bot.risk as rk
    monkeypatch.setattr(cfg, "PROJECT_ROOT", tmp_path)
    for module in (ot, rk):
        if hasattr(module, "PROJECT_ROOT"):
            monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)


@pytest.fixture(autouse=True)
def isolate_agent_env(monkeypatch):
    """Neutralise the agent layer's configuration for every test.

    Autouse rather than opt-in: a test that accidentally reaches the network is
    exactly the kind of failure that shows up as a flake weeks later, and
    remembering to request a fixture is not a defence against that.

    Tests that want the agent on pass an explicit AgentSettings or advisor to
    the constructor, which is injection rather than environment, and therefore
    unaffected by anything here.
    """
    for name, value in _AGENT_ENV.items():
        monkeypatch.setenv(name, value)
