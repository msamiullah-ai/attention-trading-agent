"""Tests for the agent layer: attention, retrieval, and the LLM advisor.

The advisor tests lean heavily on adversarial replies, because the realistic
failure here is not "the API is down" -- that path is one try/except. It is a
model that answers confidently in the wrong shape, and every one of those must
land somewhere safe.
"""

from __future__ import annotations

from datetime import date

import pytest

from trading_bot.agent import (
    CONFIRM, MIN_SHRINK, SHRINK, VETO, Advice, NullAdvisor, build_advisor,
    build_prompt, parse_advice, review,
)
from trading_bot.attention import UNSUITED, Verdict, attend, softmax, suitability
from trading_bot.config import AgentSettings, load_agent_settings
from trading_bot.retrieval import Context, EventRisk, NewsItem, Retriever, _days_between
from trading_bot.signals import BUY, HOLD, SELL


# --------------------------------------------------------------------------
# attention
# --------------------------------------------------------------------------

def test_softmax_sums_to_one():
    w = softmax({"a": 1.0, "b": 2.0, "c": 0.5})
    assert pytest.approx(sum(w.values()), abs=1e-9) == 1.0


def test_softmax_survives_tiny_temperature():
    """Near-zero temperature is how you manufacture an exp() overflow."""
    w = softmax({"a": 1.0, "b": 0.0}, temperature=1e-9)
    assert pytest.approx(sum(w.values()), abs=1e-9) == 1.0
    assert w["a"] > 0.99          # winner-take-all, but still finite


def test_softmax_empty():
    assert softmax({}) == {}


def test_low_temperature_reproduces_pick_one_strategy():
    """The claim in the docstring: their current design is one end of the dial."""
    votes = {"ema_rsi": BUY, "vwap_mean_reversion": SELL}
    w = attend(votes, regime="BULL", temperature=0.01).weights
    assert w["ema_rsi"] > 0.99


def test_high_temperature_approaches_uniform():
    votes = {"ema_rsi": BUY, "vwap_mean_reversion": SELL}
    w = attend(votes, regime="BULL", temperature=1000.0).weights
    assert abs(w["ema_rsi"] - 0.5) < 0.01


def test_suitability_derives_from_their_table():
    assert suitability("ema_rsi", "BULL") == 1.0
    assert suitability("ema_rsi", "SIDEWAYS") == UNSUITED
    assert suitability("nonexistent_strategy", "BULL") == 0.6   # ungated


def test_unsuited_strategy_keeps_a_voice():
    """The substantive difference from the boolean gate."""
    assert suitability("vwap_mean_reversion", "BULL") > 0


def test_unanimous_buy():
    v = attend({"ema_rsi": BUY, "crypto_momentum": BUY}, regime="BULL")
    assert v.action == BUY
    assert v.agreement == pytest.approx(1.0)


def test_split_vote_holds_rather_than_trading_on_noise():
    """2-2 with equal weight nets to ~0; the deadband must catch it."""
    v = attend({"ema_rsi": BUY, "crypto_momentum": BUY,
                "bollinger_squeeze": SELL, "vwap_mean_reversion": SELL},
               regime="BULL", temperature=1000.0)
    assert v.action == HOLD


def test_minority_can_be_outweighed_not_silenced():
    v = attend({"ema_rsi": BUY, "vwap_mean_reversion": SELL}, regime="BULL")
    assert v.action == BUY
    assert v.weights["vwap_mean_reversion"] > 0     # counted, just outvoted


def test_reliability_quiets_without_silencing():
    base = attend({"ema_rsi": BUY, "bollinger_squeeze": SELL}, regime="BULL")
    quiet = attend({"ema_rsi": BUY, "bollinger_squeeze": SELL}, regime="BULL",
                   reliability={"ema_rsi": 0.1})
    assert quiet.weights["ema_rsi"] < base.weights["ema_rsi"]
    assert quiet.weights["ema_rsi"] > 0


def test_reliability_out_of_range_is_clamped():
    v = attend({"ema_rsi": BUY}, regime="BULL", reliability={"ema_rsi": 99.0})
    assert pytest.approx(sum(v.weights.values())) == 1.0


def test_empty_votes_holds():
    v = attend({}, regime="BULL")
    assert v.action == HOLD and v.confidence == 0.0


def test_all_hold_is_hold_with_agreement():
    v = attend({"ema_rsi": HOLD, "crypto_momentum": HOLD}, regime="BULL")
    assert v.action == HOLD
    assert v.agreement == pytest.approx(1.0)


def test_explain_names_every_strategy():
    text = attend({"ema_rsi": BUY, "vwap_mean_reversion": SELL}, regime="BULL").explain()
    assert "ema_rsi" in text and "vwap_mean_reversion" in text


def test_threshold_is_respected():
    votes = {"ema_rsi": BUY, "vwap_mean_reversion": SELL}
    assert attend(votes, regime="BULL", threshold=0.99).action == HOLD


# --------------------------------------------------------------------------
# retrieval
# --------------------------------------------------------------------------

class FakeBroker:
    """Stands in for MCPBroker. Mirrors the real server closely enough to matter.

    `corporate_actions` is queried once per ca_type, and the live endpoint
    returns only rows of the type asked for. A fake that ignores `ca_types` and
    replays every row on all four calls would quadruple the results and make a
    passing test meaningless.
    """

    def __init__(self, responses=None, raises=False):
        self.responses = responses or {}
        self.raises = raises
        self.calls: list[tuple] = []

    def _call(self, logical, **kwargs):
        if self.raises:
            raise RuntimeError("mcp down")
        self.calls.append((logical, kwargs))
        rows = self.responses.get(logical, [])
        wanted = kwargs.get("ca_types")
        if logical == "corporate_actions" and wanted and isinstance(rows, list):
            return [r for r in rows
                    if isinstance(r, dict)
                    and wanted in str(r.get("ca_type", r.get("type", ""))).lower()]
        return rows


def test_news_failure_is_empty_not_an_exception():
    assert Retriever(FakeBroker(raises=True)).news("AAPL") == []


def test_events_failure_records_unchecked():
    """The load-bearing distinction: could not look != nothing there."""
    risk = Retriever(FakeBroker(raises=True)).events("AAPL")
    assert risk.checked is False
    assert risk.blackout is False
    assert "unavailable" in risk.reason()


def test_events_success_with_no_earnings_is_checked_and_clear():
    risk = Retriever(FakeBroker({"corporate_actions": []})).events("AAPL")
    assert risk.checked is True and risk.blackout is False


def test_earnings_are_not_reachable_from_corporate_actions():
    """This used to assert an earnings blackout. It cannot work: Alpaca's
    announcements endpoint 422s on ca_types=earnings, so the type is never
    queried and such a row is never returned. Kept as a test of the real
    behaviour so nobody re-adds the assumption -- earnings reach the model
    only as a news headline, judged in prose rather than enforced by a gate."""
    r = Retriever(FakeBroker({"corporate_actions": [
        {"ca_type": "earnings", "ex_date": "2026-09-10"}]}))
    risk = r.events("AAPL", today=date(2026, 9, 3))
    assert risk.checked is True          # we did look
    assert risk.blackout is False        # at everything the API actually offers


def test_past_earnings_is_not_a_blackout():
    r = Retriever(FakeBroker({"corporate_actions": [
        {"ca_type": "earnings", "ex_date": "2026-08-01"}]}))
    assert r.events("AAPL", today=date(2026, 9, 3)).blackout is False


def test_non_earnings_actions_are_recorded_separately():
    r = Retriever(FakeBroker({"corporate_actions": [
        {"ca_type": "dividend", "ex_date": "2026-09-05"}]}))
    risk = r.events("AAPL", today=date(2026, 9, 3))
    assert risk.blackout is False and len(risk.other_actions) == 1


def test_news_parses_dict_and_list_shapes():
    row = {"headline": "Beat", "summary": "s", "created_at": "2026-09-01T10:00:00Z"}
    assert len(Retriever(FakeBroker({"news": [row]})).news("AAPL")) == 1
    assert len(Retriever(FakeBroker({"news": {"news": [row]}})).news("AAPL")) == 1


def test_news_drops_rows_without_a_headline():
    r = Retriever(FakeBroker({"news": [{"summary": "no headline"}, {"headline": "ok"}]}))
    assert len(r.news("AAPL")) == 1


def test_news_tolerates_garbage_rows():
    r = Retriever(FakeBroker({"news": ["a string", None, {"headline": "ok"}]}))
    assert len(r.news("AAPL")) == 1


def test_prompt_caps_items():
    news = [NewsItem(f"h{i}", "", "", "2026-09-01") for i in range(20)]
    assert Context("AAPL", news=news).to_prompt(max_items=3).count("  - ") == 3


def test_prompt_says_so_when_there_is_no_news():
    assert "none returned" in Context("AAPL").to_prompt()


def test_days_between_handles_junk():
    assert _days_between(date(2026, 9, 3), "not-a-date") is None
    assert _days_between(date(2026, 9, 3), "") is None


# --------------------------------------------------------------------------
# advisor: authority limits
# --------------------------------------------------------------------------

class CannedAdvisor:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        return self.reply


def test_confirm_never_increases_size():
    """The central invariant: approval grants nothing."""
    assert parse_advice('{"action":"confirm","size_factor":5.0}').apply_to(10) == 10


def test_shrink_cannot_exceed_one():
    a = parse_advice('{"action":"shrink","size_factor":9.0,"reason":"r"}')
    assert a.apply_to(10) == 10


def test_shrink_reduces():
    a = parse_advice('{"action":"shrink","size_factor":0.5,"reason":"earnings Thursday"}')
    assert a.apply_to(10) == 5


def test_tiny_shrink_is_promoted_to_veto():
    a = parse_advice('{"action":"shrink","size_factor":0.02,"reason":"r"}')
    assert a.action == VETO and a.blocks


def test_shrink_at_the_floor_survives():
    a = parse_advice(f'{{"action":"shrink","size_factor":{MIN_SHRINK},"reason":"r"}}')
    assert a.action == SHRINK


def test_veto_blocks():
    assert parse_advice('{"action":"veto","reason":"earnings in 2 days"}').blocks


def test_veto_without_a_reason_is_downgraded():
    """The shape a hallucinated objection takes."""
    a = parse_advice('{"action":"veto","reason":""}')
    assert a.action == CONFIRM and not a.blocks
    assert "downgraded" in (a.error or "")


# --------------------------------------------------------------------------
# advisor: malformed replies all land safe
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw", [
    "", "   ", "I think you should probably sell everything",
    "{not json}", "[]", "null", '{"action":"buy_more"}',
    '{"action":"confirm","confidence":"very high"}',
    '{"action":"shrink","size_factor":"half","reason":"r"}',
])
def test_malformed_replies_never_block(raw):
    a = parse_advice(raw)
    assert not a.blocks
    assert a.apply_to(10) == 10


def test_json_wrapped_in_prose_is_recovered():
    a = parse_advice('Here is my review:\n{"action":"veto","reason":"halt pending"}\nHope that helps')
    assert a.blocks


def test_unparseable_still_counts_as_consulted():
    """Consulted-and-garbled is a different fact from never-asked."""
    assert parse_advice("garbage").consulted is True


# --------------------------------------------------------------------------
# advisor: absence
# --------------------------------------------------------------------------

def test_no_advisor_is_not_an_approval():
    a = review(None, "AAPL", "buy", 10, 100.0, "ctx")
    assert a.consulted is False and not a.blocks
    assert "not consulted" in a.summary()


def test_advisor_exception_degrades_to_rules_only():
    a = review(NullAdvisor(), "AAPL", "buy", 10, 100.0, "ctx")
    assert a.consulted is False and not a.blocks
    assert a.apply_to(10) == 10


def test_review_passes_through_a_real_veto():
    adv = CannedAdvisor('{"action":"veto","reason":"earnings 2026-09-04"}')
    assert review(adv, "AAPL", "buy", 10, 100.0, "ctx").blocks
    assert adv.calls == 1


def test_build_advisor_none_and_unknown():
    assert build_advisor(AgentSettings(advisor="none")) is None
    assert build_advisor(AgentSettings(advisor="nonsense_vendor")) is None


def test_build_advisor_without_a_key_degrades():
    """No monkeypatching of os.environ: settings are just an object now."""
    assert build_advisor(AgentSettings(advisor="anthropic", api_key="")) is None


# --------------------------------------------------------------------------
# prompt
# --------------------------------------------------------------------------

def test_prompt_excludes_account_state():
    """A reviewer that knows the account is down starts managing the account."""
    p = build_prompt("AAPL", "buy", 10, 100.0, "some context")
    for leak in ("equity", "buying power", "drawdown", "P&L", "portfolio"):
        assert leak.lower() not in p.lower()


def test_prompt_contains_the_trade():
    p = build_prompt("AAPL", "buy", 10, 100.0, "ctx")
    assert "AAPL" in p and "BUY" in p


def test_system_prompt_forbids_generation():
    from trading_bot.agent import SYSTEM_PROMPT
    assert "cannot propose a different trade" in SYSTEM_PROMPT
    assert "confirm" in SYSTEM_PROMPT and "veto" in SYSTEM_PROMPT


# --------------------------------------------------------------------------
# provider-agnostic advisor: response shapes vary, none may crash the cycle
# --------------------------------------------------------------------------

from trading_bot.agent import OpenAICompatibleAdvisor, _extract_text  # noqa: E402


def test_extract_plain_string_content():
    data = {"choices": [{"message": {"content": '{"action":"confirm"}'}}]}
    assert _extract_text(data) == '{"action":"confirm"}'


def test_extract_multipart_content():
    """Some providers return content as a list of parts."""
    data = {"choices": [{"message": {"content": [
        {"type": "text", "text": '{"action":'}, {"type": "text", "text": '"veto"}'}]}}]}
    assert _extract_text(data) == '{"action":"veto"}'


@pytest.mark.parametrize("data", [
    {}, {"choices": []}, {"choices": [{}]}, {"choices": [{"message": {}}]},
    {"choices": [{"message": {"content": None}}]}, {"error": "rate limited"},
    "not a dict", None,
])
def test_extract_tolerates_every_malformed_shape(data):
    assert _extract_text(data) == ""


def test_malformed_provider_response_becomes_a_confirm():
    """End to end: a broken response must not block a trade."""
    assert not parse_advice(_extract_text({"error": "boom"})).blocks


def test_openai_advisor_requires_a_model():
    with pytest.raises(RuntimeError, match="LLM_MODEL"):
        OpenAICompatibleAdvisor(model="")


def test_openai_advisor_works_without_a_key():
    """Local runtimes serve on localhost with no auth."""
    adv = OpenAICompatibleAdvisor(model="llama3", base_url="http://localhost:11434/v1")
    assert adv._model == "llama3" and adv._key == ""


def test_openai_advisor_strips_trailing_slash():
    adv = OpenAICompatibleAdvisor(model="gpt-4o", base_url="https://api.openai.com/v1/")
    assert adv._base == "https://api.openai.com/v1"


def test_build_advisor_accepts_provider_aliases():
    for alias in ("openai", "featherless", "openrouter", "groq", "ollama", "local", "custom"):
        built = build_advisor(AgentSettings(advisor=alias, model="some-model"))
        assert isinstance(built, OpenAICompatibleAdvisor)


def test_build_advisor_without_a_model_degrades():
    assert build_advisor(AgentSettings(advisor="openai", model="")) is None


# --------------------------------------------------------------------------
# settings: one place, and it resolves the provider endpoint itself
# --------------------------------------------------------------------------

def test_settings_default_to_off():
    s = AgentSettings()
    assert s.advisor == "none" and s.fusion is False and not s.advisor_enabled


@pytest.mark.parametrize("kind,expected", [
    ("featherless", "https://api.featherless.ai/v1"),
    ("openai", "https://api.openai.com/v1"),
    ("groq", "https://api.groq.com/openai/v1"),
    ("ollama", "http://localhost:11434/v1"),
])
def test_provider_name_alone_resolves_the_endpoint(monkeypatch, tmp_path, kind, expected):
    """TRADING_BOT_ADVISOR=featherless is enough; no URL to look up."""
    monkeypatch.setenv("TRADING_BOT_ADVISOR", kind)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    assert load_agent_settings(tmp_path / "absent.env").base_url == expected


def test_explicit_base_url_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADING_BOT_ADVISOR", "featherless")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:8000/v1")
    assert load_agent_settings(tmp_path / "absent.env").base_url == "http://localhost:8000/v1"


def test_anthropic_reads_its_own_key_variable(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADING_BOT_ADVISOR", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")
    monkeypatch.setenv("LLM_API_KEY", "other-key")
    s = load_agent_settings(tmp_path / "absent.env")
    assert s.api_key == "ant-key" and s.base_url == ""


def test_non_anthropic_reads_the_shared_llm_variables(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADING_BOT_ADVISOR", "featherless")
    monkeypatch.setenv("LLM_API_KEY", "fw-key")
    monkeypatch.setenv("LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct")
    s = load_agent_settings(tmp_path / "absent.env")
    assert s.api_key == "fw-key" and s.model == "Qwen/Qwen2.5-72B-Instruct"


@pytest.mark.parametrize("raw,expected", [
    ("true", True), ("1", True), ("yes", True), ("on", True),
    ("false", False), ("0", False), ("", False), ("maybe", False),
])
def test_fusion_flag_only_accepts_affirmatives(monkeypatch, tmp_path, raw, expected):
    monkeypatch.setenv("TRADING_BOT_FUSION", raw)
    assert load_agent_settings(tmp_path / "absent.env").fusion is expected


def test_malformed_number_falls_back_instead_of_crashing(monkeypatch, tmp_path):
    """A stray character in an optional knob must not stop the bot starting."""
    monkeypatch.setenv("TRADING_BOT_FUSION_TEMPERATURE", "hot")
    assert load_agent_settings(tmp_path / "absent.env").fusion_temperature == 0.5


def test_redacted_never_exposes_the_key():
    out = AgentSettings(advisor="featherless", api_key="secret-value-here").redacted()
    assert "secret-value-here" not in str(out)
    assert "chars" in out["api_key"]


def test_earnings_is_not_a_valid_alpaca_ca_type():
    """Alpaca's announcements endpoint 422s on `earnings`. Querying it would
    make every lookup fail, which is how the module used to report `checked`
    while having asked for something that does not exist."""
    from trading_bot.retrieval import Retriever
    assert "earnings" not in Retriever.CA_TYPES
    assert set(Retriever.CA_TYPES) == {"dividend", "merger", "spinoff", "split"}


def test_events_uses_the_rest_api_parameter_names():
    """`symbols`/`start`/`end` are silently DROPPED by the server with a warning
    and the query runs against the whole market -- a wrong answer, not an error."""
    r = Retriever(FakeBroker({"corporate_actions": []}))
    r.events("AAPL", today=date(2026, 9, 3))
    ca = [kw for logical, kw in r.broker.calls if logical == "corporate_actions"]
    assert ca, "no corporate_actions call was made"
    for kwargs in ca:
        assert set(kwargs) == {"ca_types", "since", "until", "symbol"}


def test_one_query_per_supported_ca_type():
    r = Retriever(FakeBroker({"corporate_actions": []}))
    r.events("AAPL", today=date(2026, 9, 3))
    types = [kw["ca_types"] for lg, kw in r.broker.calls if lg == "corporate_actions"]
    assert sorted(types) == sorted(Retriever.CA_TYPES)


def test_partial_failure_still_counts_as_checked():
    """If one ca_type errors but others answer, we did look -- reporting
    `checked=False` would throw away real information."""
    class Flaky(FakeBroker):
        def _call(self, logical, **kwargs):
            if kwargs.get("ca_types") == "merger":
                raise RuntimeError("boom")
            return super()._call(logical, **kwargs)

    risk = Retriever(Flaky({"corporate_actions": []})).events("AAPL")
    assert risk.checked is True
