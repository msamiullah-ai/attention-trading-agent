"""The LLM layer: an advisor with subtractive authority only.

THE ONE DESIGN DECISION THAT MATTERS

The model can veto a trade or shrink it. It cannot propose one, cannot enlarge
one, and cannot pick a symbol. Every trade still originates in the strategies
and still passes every deterministic gate before the model is asked anything.

That asymmetry is deliberate, and it is what makes an LLM safe to put in a loop
that spends money. The failure mode of a language model is confident
fabrication -- it will invent a catalyst, misread a date, or find a pattern in
noise, and it will do all three in fluent prose that reads like analysis. If
generation is allowed, a hallucination becomes a position. If only veto is
allowed, the worst a hallucination can do is decline a trade that would have
been fine. Those two error costs are not remotely symmetric, so the authority
should not be either.

Concretely, the worst case in each direction:
  - model wrongly vetoes -> one missed trade, opportunity cost only
  - model wrongly approves -> nothing; approval grants no permission the rule
    layer had not already granted

WHY VETO IS WORTH HAVING AT ALL

Because some facts are only in prose. A delta band cannot read "FDA panel votes
Thursday". A liquidity gate cannot read "company announced a strategic review".
The rule layer is blind to language, and language is where a real fraction of
single-name risk is announced ahead of time. That is the gap this fills, and it
is the only gap it is meant to fill.

WHEN THE MODEL IS UNAVAILABLE, THE BOT TRADES ANYWAY

This looks like failing open, so it is worth being precise about why it is not.
The advisor is not a safety gate. Every safety gate is deterministic, ran
earlier, and already returned approve. The model holds no permission of its own
-- it can only decline to use a veto it was handed. So with the model absent,
the bot degrades to exactly its rules-only behaviour, which is the behaviour it
has in production today.

The alternative -- halt trading when an API call fails -- would hand an outage
at a third party the power to stop the bot, which is a worse property than the
one it protects against. What matters is that absence is RECORDED as absence:
`verdict.consulted is False` is a different fact from an approval, and the two
are never collapsed.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from .logger import get_logger

log = get_logger(__name__)

CONFIRM = "confirm"
VETO = "veto"
SHRINK = "shrink"
_ACTIONS = {CONFIRM, VETO, SHRINK}

# Floor on what a shrink may leave. A model that returns 0.02 has effectively
# vetoed while reporting a confirm, and a 2%-size trade is usually worse than
# no trade -- same commission and spread, none of the payoff. Below this the
# shrink is promoted to an outright veto so the log says what happened.
MIN_SHRINK = 0.25

# Overridable with ANTHROPIC_MODEL. Sonnet is the right default here over a
# larger model: the task is narrow and heavily constrained by the prompt, it
# runs once per candidate trade per cycle, and latency sits directly in the
# order path -- a slower model does not review better, it just delays the fill.
DEFAULT_MODEL = "claude-sonnet-5"

# Sent on every OpenAI-compatible request. See the note in complete().
USER_AGENT = "alpaca-bot/1.0"

SYSTEM_PROMPT = """You are a risk reviewer for an automated trading system. \
You are the last check before an order is sent.

A trade has already been selected by quantitative strategies and has already \
passed every hard risk limit: position sizing, portfolio exposure, correlation, \
drawdown, and liquidity. Those checks are not yours to redo and you do not have \
the data to redo them.

Your only question is: does the news and event context contain a reason NOT to \
place this trade that the numbers could not see?

You have exactly three options:
  confirm  - nothing in the context argues against it
  shrink   - proceed at reduced size, with a stated concern
  veto     - do not place this trade

You cannot propose a different trade, a different symbol, a different direction, \
or a larger size. Those are not available to you and suggesting them has no effect.

Veto or shrink only for a SPECIFIC, DATED, SOURCED fact in the context provided:
  - an earnings print or scheduled binary event inside the holding window
  - a pending acquisition, halt, delisting, offering, or going-concern doubt
  - a regulatory or legal decision with a known date
  - news describing a move that has already happened, where the signal is stale

Do NOT veto for:
  - general market commentary, macro worry, or "uncertainty"
  - a vague sense that valuation is high or sentiment is poor
  - absence of news; no news is a normal condition, not a warning
  - anything you are inferring rather than reading

If you are shown your own past calls on this symbol, use them to avoid
repeating a judgement that was already proven wrong. They are evidence about
your reasoning, not about the trade -- a past mistake is not itself a reason to
veto or to confirm now.

If the context is thin or empty, confirm. Silence is not evidence of danger, \
and a reviewer who vetoes on no information is just a random number generator \
with a vocabulary.

Reply with ONLY a JSON object, no prose around it:
{"action": "confirm|shrink|veto", "size_factor": 1.0, "reason": "<one sentence, \
citing the specific fact>", "confidence": 0.0-1.0}

size_factor is used only when action is "shrink" and must be between 0.25 and \
1.0. For confirm it must be 1.0. Your reason must name the fact you relied on; \
if you cannot name one, the action is confirm."""


@dataclass
class Advice:
    """What the advisor concluded, and whether it was even asked."""

    action: str = CONFIRM
    size_factor: float = 1.0
    reason: str = ""
    confidence: float = 0.0
    consulted: bool = False          # False = no model ran; NOT the same as approval
    error: str | None = None

    @property
    def blocks(self) -> bool:
        return self.action == VETO

    def apply_to(self, qty: int) -> int:
        """Scale an order quantity. Never increases it -- the clamp is not
        defensive styling, it is the enforcement point for the whole design."""
        if self.action != SHRINK:
            return qty
        factor = max(MIN_SHRINK, min(1.0, self.size_factor))
        return max(0, int(qty * factor))

    def summary(self) -> str:
        if not self.consulted:
            return f"advisor not consulted ({self.error or 'disabled'}); rules decision stands"
        if self.action == CONFIRM:
            return "advisor confirmed"
        return f"advisor {self.action} ({self.size_factor:.2f}): {self.reason}"


class Advisor(Protocol):
    """Anything that can answer a review prompt. Keeps the model vendor out of
    the trading path, and lets tests pass a canned responder."""

    def complete(self, system: str, user: str) -> str: ...


class NullAdvisor:
    """The default. Abstains, loudly enough to show up in the log."""

    def complete(self, system: str, user: str) -> str:
        raise RuntimeError("no advisor configured")


class AnthropicAdvisor:
    """Optional. Imported lazily so the package never hard-depends on it.

    The key is read from the environment and is never logged, never written to
    a cycle report, and never included in a prompt.
    """

    def __init__(self, api_key: str, model: str = "", max_tokens: int = 300,
                 timeout: float = 20.0):
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError("anthropic package not installed") from exc
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
        self._model = model or DEFAULT_MODEL
        self._max_tokens = max_tokens
        log.info("advisor ready | model=%s timeout=%.0fs", self._model, timeout)

    def complete(self, system: str, user: str) -> str:
        resp = self._client.messages.create(
            model=self._model, max_tokens=self._max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


def _extract_text(data: Any) -> str:
    """Pull the assistant's text out of an OpenAI-shaped response.

    Kept separate from the HTTP call so the response shapes can be tested
    without a network, which is where the provider-to-provider variation
    actually lives: `content` is a plain string for most, a list of parts for
    some, and occasionally null when a model returns only a refusal.
    """
    if not isinstance(data, dict):
        return ""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return _strip_reasoning(content)
    if isinstance(content, list):
        # Multi-part content: concatenate the text parts, ignore the rest.
        return _strip_reasoning(
            "".join(p.get("text", "") for p in content if isinstance(p, dict)))
    return ""


def _strip_reasoning(text: str) -> str:
    """Remove inline <think> blocks.

    Reasoning models split two ways depending on the server: some return the
    chain of thought in a separate `reasoning` field, which never reaches here,
    and some inline it in the content wrapped in <think> tags. The second kind
    would otherwise defeat the JSON parse, since the object arrives after a
    paragraph of deliberation.
    """
    if "<think>" not in text:
        return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class OpenAICompatibleAdvisor:
    """Any server that speaks OpenAI's /v1/chat/completions.

    That is deliberately almost everything: OpenAI, OpenRouter, Groq, DeepSeek,
    Together, Fireworks, Gemini's compatibility endpoint, and local runtimes
    like Ollama, LM Studio and vLLM all accept this same request shape. Swapping
    provider is a base_url and a model name, not a code change.

    Uses urllib from the standard library on purpose. This is an optional
    feature on a path that must degrade quietly, and making the whole bot carry
    an HTTP client dependency for a component most deployments will not enable
    is a poor trade.
    """

    def __init__(self, model: str, base_url: str = "", api_key: str = "",
                 max_tokens: int = 2048, timeout: float = 60.0):
        if not model:
            raise RuntimeError("LLM_MODEL not set")
        self._base = (base_url or "https://api.openai.com/v1").rstrip("/")
        self._model = model
        # No key required: local runtimes serve on localhost without auth, and
        # sending an empty Authorization header makes some of them 401.
        self._key = api_key
        self._max_tokens = max_tokens
        self._timeout = timeout
        log.info("advisor ready | model=%s endpoint=%s", self._model, self._base)

    def complete(self, system: str, user: str) -> str:
        import urllib.request

        payload = json.dumps({
            "model": self._model,
            "max_tokens": self._max_tokens,
            # Zero temperature: this is a risk check, and the same trade with
            # the same context should get the same answer twice. A reviewer
            # that is stochastic is a reviewer you cannot audit.
            "temperature": 0,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }).encode()

        # An explicit User-Agent is required, not cosmetic. urllib defaults to
        # "Python-urllib/3.x", which Cloudflare-fronted providers block by
        # fingerprint -- Featherless returns HTTP 403 "error code: 1010" before
        # the request ever reaches the API, so a perfectly good key and model
        # look like an auth failure.
        headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"

        request = urllib.request.Request(
            f"{self._base}/chat/completions", data=payload, headers=headers)
        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            data = json.loads(response.read().decode())

        text = _extract_text(data)
        if not text.strip():
            # Distinguish "ran out of budget while thinking" from "said nothing".
            # Both arrive as empty content and both end up as a confirm, but only
            # one of them is fixed by raising max_tokens, and a caller staring at
            # "empty response" has no way to know which they have.
            choice = (data.get("choices") or [{}])[0]
            if choice.get("finish_reason") == "length":
                raise RuntimeError(
                    f"reply truncated at max_tokens={self._max_tokens} with no "
                    f"content; this model reasons before answering -- raise "
                    f"LLM_MAX_TOKENS or use a non-reasoning model")
        return text


def build_prompt(symbol: str, side: str, qty: int, price: float,
                 context: str, verdict_line: str = "", history: str = "") -> str:
    """The user-turn half of the review.

    Deliberately excludes account equity, buying power and total P&L. The
    reviewer's job does not require them, and a model that knows the account is
    down on the day tends to start managing the account -- getting protective or
    trying to make it back -- which is precisely the judgement we are not asking
    for and have not gated.
    """
    lines = [
        f"PROPOSED TRADE: {side.upper()} {qty} {symbol} at ~{price:.2f}",
        f"NOTIONAL: ~${qty * price:,.0f}",
    ]
    if verdict_line:
        lines.append(f"SIGNAL: {verdict_line}")
    lines += ["", "RETRIEVED CONTEXT:", context or "(nothing retrieved)", "",
              "Review this trade."]
    return "\n".join(lines)


def parse_advice(raw: str) -> Advice:
    """Parse the model's reply, treating anything unparseable as a confirm.

    Unparseable means the reviewer did not manage to state an objection, and a
    garbled response is not an objection. Failing to a veto here would let a
    truncation or a stray sentence silently halt trading, which is a much worse
    outcome than ignoring one malformed review.
    """
    if not raw or not raw.strip():
        return Advice(consulted=True, error="empty response")

    text = raw.strip()
    if not text.startswith("{"):
        m = re.search(r"\{.*\}", text, re.DOTALL)   # models like to add a preamble
        text = m.group(0) if m else ""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return Advice(consulted=True, error="unparseable response")
    if not isinstance(data, dict):
        return Advice(consulted=True, error="response was not an object")

    action = str(data.get("action", CONFIRM)).strip().lower()
    if action not in _ACTIONS:
        return Advice(consulted=True, error=f"unknown action {action!r}")

    reason = str(data.get("reason", "")).strip()[:300]
    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    try:
        factor = float(data.get("size_factor", 1.0))
    except (TypeError, ValueError):
        factor = 1.0

    # A veto with no stated fact is exactly what the prompt forbids, and it is
    # the shape a hallucinated objection takes. Downgraded rather than obeyed.
    if action == VETO and not reason:
        return Advice(action=CONFIRM, consulted=True, confidence=confidence,
                      error="veto without a stated reason; downgraded")

    if action == SHRINK:
        factor = max(0.0, min(1.0, factor))
        if factor < MIN_SHRINK:
            # Promoted, not clamped: a 10% trade is not a small trade, it is a
            # refusal wearing a confirm's clothing, and the log should say veto.
            return Advice(action=VETO, size_factor=0.0, reason=reason or
                          "size reduced below the useful floor", confidence=confidence,
                          consulted=True)
    else:
        factor = 1.0

    return Advice(action=action, size_factor=factor, reason=reason,
                  confidence=confidence, consulted=True)


def review(advisor: Advisor | None, symbol: str, side: str, qty: int, price: float,
           context: str, verdict_line: str = "", history: str = "") -> Advice:
    """Ask for a review. Never raises -- see the module docstring on absence."""
    if advisor is None:
        return Advice(error="disabled")
    try:
        raw = advisor.complete(
            SYSTEM_PROMPT,
            build_prompt(symbol, side, qty, price, context, verdict_line, history))
    except Exception as exc:  # noqa: BLE001 - an outage must not stop the bot
        log.warning("advisor unavailable for %s: %s", symbol, exc)
        return Advice(error=str(exc)[:200])

    advice = parse_advice(raw)
    if advice.blocks:
        log.info("ADVISOR VETO %s %s: %s", side, symbol, advice.reason)
    elif advice.action == SHRINK:
        log.info("ADVISOR SHRINK %s %s to %.0f%%: %s",
                 side, symbol, advice.size_factor * 100, advice.reason)
    return advice


def build_advisor(settings=None, **kwargs: Any) -> Advisor | None:
    """Build the advisor described by an AgentSettings.

    Returns None rather than raising whenever one cannot be built, so a missing
    key or an unknown provider degrades to rules-only instead of refusing to
    start. Takes the settings object rather than reading the environment: this
    module should not know that .env exists.
    """
    if settings is None:
        from .config import load_agent_settings
        settings = load_agent_settings()
    kind = (settings.advisor or "none").strip().lower()
    if not settings.advisor_enabled:
        return None
    builders = {
        "anthropic": AnthropicAdvisor,
        # All the same class -- the aliases just spare anyone the guess about
        # which word this config wanted.
        "openai": OpenAICompatibleAdvisor,
        "openai-compatible": OpenAICompatibleAdvisor,
        "openrouter": OpenAICompatibleAdvisor,
        "featherless": OpenAICompatibleAdvisor,
        "groq": OpenAICompatibleAdvisor,
        "ollama": OpenAICompatibleAdvisor,
        "local": OpenAICompatibleAdvisor,
        "custom": OpenAICompatibleAdvisor,
    }
    builder = builders.get(kind)
    if builder is None:
        log.warning("unknown advisor %r; running rules-only", kind)
        return None

    args: dict[str, Any] = {"model": settings.model, "timeout": settings.timeout,
                            "max_tokens": settings.max_tokens}
    if builder is AnthropicAdvisor:
        args["api_key"] = settings.api_key
    else:
        args["api_key"] = settings.api_key
        args["base_url"] = settings.base_url
    args.update(kwargs)
    try:
        return builder(**args)
    except Exception as exc:  # noqa: BLE001 - a missing key must not stop startup
        log.warning("advisor disabled: %s", exc)
        return None
