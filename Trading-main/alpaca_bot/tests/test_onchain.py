"""On-chain filter tests — network calls are mocked, no real access needed."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_bot.onchain import (  # noqa: E402
    FearGreedReading,
    fetch_fear_greed,
    position_size_multiplier,
)


def _fake_response(value: str, classification: str) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"data": [{"value": value, "value_classification": classification}]}
    return resp


def test_fetch_fear_greed_parses_a_successful_response():
    with patch("trading_bot.onchain.requests.get", return_value=_fake_response("28", "Fear")) as mock_get:
        result = fetch_fear_greed()
    assert result == FearGreedReading(value=28, classification="Fear")
    mock_get.assert_called_once()


def test_fetch_fear_greed_degrades_gracefully_on_network_error():
    with patch("trading_bot.onchain.requests.get", side_effect=ConnectionError("no route")):
        result = fetch_fear_greed()
    assert result is None


def test_fetch_fear_greed_degrades_gracefully_on_bad_status():
    resp = MagicMock()
    resp.raise_for_status.side_effect = Exception("500 Server Error")
    with patch("trading_bot.onchain.requests.get", return_value=resp):
        result = fetch_fear_greed()
    assert result is None


def test_fetch_fear_greed_degrades_gracefully_on_malformed_json():
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"unexpected": "shape"}
    with patch("trading_bot.onchain.requests.get", return_value=resp):
        result = fetch_fear_greed()
    assert result is None


def test_position_size_multiplier_none_reading_is_neutral():
    assert position_size_multiplier(None) == 1.0


def test_position_size_multiplier_boosts_on_extreme_fear():
    reading = FearGreedReading(value=15, classification="Extreme Fear")
    assert position_size_multiplier(reading) == 1.2


def test_position_size_multiplier_cuts_on_extreme_greed():
    reading = FearGreedReading(value=90, classification="Extreme Greed")
    assert position_size_multiplier(reading) == 0.5


def test_position_size_multiplier_neutral_in_the_middle():
    reading = FearGreedReading(value=50, classification="Neutral")
    assert position_size_multiplier(reading) == 1.0


def test_position_size_multiplier_boundary_values_are_inclusive():
    assert position_size_multiplier(FearGreedReading(20, "Fear")) == 1.2
    assert position_size_multiplier(FearGreedReading(80, "Greed")) == 0.5
    assert position_size_multiplier(FearGreedReading(21, "Fear")) == 1.0
    assert position_size_multiplier(FearGreedReading(79, "Greed")) == 1.0
