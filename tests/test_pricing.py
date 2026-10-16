from __future__ import annotations

import pytest

from cx_sentiment.pricing import (
    JEV_TYPESAFE_RATE_CARDS,
    OPENAI_STANDARD_RATE_CARDS,
    get_jev_price_card,
    get_openai_price_card,
)


def test_openai_rate_table_uses_exact_model_ids_and_token_rates() -> None:
    card = get_openai_price_card("gpt-6-luna")

    assert card is OPENAI_STANDARD_RATE_CARDS["gpt-6-luna"]
    assert card.estimate(input_tokens=1_000_000, output_tokens=1_000_000) == pytest.approx(0.6)
    with pytest.raises(KeyError, match="No verified OpenAI rate card"):
        get_openai_price_card("unknown-model")


def test_jev_typesafe_rate_table_charges_input_tokens_and_free_output() -> None:
    card = get_jev_price_card("jev-1.13.0")
    assert card is JEV_TYPESAFE_RATE_CARDS["jev-1.13.0"]
    assert card.estimate(input_tokens=1_000_000, output_tokens=50_000) == pytest.approx(0.042)


@pytest.mark.parametrize("legacy_prices", ["absent", "partial", "complete"])
def test_cli_uses_model_rate_card_without_environment_prices(
    monkeypatch, legacy_prices: str
) -> None:
    from cx_sentiment.cli import _provider_from_environment
    from cx_sentiment.providers.mock import MockProvider

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-6-luna")
    monkeypatch.delenv("OPENAI_INPUT_USD_PER_MILLION", raising=False)
    monkeypatch.delenv("OPENAI_OUTPUT_USD_PER_MILLION", raising=False)
    monkeypatch.delenv("OPENAI_PRICE_AS_OF", raising=False)
    if legacy_prices != "absent":
        monkeypatch.setenv("OPENAI_INPUT_USD_PER_MILLION", "999")
    if legacy_prices == "complete":
        monkeypatch.setenv("OPENAI_OUTPUT_USD_PER_MILLION", "999")
        monkeypatch.setenv("OPENAI_PRICE_AS_OF", "2000-01-01")
    captured = {}

    def create_provider(**kwargs):
        captured.update(kwargs)
        return MockProvider()

    monkeypatch.setattr("cx_sentiment.cli.OpenAIProvider", create_provider)
    _provider_from_environment("openai")

    assert captured["model"] == "gpt-6-luna"
    assert captured["price_card"] is get_openai_price_card("gpt-6-luna")
