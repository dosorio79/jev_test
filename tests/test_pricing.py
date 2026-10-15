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
