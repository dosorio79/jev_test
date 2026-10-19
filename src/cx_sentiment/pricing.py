from __future__ import annotations

from cx_sentiment.budget import PriceCard

OPENAI_PRICING_SOURCE = "https://developers.openai.com/api/docs/pricing"
JEV_PRICING_SOURCE = "https://docs.typesafe.ai/models"
RATE_CARD_AS_OF = "2026-10-15"

# Standard, short-context text rates in USD per one million tokens.
OPENAI_STANDARD_RATE_CARDS: dict[str, PriceCard] = {
    "gpt-5.4-mini": PriceCard(
        input_usd_per_million=0.75,
        output_usd_per_million=4.5,
        as_of="2026-10-16",
        source_url="https://developers.openai.com/api/docs/models/gpt-5.4-mini",
    ),
    "gpt-6-astra": PriceCard(
        input_usd_per_million=10,
        output_usd_per_million=50,
        as_of=RATE_CARD_AS_OF,
        source_url=OPENAI_PRICING_SOURCE,
    ),
    "gpt-6.1-sol": PriceCard(
        input_usd_per_million=2,
        output_usd_per_million=10,
        as_of=RATE_CARD_AS_OF,
        source_url=OPENAI_PRICING_SOURCE,
    ),
    "gpt-6-luna": PriceCard(
        input_usd_per_million=0.1,
        output_usd_per_million=0.5,
        as_of=RATE_CARD_AS_OF,
        source_url=OPENAI_PRICING_SOURCE,
    ),
}
OPENAI_STANDARD_RATE_CARDS["gpt-5.4-mini-2026-03-17"] = OPENAI_STANDARD_RATE_CARDS["gpt-5.4-mini"]


JEV_TYPESAFE_RATE_CARDS: dict[str, PriceCard] = {
    model: PriceCard(
        input_usd_per_million=0.042,
        output_usd_per_million=0,
        as_of=RATE_CARD_AS_OF,
        source_url=JEV_PRICING_SOURCE,
    )
    for model in ("jev-1.13.0", "jev-1.14.0", "jev-latest", "jev-preview")
}


def get_openai_price_card(model_id: str) -> PriceCard:
    try:
        return OPENAI_STANDARD_RATE_CARDS[model_id]
    except KeyError as exc:
        raise KeyError(
            f"No verified OpenAI rate card for {model_id!r}; add a dated rate card in pricing.py"
        ) from exc


def get_jev_price_card(model_id: str) -> PriceCard:
    try:
        return JEV_TYPESAFE_RATE_CARDS[model_id]
    except KeyError as exc:
        raise KeyError(
            f"No verified TypeSafe Jev rate card for {model_id!r}; "
            "add a dated rate card in pricing.py"
        ) from exc
