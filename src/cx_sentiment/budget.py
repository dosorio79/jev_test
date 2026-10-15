from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BudgetExceeded(RuntimeError):
    """A request would cross a configured spend or request limit."""


class PriceCard(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_usd_per_million: float = Field(ge=0)
    output_usd_per_million: float = Field(ge=0)
    as_of: str
    source_url: str | None = None

    def estimate(self, *, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_usd_per_million + output_tokens * self.output_usd_per_million
        ) / 1_000_000


class LiveSafety(BaseModel):
    live: bool = False
    max_requests: int | None = Field(default=None, ge=1)
    max_estimated_cost_usd: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def require_caps_for_live(self) -> LiveSafety:
        if self.live and self.max_requests is None:
            raise ValueError("max_requests is required for live mode")
        if self.live and self.max_estimated_cost_usd is None:
            raise ValueError("a positive budget is required for live mode")
        return self

    def assert_can_call(
        self,
        requests_made: int,
        estimated_spend_usd: float,
        next_upper_bound: float = 0,
    ) -> None:
        if not self.live:
            raise ValueError("network calls require explicit --live")
        assert self.max_requests is not None
        assert self.max_estimated_cost_usd is not None
        if requests_made >= self.max_requests:
            raise BudgetExceeded("request limit reached")
        if estimated_spend_usd + next_upper_bound > self.max_estimated_cost_usd:
            raise BudgetExceeded("estimated USD budget would be exceeded")
