"""Tracks estimated cost savings from dynamic routing."""

from dataclasses import dataclass

from gateway.config import GatewayConfig


@dataclass
class UsageRecord:
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost: float


class CostTracker:
    """Estimates USD cost per call using the price-per-1k-token rates
    configured in the environment, and keeps a running session total."""

    def __init__(self, config: GatewayConfig):
        self.config = config
        self.total: float = 0.0
        self.strong_calls: int = 0
        self.weak_calls: int = 0
        self.history: list[UsageRecord] = []

    def _rates_for(self, model_used: str) -> tuple[float, float]:
        if model_used == self.config.strong_model:
            return (
                self.config.strong_prompt_price_per_1k,
                self.config.strong_completion_price_per_1k,
            )
        return (
            self.config.weak_prompt_price_per_1k,
            self.config.weak_completion_price_per_1k,
        )

    def record(self, model_used: str, usage: dict) -> float:
        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        completion_tokens = usage.get("completion_tokens", 0) or 0

        prompt_rate, completion_rate = self._rates_for(model_used)
        cost = (prompt_tokens / 1000) * prompt_rate + (completion_tokens / 1000) * completion_rate

        self.total += cost
        if model_used == self.config.strong_model:
            self.strong_calls += 1
        else:
            self.weak_calls += 1

        self.history.append(
            UsageRecord(
                model=model_used,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost=cost,
            )
        )
        return cost

    def summary(self) -> str:
        total_calls = self.strong_calls + self.weak_calls
        weak_share = (self.weak_calls / total_calls * 100) if total_calls else 0.0
        return (
            f"calls: {total_calls} (strong: {self.strong_calls}, weak: {self.weak_calls}, "
            f"{weak_share:.1f}% routed to weak) | total est. cost: ${self.total:.5f}"
        )
