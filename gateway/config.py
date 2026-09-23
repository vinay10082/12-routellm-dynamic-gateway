"""Environment-driven configuration for the RouteLLM dynamic gateway."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

DEFAULT_ROUTER_TYPE = "mf"
DEFAULT_THRESHOLD = 0.11593
VALID_ROUTER_TYPES = {"mf", "bert", "causal_llm", "sw_ranking", "random"}


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class GatewayConfig:
    strong_model: str
    weak_model: str
    threshold: float
    router_type: str
    strong_prompt_price_per_1k: float
    strong_completion_price_per_1k: float
    weak_prompt_price_per_1k: float
    weak_completion_price_per_1k: float

    @property
    def router_model_id(self) -> str:
        return f"router-{self.router_type}-{self.threshold}"


def _get_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def load_config() -> GatewayConfig:
    strong_model = os.getenv("STRONG_MODEL_ENDPOINT")
    weak_model = os.getenv("WEAK_MODEL_ENDPOINT")

    if not strong_model:
        raise ConfigError("STRONG_MODEL_ENDPOINT is not set. Check your .env file.")
    if not weak_model:
        raise ConfigError("WEAK_MODEL_ENDPOINT is not set. Check your .env file.")

    threshold = _get_float_env("ROUTING_CONFIDENCE_THRESHOLD", DEFAULT_THRESHOLD)
    if not 0.0 <= threshold <= 1.0:
        raise ConfigError("ROUTING_CONFIDENCE_THRESHOLD must be between 0 and 1.")

    router_type = os.getenv("ROUTER_TYPE", DEFAULT_ROUTER_TYPE)
    if router_type not in VALID_ROUTER_TYPES:
        raise ConfigError(
            f"ROUTER_TYPE must be one of {sorted(VALID_ROUTER_TYPES)}, got {router_type!r}"
        )

    return GatewayConfig(
        strong_model=strong_model,
        weak_model=weak_model,
        threshold=threshold,
        router_type=router_type,
        strong_prompt_price_per_1k=_get_float_env("STRONG_MODEL_PROMPT_PRICE_PER_1K", 0.0),
        strong_completion_price_per_1k=_get_float_env("STRONG_MODEL_COMPLETION_PRICE_PER_1K", 0.0),
        weak_prompt_price_per_1k=_get_float_env("WEAK_MODEL_PROMPT_PRICE_PER_1K", 0.0),
        weak_completion_price_per_1k=_get_float_env("WEAK_MODEL_COMPLETION_PRICE_PER_1K", 0.0),
    )
