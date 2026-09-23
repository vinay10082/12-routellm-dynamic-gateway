"""Wraps routellm.Controller to expose a simple route() call."""

from dataclasses import dataclass, field
from typing import Any, Optional

from routellm.controller import Controller

from gateway.config import GatewayConfig


@dataclass
class RouteResult:
    prompt: str
    content: str
    model_used: str
    router_model_id: str
    usage: dict = field(default_factory=dict)


class Gateway:
    """Routes a chat prompt to either the strong or weak model based on
    the predicted win-rate against the configured confidence threshold."""

    def __init__(self, config: GatewayConfig):
        self.config = config
        self._controller = Controller(
            routers=[config.router_type],
            strong_model=config.strong_model,
            weak_model=config.weak_model,
        )

    def route(self, prompt: str, history: Optional[list] = None) -> RouteResult:
        messages = list(history or [])
        messages.append({"role": "user", "content": prompt})

        response = self._controller.chat.completions.create(
            model=self.config.router_model_id,
            messages=messages,
        )

        choice = response.choices[0]
        model_used = getattr(response, "model", None) or self.config.router_model_id
        usage = self._usage_to_dict(getattr(response, "usage", None))

        return RouteResult(
            prompt=prompt,
            content=choice.message.content,
            model_used=model_used,
            router_model_id=self.config.router_model_id,
            usage=usage,
        )

    @staticmethod
    def _usage_to_dict(usage: Any) -> dict:
        if usage is None:
            return {}
        if hasattr(usage, "model_dump"):
            return usage.model_dump()
        if isinstance(usage, dict):
            return usage
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }
