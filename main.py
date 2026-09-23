"""Entry point for the RouteLLM Dynamic Gateway.

Submits a query to the unified endpoint; the router predicts query
complexity and forwards the payload to the optimal (strong or weak) model.

Usage:
    python main.py "What is the capital of France?"
    python main.py --router bert --threshold 0.3 "Prove that sqrt(2) is irrational."
    python main.py                      # interactive REPL
"""

import argparse
import sys
from dataclasses import replace

from gateway.config import ConfigError, load_config
from gateway.cost import CostTracker
from gateway.router import Gateway


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="routellm-gateway",
        description=(
            "Dynamically route a prompt to a strong or weak LLM based on "
            "predicted query complexity."
        ),
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Prompt to send. Omit to start an interactive session.",
    )
    parser.add_argument(
        "--router",
        dest="router_type",
        choices=["mf", "bert", "causal_llm", "sw_ranking", "random"],
        help="Override ROUTER_TYPE.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Override ROUTING_CONFIDENCE_THRESHOLD (0-1).",
    )
    parser.add_argument("--strong", dest="strong_model", help="Override STRONG_MODEL_ENDPOINT.")
    parser.add_argument("--weak", dest="weak_model", help="Override WEAK_MODEL_ENDPOINT.")
    return parser


def apply_overrides(config, args):
    overrides = {}
    if args.router_type:
        overrides["router_type"] = args.router_type
    if args.threshold is not None:
        if not 0.0 <= args.threshold <= 1.0:
            raise ConfigError("--threshold must be between 0 and 1.")
        overrides["threshold"] = args.threshold
    if args.strong_model:
        overrides["strong_model"] = args.strong_model
    if args.weak_model:
        overrides["weak_model"] = args.weak_model
    return replace(config, **overrides) if overrides else config


def run_once(gateway: Gateway, cost_tracker: CostTracker, prompt: str, history=None):
    result = gateway.route(prompt, history=history)
    cost = cost_tracker.record(result.model_used, result.usage)

    print(f"\n[router: {result.router_model_id}] -> routed to: {result.model_used}")
    print(result.content)
    print(f"(est. cost: ${cost:.5f} | session total: ${cost_tracker.total:.5f})")
    return result


def interactive_loop(gateway: Gateway, cost_tracker: CostTracker):
    print("RouteLLM Dynamic Gateway - interactive mode. Type 'exit' or Ctrl+C to quit.\n")
    history: list = []
    while True:
        try:
            prompt = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            print(cost_tracker.summary())
            break

        result = run_once(gateway, cost_tracker, prompt, history=history)
        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": result.content})


def main():
    args = build_arg_parser().parse_args()

    try:
        config = load_config()
        config = apply_overrides(config, args)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)

    gateway = Gateway(config)
    cost_tracker = CostTracker(config)

    if args.prompt:
        run_once(gateway, cost_tracker, args.prompt)
    else:
        interactive_loop(gateway, cost_tracker)


if __name__ == "__main__":
    main()
