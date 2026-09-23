# Multi Model Architecture

## Description
An intelligent, dynamically routing LLM gateway designed to mathematically optimize the performance-cost tradeoff.

## Architecture Overview
Parametric routing classifiers directing queries between frontier and lightweight models based on predicted capability win rates.

## Prerequisites
* Python 3.11+
* `routellm`
* Active API keys for multiple LLM providers.

## Environment Variables
* `STRONG_MODEL_ENDPOINT`
* `WEAK_MODEL_ENDPOINT`
* `ROUTING_CONFIDENCE_THRESHOLD`

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in API keys and model endpoints
```

## Quick Start & Usage
`main.py` is the entry point. Submit a query to the unified endpoint; the
router automatically predicts complexity and forwards the payload to the
optimal model (strong or weak).

```bash
# One-off query
python main.py "What is the capital of France?"

# Override routing behavior per call
python main.py --router bert --threshold 0.3 "Prove that sqrt(2) is irrational."

# Interactive session (maintains conversation history)
python main.py
```

Each response prints which underlying model handled the request and an
estimated cost, based on the per-1k-token prices configured in `.env`.

## Configuration
All configuration lives in `.env` (see `.env.example`):
* `STRONG_MODEL_ENDPOINT` / `WEAK_MODEL_ENDPOINT` — LiteLLM-compatible model identifiers.
* `ROUTER_TYPE` — routing classifier: `mf`, `bert`, `causal_llm`, `sw_ranking`, or `random`.
* `ROUTING_CONFIDENCE_THRESHOLD` — win-rate threshold (0-1) above which the strong model is used.
* `*_PRICE_PER_1K` — optional USD pricing used for cost estimation.
* Provider API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ...) required by LiteLLM for the models you route to.

## Testing & CI
Evaluates routing accuracy against established benchmark datasets (e.g., MT Bench, MMLU) to ensure cost savings align with predictions.
