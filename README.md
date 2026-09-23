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

## Quick Start & Usage
Submit a query to the unified endpoint; the router automatically predicts complexity and forwards the payload to the optimal model.

## Testing & CI
Evaluates routing accuracy against established benchmark datasets (e.g., MT Bench, MMLU) to ensure cost savings align with predictions.
