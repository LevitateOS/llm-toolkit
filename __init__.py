"""
LLM Toolkit - Generic tools for LoRA fine-tuning and inference.

Components:
    - train_lora: Train LoRA adapters on HuggingFace models
    - generate_data: Validate, augment, annotate training data
    - llm_server: HTTP inference server with LoRA support
    - evaluate: Evaluate adapters with test cases
"""

from .llm_server import LLMServer, run_server
from .evaluate import ModelEvaluator, EvalResult, EvalSummary

__all__ = [
    "LLMServer",
    "run_server",
    "ModelEvaluator",
    "EvalResult",
    "EvalSummary",
]
