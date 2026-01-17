#!/usr/bin/env python3
"""
Generic LoRA adapter evaluator with multi-turn conversation test cases.

Evaluates trained adapters on custom test cases loaded from JSONL files.
Designed to be domain-agnostic - supply your own test cases and prompts.

Usage:
    python evaluate.py --model path/to/base --test-cases tests.jsonl
    python evaluate.py --adapter adapters/mymodel --test-cases tests.jsonl
    python evaluate.py --sweep-dir adapters/ --test-cases tests.jsonl

Test case format (JSONL):
    {"name": "test_name", "messages": [{"role": "user", "content": "..."}],
     "expected": "pattern" | ["p1", "p2"] | null, "category": "optional"}

    - expected: string/list = command must contain pattern(s)
    - expected: null = should produce text response, not command
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


ExpectedPattern = Optional[str | list[str]]


def matches_pattern(got_cmd: Optional[str], expected: ExpectedPattern) -> bool:
    """Check if generated command matches expected pattern(s)."""
    if got_cmd is None:
        return expected is None
    if expected is None:
        return False  # Expected text, got command

    if isinstance(expected, str):
        return expected.lower() in got_cmd.lower()
    elif isinstance(expected, list):
        return all(p.lower() in got_cmd.lower() for p in expected)
    return False


def pattern_to_str(expected: ExpectedPattern) -> str:
    """Convert pattern to display string."""
    if expected is None:
        return "(text response)"
    if isinstance(expected, list):
        return " & ".join(expected)
    return expected


@dataclass
class EvalResult:
    """Result of evaluating a single test case."""
    name: str
    messages: list
    expected: ExpectedPattern
    got_command: Optional[str]
    got_text: Optional[str]
    correct: bool
    category: str = ""
    error: Optional[str] = None


@dataclass
class EvalSummary:
    """Summary of evaluation results."""
    adapter_path: str
    total: int = 0
    correct: int = 0
    by_category: dict = field(default_factory=dict)
    results: list = field(default_factory=list)

    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "adapter_path": self.adapter_path,
            "total": self.total,
            "correct": self.correct,
            "accuracy": self.accuracy(),
            "by_category": self.by_category,
        }


class ModelEvaluator:
    """Evaluates a model/adapter on conversation test cases."""

    def __init__(
        self,
        model_path: str,
        adapter_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        tool_schema: Optional[dict] = None,
        device: str = "auto",
        dtype: str = "auto",
    ):
        print(f"Loading model from {model_path}...", file=sys.stderr)

        # Determine dtype
        if dtype == "auto":
            torch_dtype = "auto"
        elif dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device if device != "cpu" else None,
        )

        if device == "cpu":
            self.model = self.model.to("cpu")

        if adapter_path and Path(adapter_path).exists():
            if not PEFT_AVAILABLE:
                raise RuntimeError("peft not installed, cannot load adapter")
            print(f"Loading adapter from {adapter_path}...", file=sys.stderr)
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.device = next(self.model.parameters()).device
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.tool_schema = tool_schema
        print(f"Model loaded on {self.device}", file=sys.stderr)

    def generate(self, conversation: list[dict]) -> tuple[Optional[str], Optional[str]]:
        """
        Generate response for a conversation.
        Returns: (command, text) - one will be None
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(conversation)

        # Apply chat template
        template_kwargs = {
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        if self.tool_schema:
            template_kwargs["tools"] = [self.tool_schema]

        inputs = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        raw_output = self.tokenizer.decode(generated_ids, skip_special_tokens=False)

        # Try to extract tool call (common patterns)
        # Pattern 1: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        tool_call_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', raw_output, re.DOTALL)
        if tool_call_match:
            try:
                call = json.loads(tool_call_match.group(1))
                if "arguments" in call and "command" in call["arguments"]:
                    return call["arguments"]["command"], None
            except json.JSONDecodeError:
                pass

        # Pattern 2: SmolLM3 style - call:function_name{param:value}
        smol_match = re.search(r'call:(\w+)\{command:<escape>(.*?)<escape>\}', raw_output, re.DOTALL)
        if smol_match:
            return smol_match.group(2).strip(), None

        # Pattern 3: Assistant content with ```bash blocks
        bash_match = re.search(r'```(?:bash|sh)?\s*\n?(.*?)\n?```', raw_output, re.DOTALL)
        if bash_match:
            return bash_match.group(1).strip(), None

        # No tool call found - return as text
        text = re.sub(r'<[^>]+>', '', raw_output).strip()
        return None, text if text else None

    def evaluate(self, test_cases: list[dict]) -> EvalSummary:
        """Evaluate on all test cases."""
        summary = EvalSummary(adapter_path="")
        summary.total = len(test_cases)

        for tc in test_cases:
            name = tc.get("name", "unnamed")
            messages = tc["messages"]
            expected = tc.get("expected")
            category = tc.get("category", "other")

            try:
                got_cmd, got_text = self.generate(messages)

                if expected is None:
                    # Expected text response
                    correct = got_cmd is None and got_text is not None and len(got_text) > 0
                else:
                    # Expected command
                    correct = matches_pattern(got_cmd, expected)

                if correct:
                    summary.correct += 1

                # Track by category
                if category not in summary.by_category:
                    summary.by_category[category] = {"total": 0, "correct": 0}
                summary.by_category[category]["total"] += 1
                if correct:
                    summary.by_category[category]["correct"] += 1

                summary.results.append(EvalResult(
                    name=name,
                    messages=messages,
                    expected=expected,
                    got_command=got_cmd,
                    got_text=got_text[:100] if got_text else None,
                    correct=correct,
                    category=category,
                ))

            except Exception as e:
                summary.results.append(EvalResult(
                    name=name,
                    messages=messages,
                    expected=expected,
                    got_command=None,
                    got_text=None,
                    correct=False,
                    category=category,
                    error=str(e),
                ))

        return summary


def load_test_cases(path: Path) -> list[dict]:
    """Load test cases from JSONL file."""
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                cases.append(json.loads(line))
    return cases


def load_system_prompt(path: Path) -> str:
    """Load system prompt from file."""
    return path.read_text().strip()


def load_tool_schema(path: Path) -> dict:
    """Load tool schema from JSON file."""
    with open(path) as f:
        return json.load(f)


def print_summary(summary: EvalSummary, verbose: bool = False):
    """Print evaluation summary."""
    print(f"\n{'=' * 60}")
    print(f"Adapter: {summary.adapter_path}")
    print(f"{'=' * 60}")
    print(f"  Total tests:  {summary.total}")
    print(f"  Correct:      {summary.correct} ({100 * summary.accuracy():.1f}%)")

    if summary.by_category:
        print(f"\n  By category:")
        for cat, stats in sorted(summary.by_category.items()):
            pct = 100 * stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            print(f"    {cat:15} {stats['correct']}/{stats['total']} ({pct:.0f}%)")

    if verbose:
        print(f"\n  Detailed results:")
        for r in summary.results:
            status = "PASS" if r.correct else "FAIL"
            expected_str = pattern_to_str(r.expected)

            print(f"\n    [{status}] {r.name} ({r.category})")
            print(f"         Messages: {len(r.messages)} turn(s)")
            if r.messages:
                last_msg = r.messages[-1]["content"]
                print(f"         Last user: {last_msg[:40]}{'...' if len(last_msg) > 40 else ''}")
            print(f"         Expected: {expected_str}")
            if r.got_command:
                cmd_display = r.got_command[:60]
                print(f"         Got cmd:  {cmd_display}{'...' if len(r.got_command) > 60 else ''}")
            elif r.got_text:
                print(f"         Got text: {r.got_text[:60]}...")
            if r.error:
                print(f"         Error: {r.error}")


def evaluate_adapter(
    model_path: Path,
    adapter_path: Optional[Path],
    test_cases: list[dict],
    system_prompt: Optional[str] = None,
    tool_schema: Optional[dict] = None,
    device: str = "auto",
    dtype: str = "auto",
) -> EvalSummary:
    """Evaluate a single adapter."""
    evaluator = ModelEvaluator(
        str(model_path),
        str(adapter_path) if adapter_path else None,
        system_prompt=system_prompt,
        tool_schema=tool_schema,
        device=device,
        dtype=dtype,
    )
    summary = evaluator.evaluate(test_cases)
    summary.adapter_path = str(adapter_path) if adapter_path else "base_model"
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LoRA adapters with custom conversation tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate base model
  python evaluate.py --model ./model --test-cases tests.jsonl

  # Evaluate adapter
  python evaluate.py --model ./model --adapter ./adapter --test-cases tests.jsonl

  # Evaluate all adapters in directory
  python evaluate.py --model ./model --sweep-dir ./adapters --test-cases tests.jsonl

Test case format (JSONL):
  {"name": "test1", "messages": [{"role": "user", "content": "hello"}], "expected": null}
  {"name": "test2", "messages": [{"role": "user", "content": "run ls"}], "expected": "ls"}
        """,
    )
    parser.add_argument("--model", "-m", required=True, help="Base model path")
    parser.add_argument("--adapter", "-a", default=None, help="Single adapter to evaluate")
    parser.add_argument("--sweep-dir", "-s", default=None, help="Directory containing multiple adapters")
    parser.add_argument("--test-cases", "-t", required=True, help="JSONL file with test cases")
    parser.add_argument("--system-prompt", "-p", default=None, help="System prompt file")
    parser.add_argument("--tool-schema", default=None, help="Tool schema JSON file")
    parser.add_argument("--output", "-o", default=None, help="Output JSON file for results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed results")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device to use")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"], help="Data type")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}", file=sys.stderr)
        sys.exit(1)

    test_cases_path = Path(args.test_cases).resolve()
    if not test_cases_path.exists():
        print(f"Error: Test cases not found at {test_cases_path}", file=sys.stderr)
        sys.exit(1)

    test_cases = load_test_cases(test_cases_path)
    print(f"Loaded {len(test_cases)} test cases from {test_cases_path}", file=sys.stderr)

    system_prompt = None
    if args.system_prompt:
        system_prompt = load_system_prompt(Path(args.system_prompt))

    tool_schema = None
    if args.tool_schema:
        tool_schema = load_tool_schema(Path(args.tool_schema))

    results = []

    if args.sweep_dir:
        sweep_dir = Path(args.sweep_dir).resolve()
        adapter_dirs = [
            d for d in sweep_dir.iterdir()
            if d.is_dir() and (d / "adapter_config.json").exists()
        ]

        if not adapter_dirs:
            print(f"No adapters found in {sweep_dir}", file=sys.stderr)
            sys.exit(1)

        print(f"Found {len(adapter_dirs)} adapters to evaluate")

        for i, adapter_dir in enumerate(sorted(adapter_dirs), 1):
            print(f"\n[{i}/{len(adapter_dirs)}] Evaluating {adapter_dir.name}...")
            try:
                summary = evaluate_adapter(
                    model_path, adapter_dir, test_cases,
                    system_prompt=system_prompt,
                    tool_schema=tool_schema,
                    device=args.device,
                    dtype=args.dtype,
                )
                print_summary(summary, args.verbose)
                results.append(summary.to_dict())
            except Exception as e:
                print(f"  Error: {e}")
                results.append({"adapter_path": str(adapter_dir), "error": str(e)})

        # Rank by accuracy
        valid_results = [r for r in results if "accuracy" in r]
        if valid_results:
            valid_results.sort(key=lambda x: x["accuracy"], reverse=True)
            print("\n" + "=" * 60)
            print("  RANKING")
            print("=" * 60)
            for i, r in enumerate(valid_results[:10], 1):
                print(f"  {i}. {Path(r['adapter_path']).name}: {100 * r['accuracy']:.1f}%")

    elif args.adapter:
        adapter_path = Path(args.adapter).resolve()
        summary = evaluate_adapter(
            model_path, adapter_path, test_cases,
            system_prompt=system_prompt,
            tool_schema=tool_schema,
            device=args.device,
            dtype=args.dtype,
        )
        print_summary(summary, args.verbose)
        results.append(summary.to_dict())

    else:
        print("Evaluating base model (no adapter)...")
        summary = evaluate_adapter(
            model_path, None, test_cases,
            system_prompt=system_prompt,
            tool_schema=tool_schema,
            device=args.device,
            dtype=args.dtype,
        )
        print_summary(summary, args.verbose)
        results.append(summary.to_dict())

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
