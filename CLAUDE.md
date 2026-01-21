# CLAUDE.md - LLM Toolkit

## ⛔ STOP. READ. THEN ACT.

Every time you think you know where something goes - **stop. Read first.**

Every time you think something is worthless and should be deleted - **stop. Read it first.**

Every time you're about to write code - **stop. Read what already exists first.**

The five minutes you spend reading will save hours of cleanup.

---

## What is llm-toolkit?

Generic LoRA fine-tuning toolkit for training, generating data, and serving language models.

## Tools

| Script | Purpose |
|--------|---------|
| `train_lora.py` | Train LoRA adapters |
| `generate_data.py` | Validate, augment, annotate training data |
| `llm_server.py` | HTTP inference server with LoRA support |
| `evaluate.py` | Evaluate adapters with test cases |

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run scripts
python train_lora.py --help
python generate_data.py --help
```

## Data Formats

Training data uses JSONL with chat format:
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

## Common Mistakes

1. **Hardcoding model paths** - Use CLI arguments
2. **Not handling OOM** - Use `--use-4bit` for memory-constrained environments
3. **Destroying training data** - ALWAYS check `git status --ignored` before deleting directories

## Critical Warning

Training data in `training/` may be gitignored but VALUABLE (cost time/money to generate). Never delete gitignored directories without asking the user first.
