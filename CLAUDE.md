# CLAUDE.md - llm-toolkit

## What is llm-toolkit?

Generic LoRA fine-tuning toolkit. Training, data generation, and inference serving for language models.

## What Belongs Here

- LoRA training scripts
- Data generation/validation
- Inference server
- Evaluation tools

## What Does NOT Belong Here

| Don't put here | Put it in |
|----------------|-----------|
| OS building | `leviso/` |
| Package management | `tools/recipe/` |

## Scripts

| Script | Purpose |
|--------|---------|
| `train_lora.py` | Train LoRA adapters |
| `generate_data.py` | Validate, augment, annotate data |
| `llm_server.py` | HTTP inference server |
| `evaluate.py` | Evaluate adapters |

## Commands

```bash
pip install -r requirements.txt
python train_lora.py --help
python generate_data.py --help
```

## Data Format

JSONL with chat format:
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

## Critical Warning

Training data in `training/` may be gitignored but VALUABLE. Never delete without asking.
