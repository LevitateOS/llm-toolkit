# llm-toolkit

Python scripts for LoRA fine-tuning. Generic toolkit, not integrated with LevitateOS installer yet.

## Status

**Alpha.** Training works. Not integrated with installer TUI.

| Works | Doesn't work yet |
|-------|------------------|
| LoRA training (4-bit, 8-bit) | LevitateOS installer integration |
| Training data validation | Automated deployment |
| HTTP inference server | |
| Adapter evaluation | |

## Scripts

| Script | Purpose |
|--------|---------|
| `train_lora.py` | Train LoRA adapters |
| `generate_data.py` | Validate/augment training data |
| `llm_server.py` | HTTP inference server |
| `evaluate.py` | Evaluate adapters |

## Installation

```bash
pip install -r requirements.txt
```

Requires: PyTorch, transformers, peft, bitsandbytes (optional).

## Training

```bash
# Basic training
python train_lora.py --model ./base-model --data training.jsonl --output ./adapter

# With 4-bit quantization (less VRAM)
python train_lora.py --model ./base-model --data training.jsonl --use-4bit
```

## Data Format

Training data (JSONL):

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

## Serving

```bash
python llm_server.py --model ./base-model --adapter ./adapter --port 8080
```

```bash
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

## Hardware Requirements

| Setup | VRAM |
|-------|------|
| 4-bit quantized | 2-4 GB |
| 8-bit | 4-6 GB |
| Full precision | 8+ GB |
| CPU only | 8+ GB RAM (slow) |

## Known Limitations

- Not integrated with LevitateOS installer
- No automated deployment pipeline
- Training data must be manually prepared
- Server is single-threaded

## License

MIT
