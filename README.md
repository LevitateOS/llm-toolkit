# llm-toolkit

Generic LoRA fine-tuning toolkit for training and serving language models.

## Tools

| Tool | Purpose |
|------|---------|
| `train_lora.py` | Train LoRA adapters on HuggingFace models |
| `generate_data.py` | Validate, augment, and annotate training data |
| `llm_server.py` | HTTP inference server with LoRA support |
| `evaluate.py` | Evaluate adapters with custom test cases |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
# Train LoRA adapter
python train_lora.py \
    --model path/to/base-model \
    --data training.jsonl \
    --output adapters/my-adapter \
    --epochs 3 \
    --lora-r 16

# With 4-bit quantization (saves GPU memory)
python train_lora.py --model ./model --data data.jsonl --use-4bit
```

### Data Generation

```bash
# Validate training data format
python generate_data.py validate training.jsonl

# Augment with variations
python generate_data.py augment training.jsonl -o augmented.jsonl

# Add thinking annotations (requires ANTHROPIC_API_KEY)
python generate_data.py annotate training.jsonl -o annotated.jsonl
```

### Serving

```bash
# Start HTTP server
python llm_server.py \
    --model path/to/base-model \
    --adapter adapters/my-adapter \
    --port 8080

# Query the server
curl -X POST http://localhost:8080/generate \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

### Evaluation

```bash
# Evaluate base model
python evaluate.py --model ./model --test-cases tests.jsonl

# Evaluate adapter
python evaluate.py --model ./model --adapter ./adapter --test-cases tests.jsonl

# Compare multiple adapters
python evaluate.py --model ./model --sweep-dir ./adapters --test-cases tests.jsonl
```

## Data Formats

### Training Data (JSONL)

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Or with expected response:

```json
{
  "messages": [{"role": "user", "content": "..."}],
  "expected_response": {"type": "command", "command": "ls -la", "thinking": "User wants to list files"}
}
```

### Test Cases (JSONL)

```json
{"name": "test1", "messages": [{"role": "user", "content": "hello"}], "expected": null, "category": "greeting"}
{"name": "test2", "messages": [{"role": "user", "content": "list files"}], "expected": "ls", "category": "command"}
{"name": "test3", "messages": [...], "expected": ["ls", "-la"], "category": "command"}
```

- `expected: null` - Expects text response
- `expected: "pattern"` - Expects command containing pattern
- `expected: ["p1", "p2"]` - Expects command containing all patterns

## License

MIT
