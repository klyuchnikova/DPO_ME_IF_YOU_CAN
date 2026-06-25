# Data preparation
```
data/processed/
├── train.jsonl   # 2k–5k examples (1k minimum for quick runs)
├── val.jsonl     # ~300–500
└── test.jsonl    # ~300–500
```

```json
{"id": 0, "prompt": "How do I treat a headache?", "chosen": "Rest, hydrate, and see a doctor if severe.", "rejected": "Take random pills and ignore it."}
```

Fields:

| Field | Required | Notes |
|-------|----------|-------|
| `prompt` | yes | User instruction / context |
| `chosen` | yes | Preferred response |
| `rejected` | yes | Worse response |
| `id` | no | Stable index (defaults to line number) |

## What we use

- **Dahoas/rm-static** (easy, fast)
- **Anthropic HH-RLHF** (filtered subset, more credible)

we filter by length (same tokenizer as training)

MAX_PROMPT = 256
MAX_TOTAL = 512
MIN_RESP = 8


## Base-model preference accuracy

On val/test, score each pair with the **base** model (no LoRA):

\[
\text{correct} = \mathbb{1}[\log\pi(y_w|x) > \log\pi(y_l|x)]
\]

Target - 50–65%. But the Dohaos results in 54 while HH-RLHF is below 50.
Anyway to test it run

```bash
# once per model / split
python scripts/05_evaluate.py --config configs/qwen_0.5b_dpo.yaml \
  --exp-name qwen_0.5b_base --split test --skip-runtime --skip-visualize

# aggregate all runs (base + trained) into one table
python scripts/06_summarize_results.py --split test
```