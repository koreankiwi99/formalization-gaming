# Error Classification Prompts

Prompts for LLM-as-Judge faithfulness evaluation.

## Files

| File | Description |
|------|-------------|
| `v26_system.txt` | Final system prompt (used in paper) |
| `v26_user.txt` | Final user prompt template |
| `v26_divergent_system.txt` | System prompt for divergent case analysis |
| `v26_divergent_user.txt` | User prompt for divergent case analysis |
| `v1.txt` - `v6.txt` | Earlier iterations (for reference) |

## Usage

```bash
PYTHONPATH=src:$PYTHONPATH python src/analysis/evaluate_faithfulness.py \
    --cases results/faithfulness_test_cases.json \
    --system_prompt prompts/error-classification/v26_system.txt \
    --prompt prompts/error-classification/v26_user.txt \
    --model anthropic/claude-opus-4.5
```
