# Do LLMs Game Formalization? Evaluating Faithfulness in Logical Reasoning

This repository contains code and data for the paper:

**Do LLMs Game Formalization? Evaluating Faithfulness in Logical Reasoning**
Kyuhee Kim, Auguste Poiroux, Antoine Bosselut
*Published at the VerifAI-2 Workshop, ICLR 2026*

We evaluate GPT-5 and DeepSeek-R1 on 303 first-order logic problems from FOLIO and Multi-LogiEval, comparing unified generation against a two-stage pipeline that separates formalization from proving.

## Repository Structure

```
src/
├── experiments/          # Main experiment scripts
│   ├── test_simplelean.py           # Unified approach (Baseline/Directed/Nudged)
│   └── test_twostage.py             # Two-Stage approach
├── analysis/             # Analysis and evaluation scripts
│   └── evaluate_faithfulness.py     # LLM-as-Judge faithfulness evaluation
├── datasets/             # Dataset loading utilities
└── utils/                # Lean verification, API clients, answer parsing
prompts/
├── bidirectional/        # Unified approach prompts (Baseline/Directed/Nudged)
├── twostage/             # Two-Stage prompts (Stage 1 & 2)
└── error-classification/ # LLM-as-Judge prompts (v26 = final version)
data/
├── folio/                # FOLIO dataset (203 validation problems)
└── multilogieval/        # Multi-LogiEval dataset (100 sampled problems)
results/
├── folio/                # FOLIO experiment results by model/condition
├── multilogieval/        # Multi-LogiEval experiment results
├── llm-as-judge/         # Faithfulness classification results
└── prediction_error_fig/ # Sankey diagrams (Figure 3 in paper)
notebooks/                # Analysis notebooks
samples/
├── dataset_errors/       # Flagged dataset error cases (Appendix H)
└── lucky_gaming/         # Case studies of undetected gaming
```

## Setup

```bash
# Environment
conda create -n llm-lean python=3.10
conda activate llm-lean
pip install -r requirements.txt

# Lean 4 (via elan)
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh

# API keys (copy and fill in .env)
cp .env.example .env
```

## Running Experiments

```bash
# Unified approach (Baseline condition)
PYTHONPATH=src:$PYTHONPATH python src/experiments/test_simplelean.py \
    --model gpt-5-2025-08-07 --condition baseline --num_questions 0

# Two-Stage approach
PYTHONPATH=src:$PYTHONPATH python src/experiments/test_twostage.py \
    --model gpt-5-2025-08-07 --num_questions 0

# LLM-as-Judge faithfulness evaluation
PYTHONPATH=src:$PYTHONPATH python src/analysis/evaluate_faithfulness.py \
    --cases results/faithfulness_test_cases.json \
    --system_prompt prompts/error-classification/v26_system.txt \
    --prompt prompts/error-classification/v26_user.txt \
    --model anthropic/claude-opus-4.5
```

## Key Files

| Paper Section | Code/Data |
|---------------|-----------|
| Table 3 (Main Results) | `results/folio/`, `results/multilogieval/` |
| Table 7 (Fabrication) | `notebooks/5.fabrication_analysis.ipynb` |
| Figure 3 (Sankey) | `results/prediction_error_fig/sankey_*.png` |
| Appendix H (Dataset Errors) | `samples/dataset_errors/` |
| Appendix E (LLM-as-Judge Prompt) | `prompts/error-classification/v26_*.txt` |

## Citation

```bibtex
@misc{kim2026formalization,
  title={Do LLMs Game Formalization? Evaluating Faithfulness in Logical Reasoning},
  author={Kim, Kyuhee and Poiroux, Auguste and Bosselut, Antoine},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.
