#!/usr/bin/env python3
"""
Create test cases for faithfulness evaluation - FIXED version.

Correct ID matching:
- FOLIO: use 'example_id' from results to lookup in original data
- MultiLogiEval: use 'case_idx' from results to lookup by 'idx' in sampled data
"""

import json
import re
import random
from pathlib import Path
from collections import defaultdict, Counter

random.seed(42)

RESULTS_DIR = Path("results")
DATA_DIR = Path("data")

CONDITIONS = ["baseline", "bidir_true", "bidir_false", "spooky_true", "spooky_false"]
DATASETS = ["folio", "multilogieval"]
MODELS = ["gpt-5", "deepseek"]

# Load FOLIO data - index by example_id
def load_folio_data():
    path = DATA_DIR / "folio/original/folio-validation.json"
    with open(path) as f:
        data = json.load(f)
    return {e['example_id']: e for e in data}

# Load MultiLogiEval data - index by idx
def load_multilogieval_data():
    path = DATA_DIR / "multilogieval/multilogieval-sampled.json"
    with open(path) as f:
        data = json.load(f)
    return {e['idx']: e for e in data}

FOLIO_DATA = load_folio_data()
MULTILOGIEVAL_DATA = load_multilogieval_data()
print(f"Loaded {len(FOLIO_DATA)} FOLIO entries (indexed by example_id)")
print(f"Loaded {len(MULTILOGIEVAL_DATA)} MultiLogiEval entries (indexed by idx)")

def get_premises_conclusion(dataset, entry):
    """Get premises and conclusion from dataset using correct ID field."""
    if dataset == 'folio':
        # Use example_id from the result entry
        example_id = entry.get('example_id')
        if example_id is None:
            return '', ''
        data = FOLIO_DATA.get(example_id, {})
        premises = data.get('premises', [])
        if isinstance(premises, list):
            premises = '\n'.join(premises)
        return premises, data.get('conclusion', '')
    else:  # multilogieval
        # Use case_idx which matches idx in sampled data
        case_idx = entry.get('case_idx')
        if case_idx is None:
            return '', ''
        data = MULTILOGIEVAL_DATA.get(case_idx, {})
        return data.get('context', ''), data.get('question', '')

def load_jsonl(path):
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(l) for l in f]

def extract_axioms(code):
    if not code:
        return set()
    return set(a.strip() for a in re.findall(r'^axiom\s+.+$', code, re.MULTILINE))

def is_wrong_direction(gt, pred):
    gt_norm = gt.lower() if gt else ''
    pred_norm = pred.lower() if pred else ''

    gt_norm = {'true': 'true', 'false': 'false', 'uncertain': 'uncertain',
               'yes': 'true', 'no': 'false'}.get(gt_norm, gt_norm)
    pred_norm = {'true': 'true', 'false': 'false', 'uncertain': 'uncertain',
                 'yes': 'true', 'no': 'false', 'unknown': 'uncertain'}.get(pred_norm, pred_norm)

    wrong_pairs = [('true', 'false'), ('false', 'true'), ('uncertain', 'true'), ('uncertain', 'false')]
    return (gt_norm, pred_norm) in wrong_pairs

# Load all results
print("\nLoading results from all conditions...")
all_results = {}
for dataset in DATASETS:
    for model in MODELS:
        for condition in CONDITIONS:
            for run in [1, 2, 3]:
                path = RESULTS_DIR / dataset / model / f"{condition}_run{run}" / "results.jsonl"
                entries = load_jsonl(path)
                for e in entries:
                    key = (dataset, model, condition, e.get('case_idx'), run)
                    all_results[key] = e

print(f"Loaded {len(all_results)} total result entries")

# CATEGORY 1: Wrong Direction Errors
print("\n" + "="*60)
print("CATEGORY 1: WRONG DIRECTION ERRORS")
print("="*60)

wrong_direction_cases = []
for (dataset, model, condition, case_idx, run), e in all_results.items():
    gt = e.get('ground_truth')
    pred = e.get('prediction')

    if is_wrong_direction(gt, pred):
        premises, conclusion = get_premises_conclusion(dataset, e)
        wrong_direction_cases.append({
            'source': 'wrong_direction',
            'dataset': dataset,
            'model': model,
            'condition': condition,
            'case_idx': case_idx,
            'run': run,
            'ground_truth': gt,
            'prediction': pred,
            'lean_code': e.get('lean_code', ''),
            'premises': premises,
            'conclusion': conclusion
        })

print(f"Found {len(wrong_direction_cases)} wrong direction cases")
# Check premise coverage
empty_premises = sum(1 for c in wrong_direction_cases if not c['premises'])
print(f"Cases with empty premises: {empty_premises}")

# CATEGORY 2: Divergent Cases
print("\n" + "="*60)
print("CATEGORY 2: DIVERGENT CASES")
print("="*60)

divergent_cases = []
for dataset in DATASETS:
    for model in MODELS:
        for prefix in ['bidir', 'spooky']:
            true_cond = f"{prefix}_true"
            false_cond = f"{prefix}_false"

            for run in [1, 2, 3]:
                true_results = {}
                false_results = {}

                for (ds, m, cond, case_idx, r), e in all_results.items():
                    if ds == dataset and m == model and r == run:
                        if cond == true_cond:
                            true_results[case_idx] = e
                        elif cond == false_cond:
                            false_results[case_idx] = e

                for case_idx in set(true_results.keys()) & set(false_results.keys()):
                    t_entry = true_results[case_idx]
                    f_entry = false_results[case_idx]

                    t_pred = (t_entry.get('prediction') or '').lower()
                    f_pred = (f_entry.get('prediction') or '').lower()

                    t_proved = t_pred in ['true', 'yes']
                    f_proved = f_pred in ['false', 'no']

                    if t_proved and f_proved:
                        premises, conclusion = get_premises_conclusion(dataset, t_entry)
                        divergent_cases.append({
                            'source': 'divergent',
                            'dataset': dataset,
                            'model': model,
                            'condition_pair': prefix,
                            'case_idx': case_idx,
                            'run': run,
                            'ground_truth': t_entry.get('ground_truth'),
                            'true_lean_code': t_entry.get('lean_code', ''),
                            'false_lean_code': f_entry.get('lean_code', ''),
                            'premises': premises,
                            'conclusion': conclusion
                        })

print(f"Found {len(divergent_cases)} divergent cases")
empty_premises = sum(1 for c in divergent_cases if not c['premises'])
print(f"Cases with empty premises: {empty_premises}")

# Build divergent keys for exclusion
divergent_keys = set()
for c in divergent_cases:
    prefix = c['condition_pair']
    divergent_keys.add((c['dataset'], c['model'], f'{prefix}_true', c['case_idx'], c['run']))
    divergent_keys.add((c['dataset'], c['model'], f'{prefix}_false', c['case_idx'], c['run']))

wrong_direction_filtered = [c for c in wrong_direction_cases
                           if (c['dataset'], c['model'], c['condition'], c['case_idx'], c['run']) not in divergent_keys]
print(f"\nWrong direction after filtering divergent: {len(wrong_direction_filtered)}")
wrong_direction_cases = wrong_direction_filtered

# CATEGORY 3: Fabrication Cases (Two-stage)
print("\n" + "="*60)
print("CATEGORY 3: FABRICATION CASES (two-stage)")
print("="*60)

twostage_results = {}
for dataset in DATASETS:
    for model_name in ['gpt-5', 'deepseek-r1']:
        for run in [1, 2, 3]:
            path = RESULTS_DIR / dataset / "twostage" / f"{model_name}_run{run}" / "results.jsonl"
            entries = load_jsonl(path)
            for e in entries:
                key = (dataset, model_name, e.get('case_idx'), run)
                twostage_results[key] = e

fabrication_cases = []
for (dataset, model, case_idx, run), e in twostage_results.items():
    if model != 'gpt-5':
        continue

    s1_code = e.get('stage1_code', '')
    s2_code = e.get('stage2_code', '')
    if not s1_code or not s2_code:
        continue

    ax1 = extract_axioms(s1_code)
    ax2 = extract_axioms(s2_code)
    added = ax2 - ax1
    removed = ax1 - ax2

    if added and not removed:
        premises, conclusion = get_premises_conclusion(dataset, e)
        fabrication_cases.append({
            'source': 'fabrication',
            'dataset': dataset,
            'model': model,
            'case_idx': case_idx,
            'run': run,
            'ground_truth': e.get('ground_truth'),
            'prediction': e.get('prediction'),
            'lean_code': s2_code,  # Use stage2 code
            'stage1_code': s1_code,
            'stage2_code': s2_code,
            'added_axioms': list(added),
            'premises': premises,
            'conclusion': conclusion
        })

print(f"Found {len(fabrication_cases)} fabrication cases")
empty_premises = sum(1 for c in fabrication_cases if not c['premises'])
print(f"Cases with empty premises: {empty_premises}")

# CATEGORY 4: Stratified Correct Samples
print("\n" + "="*60)
print("CATEGORY 4: STRATIFIED CORRECT SAMPLES")
print("="*60)

correct_by_stratum = defaultdict(list)
for (dataset, model, condition, case_idx, run), e in all_results.items():
    if condition != 'baseline' or run != 1:
        continue
    if not e.get('correct'):
        continue
    lean_ver = e.get('lean_verification', {})
    if not lean_ver.get('success', False):
        continue

    gt = e.get('ground_truth')
    premises, conclusion = get_premises_conclusion(dataset, e)

    stratum = (dataset, model, gt)
    correct_by_stratum[stratum].append({
        'source': 'stratified_correct',
        'dataset': dataset,
        'model': model,
        'condition': 'baseline',
        'case_idx': case_idx,
        'ground_truth': gt,
        'prediction': e.get('prediction'),
        'lean_code': e.get('lean_code', ''),
        'premises': premises,
        'conclusion': conclusion
    })

stratified_correct = []
for stratum in sorted(correct_by_stratum.keys()):
    cases = correct_by_stratum[stratum]
    n_sample = min(5, len(cases))
    sampled = random.sample(cases, n_sample)
    stratified_correct.extend(sampled)
    print(f"  {stratum}: {n_sample}/{len(cases)}")

print(f"\nTotal stratified correct: {len(stratified_correct)}")
empty_premises = sum(1 for c in stratified_correct if not c['premises'])
print(f"Cases with empty premises: {empty_premises}")

# Save individual files
output_dir = RESULTS_DIR / "extracted_errors"
output_dir.mkdir(exist_ok=True)

with open(output_dir / "test_wrong_direction.json", 'w') as f:
    json.dump(wrong_direction_cases, f, indent=2)
print(f"\nSaved {len(wrong_direction_cases)} wrong_direction cases")

with open(output_dir / "test_divergent.json", 'w') as f:
    json.dump(divergent_cases, f, indent=2)
print(f"Saved {len(divergent_cases)} divergent cases")

with open(output_dir / "test_fabrication.json", 'w') as f:
    json.dump(fabrication_cases, f, indent=2)
print(f"Saved {len(fabrication_cases)} fabrication cases")

with open(output_dir / "test_stratified_correct.json", 'w') as f:
    json.dump(stratified_correct, f, indent=2)
print(f"Saved {len(stratified_correct)} stratified_correct cases")

# Summary
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
total = len(wrong_direction_cases) + len(divergent_cases) + len(fabrication_cases) + len(stratified_correct)
print(f"Total test cases: {total}")
