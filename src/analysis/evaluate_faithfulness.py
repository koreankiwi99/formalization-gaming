#!/usr/bin/env python3
"""
Evaluate faithfulness of Lean formalizations using LLM-as-judge.

Supports v9 prompt format (multiple errors per case).

Usage:
    PYTHONPATH=src:$PYTHONPATH python src/analysis/evaluate_faithfulness.py \
        --cases results/faithfulness_test_cases_v9.json \
        --prompt prompts/error-classification/v9_faithfulness.txt \
        --output results/faithfulness_v9_results.csv \
        --model gpt-4o

    # With separate system/user prompts (v26+):
    PYTHONPATH=src:$PYTHONPATH python src/analysis/evaluate_faithfulness.py \
        --cases results/faithfulness_test_cases_v9.json \
        --system_prompt prompts/error-classification/v26_system.txt \
        --prompt prompts/error-classification/v26_user.txt \
        --output results/faithfulness_v26_results.csv
"""

import json
import re
import os
import sys
import time
import argparse
import asyncio
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio


def load_prompt_template(prompt_path: str) -> str:
    """Load prompt template from file."""
    with open(prompt_path, 'r') as f:
        return f.read().strip()


def get_lean_code(case: dict) -> str:
    """Extract Lean code from case (handles baseline, two-stage, and divergent)."""
    # Divergent cases have both true and false codes
    if 'true_lean_code' in case and 'false_lean_code' in case:
        true_code = case.get('true_lean_code', '')
        false_code = case.get('false_lean_code', '')
        return f"=== TRUE DIRECTION CODE ===\n{true_code}\n\n=== FALSE DIRECTION CODE ===\n{false_code}"
    # Two-stage cases have stage2_code as final output
    if 'stage2_code' in case:
        return case.get('stage2_code', '')
    return case.get('lean_code', '')


def format_premises(premises) -> str:
    """Format premises for prompt."""
    if isinstance(premises, list):
        return '\n'.join(f"- {p}" for p in premises)
    return str(premises) if premises else 'N/A'


async def analyze_case(case: dict, client: AsyncOpenAI, prompt_template: str,
                       model: str, semaphore: asyncio.Semaphore,
                       system_prompt: str = None) -> dict:
    """Analyze a single case using LLM."""
    async with semaphore:
        lean_code = get_lean_code(case)
        premises = format_premises(case.get('premises', []))
        conclusion = case.get('conclusion', 'N/A')

        # Format prompt
        prompt = prompt_template.format(
            premises=premises[:3000],
            conclusion=conclusion,
            ground_truth=case.get('ground_truth', 'N/A'),
            prediction=case.get('prediction', 'N/A'),
            lean_code=lean_code[:5000]
        )

        # Build messages
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=2048
            )

            response_text = response.choices[0].message.content

            # Extract JSON - handle both is_faithful and formalization_faithful
            json_match = re.search(r'\{[\s\S]*"(?:is_faithful|formalization_faithful)"[\s\S]*\}', response_text)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    # Normalize key name
                    if 'formalization_faithful' in result and 'is_faithful' not in result:
                        result['is_faithful'] = result['formalization_faithful']
                    return result
                except json.JSONDecodeError:
                    pass

            return {
                "is_faithful": None,
                "errors": [],
                "confidence": "LOW",
                "parse_error": response_text[:500]
            }

        except Exception as e:
            return {
                "is_faithful": None,
                "errors": [],
                "confidence": "LOW",
                "api_error": str(e)[:200]
            }


async def main():
    parser = argparse.ArgumentParser(description='Evaluate faithfulness of Lean formalizations')
    parser.add_argument('--cases', required=True, help='Path to test cases JSON file')
    parser.add_argument('--prompt', required=True, help='Path to user prompt template file')
    parser.add_argument('--system_prompt', help='Path to system prompt file (optional)')
    parser.add_argument('--output', help='Output CSV path')
    parser.add_argument('--model', default='gpt-4o', help='Model to use for analysis')
    parser.add_argument('--concurrency', type=int, default=10, help='Concurrent API calls')
    parser.add_argument('--limit', type=int, default=0, help='Limit cases to analyze (0=all)')

    args = parser.parse_args()

    # Load environment
    load_dotenv()
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key)

    # Load prompt templates
    print(f"Loading user prompt from: {args.prompt}")
    prompt_template = load_prompt_template(args.prompt)

    system_prompt = None
    if args.system_prompt:
        print(f"Loading system prompt from: {args.system_prompt}")
        system_prompt = load_prompt_template(args.system_prompt)

    # Load test cases
    print(f"Loading cases from: {args.cases}")
    with open(args.cases, 'r') as f:
        cases = json.load(f)

    if args.limit > 0:
        cases = cases[:args.limit]

    print(f"Analyzing {len(cases)} cases with concurrency={args.concurrency}")

    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(args.concurrency)

    # Analyze all cases
    tasks = [analyze_case(case, client, prompt_template, args.model, semaphore, system_prompt)
             for case in cases]

    results = await tqdm_asyncio.gather(*tasks, desc="Analyzing")

    # Combine cases with results
    rows = []
    for case, result in zip(cases, results):
        # Base info from case
        row = {
            'source': case.get('source'),
            'dataset': case.get('dataset'),
            'model': case.get('model'),
            'condition': case.get('condition'),
            'run': case.get('run'),
            'case_idx': case.get('case_idx'),
            'ground_truth': case.get('ground_truth'),
            'prediction': case.get('prediction'),
            'direction_pred': case.get('direction_pred'),
            'matches_gt': case.get('matches_gt'),
            'correct': case.get('correct'),
        }

        # Analysis results
        row['is_faithful'] = result.get('is_faithful')
        row['confidence'] = result.get('confidence', 'LOW')

        # Handle errors array (v9 format)
        errors = result.get('errors', [])
        if errors:
            # Collect all error categories and subtypes
            categories = [e.get('category') for e in errors if e.get('category')]
            subtypes = [e.get('subtype') for e in errors if e.get('subtype')]
            explanations = [e.get('explanation', '') for e in errors if e.get('explanation')]

            row['error_count'] = len(errors)
            row['categories'] = '|'.join(categories)
            row['subtypes'] = '|'.join(subtypes)
            row['explanation'] = ' | '.join(explanations[:2])  # First 2 explanations

            # Primary error (first one)
            row['primary_category'] = errors[0].get('category') if errors else None
            row['primary_subtype'] = errors[0].get('subtype') if errors else None
        else:
            row['error_count'] = 0
            row['categories'] = ''
            row['subtypes'] = ''
            row['explanation'] = ''
            row['primary_category'] = None
            row['primary_subtype'] = None

        # Handle parse/API errors
        if 'parse_error' in result:
            row['error_message'] = result['parse_error']
        elif 'api_error' in result:
            row['error_message'] = result['api_error']
        else:
            row['error_message'] = ''

        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Auto-generate output path
    if args.output:
        output_path = args.output
    else:
        prompt_name = Path(args.prompt).stem
        output_path = f'results/faithfulness_{prompt_name}_results.csv'

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Print summary
    print(f"\n{'='*70}")
    print("FAITHFULNESS ANALYSIS SUMMARY")
    print('='*70)
    print(f"Total analyzed: {len(df)}")
    print(f"\nBy Source:")
    print(df['source'].value_counts())
    print(f"\nFaithfulness:")
    print(df['is_faithful'].value_counts(dropna=False))
    print(f"\nBy Primary Category:")
    print(df['primary_category'].value_counts(dropna=False))

    # Breakdown by source
    print(f"\n{'='*70}")
    print("FAITHFULNESS BY SOURCE")
    print('='*70)
    for source in df['source'].unique():
        subset = df[df['source'] == source]
        faithful = subset['is_faithful'].sum() if subset['is_faithful'].notna().any() else 0
        total = len(subset)
        pct = faithful / total * 100 if total > 0 else 0
        print(f"  {source}: {faithful}/{total} faithful ({pct:.1f}%)")

    print(f"\nSaved to: {output_path}")


if __name__ == '__main__':
    asyncio.run(main())
