#!/usr/bin/env python3
"""
Two-Stage formal verification experiment.

Stage 1: Translation - LLM generates axioms + theorem with sorry
Stage 2: Proving - LLM generates proof term to replace sorry

Usage:
    PYTHONPATH=src:$PYTHONPATH python src/experiments/test_twostage.py \
        --dataset folio --model gpt-5

    PYTHONPATH=src:$PYTHONPATH python src/experiments/test_twostage.py \
        --dataset multilogieval --model deepseek-r1 --depths d4,d5
"""

import sys
import asyncio
import argparse
import traceback
from pathlib import Path
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.lean_utils import extract_lean_code, create_lean_server, verify_with_lean_async
from utils.api_client import create_client
from utils.savers import TwoStageSaver
from utils.answer_parsing import parse_answer
from utils.prompts import load_prompt, format_system_prompt
from utils.datasets import load_folio, load_multilogieval_sampled


# Default prompt directory
DEFAULT_PROMPT_DIR = "prompts/twostage"

def get_prompt_paths(prompt_dir: str) -> dict:
    """Get prompt paths for a given directory."""
    return {
        "stage1_system": f"{prompt_dir}/two-stage1_system.txt",
        "stage1_user": f"{prompt_dir}/two-stage1_user.txt",
        "stage1_feedback": f"{prompt_dir}/two-stage1_feedback.txt",
        "stage1_no_code": f"{prompt_dir}/two-stage1_no_code.txt",
        "stage2_system": f"{prompt_dir}/two-stage2_system.txt",
        "stage2_user": f"{prompt_dir}/two-stage2_user.txt",
        "stage2_feedback": f"{prompt_dir}/two-stage2_feedback.txt",
        "stage2_no_proof": f"{prompt_dir}/two-stage2_no_proof.txt",
    }

# Answer format for parsing (set per dataset)
# FOLIO: "true_false" (True/False/Uncertain)
# MultiLogiEval: "yes_no" (Yes/No/Uncertain)


def get_token_usage(response) -> dict:
    """Extract token usage from response."""
    if not response.usage:
        return {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}

    usage = response.usage
    if isinstance(usage, dict):
        return {
            'prompt_tokens': usage.get('prompt_tokens', 0) or 0,
            'completion_tokens': usage.get('completion_tokens', 0) or 0,
            'total_tokens': usage.get('total_tokens', 0) or 0,
        }
    return {
        'prompt_tokens': getattr(usage, 'prompt_tokens', 0) or 0,
        'completion_tokens': getattr(usage, 'completion_tokens', 0) or 0,
        'total_tokens': getattr(usage, 'total_tokens', 0) or 0,
    }


async def run_stage1(
    client,
    case: dict,
    lean_server,
    model: str,
    max_iterations: int,
    system_prompt: str,
    user_prompt_template: str,
    feedback_template: str,
    no_code_template: str,
    answer_format: str,
) -> tuple:
    """
    Stage 1: Generate and type-check axioms + theorem with sorry.

    Returns: (stage1_code, success, iterations_data, last_prediction)
    """
    # Format user prompt
    premises = case.get('premises', case.get('context', ''))
    conclusion = case.get('conclusion', case.get('question', ''))
    user_prompt = user_prompt_template.format(premises=premises, conclusion=conclusion)

    conversation_history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    iterations = []
    last_prediction = None
    stage1_code = None

    for iteration in range(max_iterations):
        response = await client.chat.completions.create(
            model=model,
            messages=conversation_history
        )

        llm_response = response.choices[0].message.content or ""
        reasoning_content = getattr(response.choices[0].message, 'reasoning_content', None)
        token_usage = get_token_usage(response)

        conversation_history.append({"role": "assistant", "content": llm_response})

        # Parse prediction
        prediction, parse_status = parse_answer(llm_response, answer_format)
        last_prediction = prediction

        # Extract Lean code
        lean_code = extract_lean_code(llm_response)

        iteration_data = {
            'iteration': iteration + 1,
            'llm_response': llm_response,
            'reasoning_content': reasoning_content,
            'prediction': prediction,
            'parse_status': parse_status,
            'lean_code': lean_code,
            'lean_error': None,
            'token_usage': token_usage,
        }

        if lean_code:
            # Verify type-checking (sorry is acceptable at this stage)
            verification = await verify_with_lean_async(lean_code, lean_server)

            if verification['success']:
                # Type-check passed
                iteration_data['lean_verification'] = verification
                iterations.append(iteration_data)
                stage1_code = lean_code
                return stage1_code, True, iterations, last_prediction

            # Type-check failed - provide feedback
            error_msg = '\n'.join(verification.get('errors', []))
            iteration_data['lean_error'] = error_msg
            iteration_data['lean_verification'] = verification

            if iteration < max_iterations - 1:
                feedback = feedback_template.format(
                    lean_code=lean_code,
                    error_messages=error_msg
                )
                conversation_history.append({"role": "user", "content": feedback})
        else:
            iteration_data['lean_error'] = "No Lean code found in response"
            if iteration < max_iterations - 1:
                conversation_history.append({"role": "user", "content": no_code_template})

        iterations.append(iteration_data)

    return None, False, iterations, last_prediction


async def run_stage2(
    client,
    stage1_code: str,
    lean_server,
    model: str,
    max_iterations: int,
    system_prompt: str,
    user_prompt_template: str,
    feedback_template: str,
    no_proof_template: str,
    answer_format: str,
) -> tuple:
    """
    Stage 2: Generate proof term to replace sorry.

    Returns: (full_code, success, iterations_data, last_prediction)
    """
    user_prompt = user_prompt_template.format(stage1_code=stage1_code)

    conversation_history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    iterations = []
    last_prediction = None
    full_code = None

    for iteration in range(max_iterations):
        response = await client.chat.completions.create(
            model=model,
            messages=conversation_history
        )

        llm_response = response.choices[0].message.content or ""
        reasoning_content = getattr(response.choices[0].message, 'reasoning_content', None)
        token_usage = get_token_usage(response)

        conversation_history.append({"role": "assistant", "content": llm_response})

        # Parse prediction
        prediction, parse_status = parse_answer(llm_response, answer_format)
        last_prediction = prediction

        # Extract complete Lean code (with proof replacing sorry)
        full_code = extract_lean_code(llm_response)

        iteration_data = {
            'iteration': iteration + 1,
            'llm_response': llm_response,
            'reasoning_content': reasoning_content,
            'prediction': prediction,
            'parse_status': parse_status,
            'lean_code': full_code,
            'lean_error': None,
            'token_usage': token_usage,
        }

        if full_code:

            # Verify complete proof
            verification = await verify_with_lean_async(full_code, lean_server)

            if verification['success']:
                iteration_data['lean_verification'] = verification
                iteration_data['full_code'] = full_code
                iterations.append(iteration_data)
                return full_code, True, iterations, last_prediction

            # Proof failed - provide feedback
            error_msg = '\n'.join(verification.get('errors', []))
            iteration_data['lean_error'] = error_msg
            iteration_data['lean_verification'] = verification
            iteration_data['full_code'] = full_code

            if iteration < max_iterations - 1:
                feedback = feedback_template.format(
                    full_code=full_code,
                    error_messages=error_msg
                )
                conversation_history.append({"role": "user", "content": feedback})
        else:
            iteration_data['lean_error'] = "No Lean code found in response"
            if iteration < max_iterations - 1:
                conversation_history.append({"role": "user", "content": no_proof_template})

        iterations.append(iteration_data)

    return None, False, iterations, last_prediction


async def run_two_stage_case(
    client,
    case: dict,
    lean_server,
    semaphore: asyncio.Semaphore,
    model: str,
    max_stage1_iterations: int,
    max_stage2_iterations: int,
    prompts: dict,
    dataset: str,
    answer_format: str,
) -> dict:
    """Run complete two-stage pipeline for one case."""
    async with semaphore:
        ground_truth = case.get('ground_truth', case.get('label', case.get('answer', '')))

        result = {
            'case_idx': case.get('idx', case.get('id')),
            'ground_truth': ground_truth,
            'prediction': None,
            'correct': False,
            'stage1_success': False,
            'stage2_success': False,
            'fail_stage': None,
            'stage1_code': None,
            'stage2_code': None,
            'stage1_iterations': [],
            'stage2_iterations': [],
        }

        # Add dataset-specific metadata
        if dataset == 'folio':
            result['story_id'] = case.get('story_id')
            result['example_id'] = case.get('example_id')
        else:
            result['logic_type'] = case.get('logic_type')
            result['depth'] = case.get('depth')
            result['rule'] = case.get('rule')

        try:
            # STAGE 1: Translation
            stage1_code, s1_success, s1_iters, s1_pred = await run_stage1(
                client=client,
                case=case,
                lean_server=lean_server,
                model=model,
                max_iterations=max_stage1_iterations,
                system_prompt=prompts['stage1_system'],
                user_prompt_template=prompts['stage1_user'],
                feedback_template=prompts['stage1_feedback'],
                no_code_template=prompts['stage1_no_code'],
                answer_format=answer_format,
            )

            result['stage1_iterations'] = s1_iters
            result['stage1_success'] = s1_success
            result['stage1_code'] = stage1_code

            if not s1_success:
                result['fail_stage'] = 'stage1'
                result['prediction'] = s1_pred
                # Check correctness based on last prediction
                if s1_pred:
                    result['correct'] = s1_pred.lower() == ground_truth.lower()
                return result

            # STAGE 2: Proving
            stage2_code, s2_success, s2_iters, s2_pred = await run_stage2(
                client=client,
                stage1_code=stage1_code,
                lean_server=lean_server,
                model=model,
                max_iterations=max_stage2_iterations,
                system_prompt=prompts['stage2_system'],
                user_prompt_template=prompts['stage2_user'],
                feedback_template=prompts['stage2_feedback'],
                no_proof_template=prompts['stage2_no_proof'],
                answer_format=answer_format,
            )

            result['stage2_iterations'] = s2_iters
            result['stage2_success'] = s2_success
            result['stage2_code'] = stage2_code
            result['prediction'] = s2_pred

            if s2_success:
                # Proof verified
                if s2_pred:
                    result['correct'] = s2_pred.lower() == ground_truth.lower()
            else:
                result['fail_stage'] = 'stage2'
                # Use last prediction even if proof failed
                if s2_pred:
                    result['correct'] = s2_pred.lower() == ground_truth.lower()

        except Exception as e:
            result['error'] = f"{type(e).__name__}: {str(e)}"
            result['error_traceback'] = traceback.format_exc()
            result['fail_stage'] = 'exception'
            print(f"Exception in case {result['case_idx']}: {type(e).__name__}: {e}")

        return result


async def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Two-Stage formal verification experiment")
    parser.add_argument('--dataset', choices=['folio', 'multilogieval'], required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--max_stage1_iterations', type=int, default=3)
    parser.add_argument('--max_stage2_iterations', type=int, default=3)
    parser.add_argument('--concurrency', type=int, default=5)
    parser.add_argument('--resume', default=None, help='Resume from existing directory')
    parser.add_argument('--num_questions', type=int, default=0, help='Limit questions (0=all)')
    # MultiLogiEval options
    parser.add_argument('--depths', default='d3,d4,d5', help='Depths to include')
    parser.add_argument('--logic_types', default='fol', help='Logic types to include')
    parser.add_argument('--samples_per_combination', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_file', default=None, help='Pre-sampled data file')
    # FOLIO options
    parser.add_argument('--folio_file', default='data/folio/original/folio-validation.json')
    # Prompt options
    parser.add_argument('--prompt_dir', default=DEFAULT_PROMPT_DIR,
                        help='Directory containing prompt files (default: prompts/twostage)')

    args = parser.parse_args()

    # Load prompts from specified directory
    prompt_paths = get_prompt_paths(args.prompt_dir)
    prompts = {key: load_prompt(path) for key, path in prompt_paths.items()}
    # Format stage2 prompts with dataset-specific answer format
    prompts['stage2_system'] = format_system_prompt(prompts['stage2_system'], args.dataset)
    # User prompt has {stage1_code} placeholder, so use simple replace for answer_format
    answer_format_str = "True/False/Uncertain" if args.dataset == "folio" else "Yes/No/Uncertain"
    prompts['stage2_user'] = prompts['stage2_user'].replace("{answer_format}", answer_format_str)
    print(f"Using prompts from: {args.prompt_dir}")

    # Load dataset
    if args.dataset == 'folio':
        cases = load_folio(args.folio_file)
        depths = None
        logic_types = None
    else:
        depths = args.depths.split(',') if args.depths else None
        logic_types = args.logic_types.split(',') if args.logic_types else None

        if args.data_file:
            cases = load_multilogieval_sampled(args.data_file)
            print(f"Loaded {len(cases)} cases from {args.data_file}")
        else:
            cases = load_multilogieval_sampled(
                data_dir='data/multi_logi_original/data',
                depths=depths,
                logic_types=logic_types,
                samples_per_combination=args.samples_per_combination,
                seed=args.seed
            )

    # Limit cases if specified
    if args.num_questions > 0:
        cases = cases[:args.num_questions]

    print(f"Running two-stage experiment on {len(cases)} cases")

    # Initialize client and Lean server
    client = create_client(model=args.model)
    lean_server = create_lean_server()

    # Determine output subdirectory based on prompt type
    prompt_suffix = "_nosample" if "nosample" in args.prompt_dir else ""

    # Initialize saver
    saver = TwoStageSaver(
        output_dir=f"results/{args.dataset}/twostage{prompt_suffix}",
        dataset=args.dataset,
        model=args.model,
        resume_dir=args.resume,
        max_stage1_iterations=args.max_stage1_iterations,
        max_stage2_iterations=args.max_stage2_iterations,
        concurrency=args.concurrency,
        depths=depths,
        logic_types=logic_types,
    )

    # Filter out completed cases
    pending_cases = [c for c in cases if not saver.is_completed(c.get('idx', c.get('id')))]
    print(f"Pending: {len(pending_cases)} cases (skipping {len(cases) - len(pending_cases)} completed)")

    if not pending_cases:
        print("All cases already completed!")
        saver.finalize()
        return

    # Run experiments with concurrency control
    semaphore = asyncio.Semaphore(args.concurrency)
    answer_format = "true_false" if args.dataset == "folio" else "yes_no"

    async def process_case(case):
        result = await run_two_stage_case(
            client=client,
            case=case,
            lean_server=lean_server,
            semaphore=semaphore,
            model=args.model,
            max_stage1_iterations=args.max_stage1_iterations,
            max_stage2_iterations=args.max_stage2_iterations,
            prompts=prompts,
            dataset=args.dataset,
            answer_format=answer_format,
        )
        await saver.save_result(result, result['case_idx'])
        return result

    tasks = [process_case(case) for case in pending_cases]
    await tqdm_asyncio.gather(*tasks, desc="Two-Stage")

    # Finalize
    summary = saver.finalize()
    print(f"\nAccuracy: {summary['accuracy']*100:.1f}%")
    print(f"Stage 1 Success: {summary['stage1_success_rate']*100:.1f}%")
    print(f"Stage 2 Success: {summary['stage2_success_rate']*100:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
