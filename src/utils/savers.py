"""Result saving utilities."""

import json
import os
import asyncio
from datetime import datetime


class BaseSaver:
    """Base class for saving experiment results."""

    def __init__(self, output_dir="results", experiment_name="test"):
        """Initialize saver with output directory and experiment name.

        Args:
            output_dir: Base output directory
            experiment_name: Name of the experiment for file naming
        """
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)
        self._save_lock = asyncio.Lock()

    def _append_to_json(self, file_path, result):
        """Append a result to a JSON file.

        Args:
            file_path: Path to the JSON file
            result: Result dictionary to append
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            data.append(result)
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not update JSON file: {e}")

    def _update_json_at_index(self, file_path, result, index):
        """Update a specific entry in a JSON file.

        Args:
            file_path: Path to the JSON file
            result: Result dictionary to update
            index: Index of the entry to update
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            if 0 <= index < len(data):
                data[index] = result
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
            else:
                print(f"Warning: Index {index} out of range for JSON file")
        except Exception as e:
            print(f"Warning: Could not update JSON file at index {index}: {e}")


class SimpleLeanSaver(BaseSaver):
    """Unified saver for SimpleLean experiments (FOLIO and MultiLogiEval).

    - Auto-detects dataset type from result fields for progress logging
    - Incremental JSONL saving (crash-safe)
    - Simplified txt output (LLM responses only)
    - One experiment = one condition (no mixing)
    """

    def __init__(self, output_dir="results/simplelean", dataset="folio",
                 resume_dir=None, model="gpt-5", condition=None,
                 max_iterations=3, max_completion_tokens=0, concurrency=5,
                 depths=None, logic_types=None):
        super().__init__(output_dir, dataset)
        self.dataset = dataset
        self.condition = condition  # e.g., "implicit" or "explicit"
        self.model = model
        self.max_iterations = max_iterations
        self.max_completion_tokens = max_completion_tokens
        self.concurrency = concurrency
        self.depths = depths
        self.logic_types = logic_types

        if resume_dir:
            self.base_dir = resume_dir
            print(f"Resuming in existing directory: {self.base_dir}")
        else:
            model_name = model.replace("/", "-").replace(":", "-")
            cond_suffix = f"_{condition}" if condition else ""
            self.base_dir = f"{output_dir}/{model_name}_{dataset}{cond_suffix}_{self.timestamp}"
            os.makedirs(self.base_dir, exist_ok=True)

        self.config_file = f"{self.base_dir}/config.json"
        self.jsonl_file = f"{self.base_dir}/results.jsonl"
        self.progress_file = f"{self.base_dir}/progress.txt"
        self.summary_file = f"{self.base_dir}/summary.json"
        self.responses_dir = f"{self.base_dir}/responses"

        os.makedirs(self.responses_dir, exist_ok=True)

        self.results = []
        self.completed = set()  # case_idx

        if not resume_dir:
            self._init_files()
        else:
            self._load_existing()

    def _init_files(self):
        """Initialize output files."""
        config = {
            "model": self.model,
            "dataset": self.dataset,
            "condition": self.condition,
            "max_iterations": self.max_iterations,
            "max_completion_tokens": self.max_completion_tokens,
            "concurrency": self.concurrency,
            "depths": self.depths,
            "logic_types": self.logic_types,
            "started_at": self.timestamp
        }
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)

        with open(self.progress_file, 'w') as f:
            f.write(f"SimpleLean - {self.dataset.upper()} - Started at {self.timestamp}\n")
            f.write(f"Model: {self.model}\n")
            if self.condition:
                f.write(f"Condition: {self.condition}\n")
            if self.depths:
                f.write(f"Depths: {self.depths}\n")
            if self.logic_types:
                f.write(f"Logic types: {self.logic_types}\n")
            f.write(f"Max iterations: {self.max_iterations}\n")
            f.write("=" * 70 + "\n\n")

    def _load_existing(self):
        """Load existing results for resume support from JSONL."""
        if os.path.exists(self.jsonl_file):
            error_count = 0
            with open(self.jsonl_file, 'r') as f:
                for line in f:
                    if line.strip():
                        result = json.loads(line)
                        case_idx = result.get('case_idx')
                        if case_idx is not None:
                            # Skip error cases so they get retried
                            if result.get('error'):
                                error_count += 1
                                continue
                            self.completed.add(case_idx)
                            self.results.append(result)

            print(f"Loaded {len(self.completed)} existing results for resume (skipping {error_count} errors)")

            with open(self.progress_file, 'a') as f:
                f.write(f"\n{'='*70}\n")
                f.write(f"RESUMED at {self.timestamp}\n")
                f.write(f"{'='*70}\n\n")

    def is_completed(self, case_idx):
        """Check if a case has been completed."""
        return case_idx in self.completed

    def _get_case_info(self, result):
        """Auto-detect dataset type and return info string for progress."""
        # MultiLogiEval
        if result.get('logic_type') and result.get('depth'):
            return f"{result.get('logic_type')}/{result.get('depth')}"
        # FOLIO
        if result.get('story_id') is not None:
            return f"s{result.get('story_id')}/e{result.get('example_id', '?')}"
        return ""

    async def save_result(self, result, case_idx):
        """Save a single result incrementally (async-safe)."""
        async with self._save_lock:
            self.results.append(result)
            self.completed.add(case_idx)

            # Append to JSONL (crash-safe incremental saving)
            with open(self.jsonl_file, 'a') as f:
                f.write(json.dumps(result) + '\n')

            # Save individual response (txt only)
            self._save_individual_response(result, case_idx)

            # Update progress with auto-detected info
            case_info = self._get_case_info(result)
            lean_verification = result.get('lean_verification') or {}
            lean_status = "PASS" if lean_verification.get('success') else "FAIL"
            num_iters = result.get('num_iterations', 1)
            with open(self.progress_file, 'a') as f:
                f.write(f"[{datetime.now().strftime('%H:%M:%S')}] "
                       f"Case {case_idx} ({case_info}): "
                       f"{result.get('ground_truth')} → {result.get('prediction')} "
                       f"{'✓' if result.get('correct') else '✗'} "
                       f"Lean={lean_status} iters={num_iters} "
                       f"(total: {len(self.results)})\n")

    def _save_individual_response(self, result, case_idx):
        """Save individual response file with LLM responses only."""
        txt_file = f"{self.responses_dir}/case_{case_idx}.txt"
        try:
            with open(txt_file, 'w') as f:
                iterations = result.get('iterations', [])
                for i, iter_data in enumerate(iterations):
                    if i > 0:
                        f.write("\n" + "=" * 70 + "\n")
                        f.write(f"ITERATION {iter_data.get('iteration', i+1)}\n")
                        f.write("=" * 70 + "\n\n")
                    f.write(iter_data.get('llm_response', '') + "\n")
        except Exception as e:
            print(f"Warning: Could not save TXT response: {e}")

    def finalize(self):
        """Generate final summary."""
        n_total = len(self.results)
        n_correct = sum(1 for r in self.results if r.get('correct', False))
        n_lean_pass = sum(1 for r in self.results
                        if r.get('lean_verification') and r['lean_verification'].get('success', False))

        summary = {
            'model': self.model,
            'dataset': self.dataset,
            'condition': self.condition,
            'total': n_total,
            'correct': n_correct,
            'accuracy': n_correct / n_total if n_total > 0 else 0,
            'lean_pass': n_lean_pass,
            'lean_pass_rate': n_lean_pass / n_total if n_total > 0 else 0,
        }

        # Add breakdown by depth/logic_type for MultiLogiEval
        if self.dataset == 'multilogieval':
            by_depth = {}
            by_logic = {}
            for r in self.results:
                d = r.get('depth', 'unknown')
                lt = r.get('logic_type', 'unknown')
                for key, store in [(d, by_depth), (lt, by_logic)]:
                    if key not in store:
                        store[key] = {'total': 0, 'correct': 0, 'lean_pass': 0}
                    store[key]['total'] += 1
                    if r.get('correct'):
                        store[key]['correct'] += 1
                    if r.get('lean_verification', {}).get('success'):
                        store[key]['lean_pass'] += 1
            summary['by_depth'] = by_depth
            summary['by_logic'] = by_logic

        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Update progress file
        with open(self.progress_file, 'a') as f:
            f.write("\n" + "=" * 70 + "\n")
            f.write("FINAL RESULTS\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Total: {n_total}\n")
            f.write(f"Accuracy: {n_correct}/{n_total} ({summary['accuracy']*100:.1f}%)\n")
            f.write(f"Lean Pass: {n_lean_pass}/{n_total} ({summary['lean_pass_rate']*100:.1f}%)\n")
            f.write(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        print(f"\n{'='*70}")
        print("Results saved:")
        print(f"{'='*70}")
        print(f"Results:        {self.jsonl_file}")
        print(f"Summary:        {self.summary_file}")
        print(f"Progress log:   {self.progress_file}")
        print(f"Responses:      {self.responses_dir}/")
        print(f"{'='*70}")

        return summary


class TwoStageSaver(BaseSaver):
    """Saver for two-stage experiments.

    Stage 1: Translation (axioms + theorem with sorry)
    Stage 2: Proving (fill in the proof)
    """

    def __init__(self, output_dir="results/twostage", dataset="folio",
                 resume_dir=None, model="gpt-5",
                 max_stage1_iterations=3, max_stage2_iterations=3,
                 concurrency=5, depths=None, logic_types=None):
        super().__init__(output_dir, dataset)
        self.dataset = dataset
        self.model = model
        self.max_stage1_iterations = max_stage1_iterations
        self.max_stage2_iterations = max_stage2_iterations
        self.concurrency = concurrency
        self.depths = depths
        self.logic_types = logic_types

        if resume_dir:
            self.base_dir = resume_dir
            print(f"Resuming in existing directory: {self.base_dir}")
        else:
            model_name = model.replace("/", "-").replace(":", "-")
            self.base_dir = f"{output_dir}/{model_name}_{dataset}_twostage_{self.timestamp}"
            os.makedirs(self.base_dir, exist_ok=True)

        self.config_file = f"{self.base_dir}/config.json"
        self.jsonl_file = f"{self.base_dir}/results.jsonl"
        self.progress_file = f"{self.base_dir}/progress.txt"
        self.summary_file = f"{self.base_dir}/summary.json"
        self.responses_dir = f"{self.base_dir}/responses"

        os.makedirs(self.responses_dir, exist_ok=True)

        self.results = []
        self.completed = set()

        # Stage-specific counters
        self.stage1_success = 0
        self.stage2_success = 0

        if not resume_dir:
            self._init_files()
        else:
            self._load_existing()

    def _init_files(self):
        """Initialize output files."""
        config = {
            "model": self.model,
            "dataset": self.dataset,
            "experiment_type": "twostage",
            "max_stage1_iterations": self.max_stage1_iterations,
            "max_stage2_iterations": self.max_stage2_iterations,
            "concurrency": self.concurrency,
            "depths": self.depths,
            "logic_types": self.logic_types,
            "started_at": self.timestamp
        }
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)

        with open(self.progress_file, 'w') as f:
            f.write(f"TwoStage - {self.dataset.upper()} - Started at {self.timestamp}\n")
            f.write(f"Model: {self.model}\n")
            f.write(f"Stage 1 max iterations: {self.max_stage1_iterations}\n")
            f.write(f"Stage 2 max iterations: {self.max_stage2_iterations}\n")
            if self.depths:
                f.write(f"Depths: {self.depths}\n")
            if self.logic_types:
                f.write(f"Logic types: {self.logic_types}\n")
            f.write("=" * 70 + "\n\n")

    def _load_existing(self):
        """Load existing results for resume support from JSONL."""
        if os.path.exists(self.jsonl_file):
            error_count = 0
            with open(self.jsonl_file, 'r') as f:
                for line in f:
                    if line.strip():
                        result = json.loads(line)
                        case_idx = result.get('case_idx')
                        if case_idx is not None:
                            # Skip error cases so they get retried
                            if result.get('error'):
                                error_count += 1
                                continue
                            self.completed.add(case_idx)
                            self.results.append(result)
                            # Update stage counters
                            if result.get('stage1_success'):
                                self.stage1_success += 1
                            if result.get('stage2_success'):
                                self.stage2_success += 1

            print(f"Loaded {len(self.completed)} existing results for resume (skipping {error_count} errors)")

            with open(self.progress_file, 'a') as f:
                f.write(f"\n{'='*70}\n")
                f.write(f"RESUMED at {self.timestamp}\n")
                f.write(f"{'='*70}\n\n")

    def is_completed(self, case_idx):
        """Check if a case has been completed."""
        return case_idx in self.completed

    def _get_case_info(self, result):
        """Auto-detect dataset type and return info string for progress."""
        if result.get('logic_type') and result.get('depth'):
            return f"{result.get('logic_type')}/{result.get('depth')}"
        if result.get('story_id') is not None:
            return f"s{result.get('story_id')}/e{result.get('example_id', '?')}"
        return ""

    async def save_result(self, result, case_idx):
        """Save a single result incrementally (async-safe)."""
        async with self._save_lock:
            self.results.append(result)
            self.completed.add(case_idx)

            # Update stage counters
            if result.get('stage1_success'):
                self.stage1_success += 1
            if result.get('stage2_success'):
                self.stage2_success += 1

            # Append to JSONL
            with open(self.jsonl_file, 'a') as f:
                f.write(json.dumps(result) + '\n')

            # Save individual response
            self._save_individual_response(result, case_idx)

            # Update progress
            case_info = self._get_case_info(result)
            s1_iters = len(result.get('stage1_iterations', []))
            s2_iters = len(result.get('stage2_iterations', []))
            fail_stage = result.get('fail_stage', '')
            status = "PASS" if result.get('stage2_success') else f"FAIL@{fail_stage}"

            with open(self.progress_file, 'a') as f:
                f.write(f"[{datetime.now().strftime('%H:%M:%S')}] "
                       f"Case {case_idx} ({case_info}): "
                       f"{result.get('ground_truth')} -> {result.get('prediction')} "
                       f"{'OK' if result.get('correct') else 'X'} "
                       f"{status} s1={s1_iters} s2={s2_iters} "
                       f"(total: {len(self.results)})\n")

    def _save_individual_response(self, result, case_idx):
        """Save responses by stage and iteration."""
        txt_file = f"{self.responses_dir}/case_{case_idx}.txt"
        try:
            with open(txt_file, 'w') as f:
                # Stage 1 iterations
                f.write("=" * 70 + "\n")
                f.write("STAGE 1: TRANSLATION\n")
                f.write("=" * 70 + "\n\n")
                for i, iter_data in enumerate(result.get('stage1_iterations', [])):
                    if i > 0:
                        f.write("\n" + "-" * 50 + "\n")
                    f.write(f"--- Iteration {i+1} ---\n\n")
                    # Write reasoning content if present (DeepSeek-R1)
                    if iter_data.get('reasoning_content'):
                        f.write("=== REASONING ===\n")
                        f.write(iter_data.get('reasoning_content', '') + "\n\n")
                        f.write("=== RESPONSE ===\n")
                    f.write(iter_data.get('llm_response', '') + "\n")
                    if iter_data.get('lean_error'):
                        f.write(f"\n[LEAN ERROR]: {iter_data.get('lean_error')}\n")

                # Stage 2 iterations (if Stage 1 succeeded)
                if result.get('stage2_iterations'):
                    f.write("\n\n" + "=" * 70 + "\n")
                    f.write("STAGE 2: PROVING\n")
                    f.write("=" * 70 + "\n\n")
                    for i, iter_data in enumerate(result.get('stage2_iterations', [])):
                        if i > 0:
                            f.write("\n" + "-" * 50 + "\n")
                        f.write(f"--- Iteration {i+1} ---\n\n")
                        if iter_data.get('reasoning_content'):
                            f.write("=== REASONING ===\n")
                            f.write(iter_data.get('reasoning_content', '') + "\n\n")
                            f.write("=== RESPONSE ===\n")
                        f.write(iter_data.get('llm_response', '') + "\n")
                        if iter_data.get('lean_error'):
                            f.write(f"\n[LEAN ERROR]: {iter_data.get('lean_error')}\n")
        except Exception as e:
            print(f"Warning: Could not save TXT response: {e}")

    def finalize(self):
        """Generate final summary with stage breakdown."""
        n_total = len(self.results)
        n_correct = sum(1 for r in self.results if r.get('correct', False))
        n_stage1_success = sum(1 for r in self.results if r.get('stage1_success', False))
        n_stage2_success = sum(1 for r in self.results if r.get('stage2_success', False))
        n_stage1_fail = sum(1 for r in self.results if r.get('fail_stage') == 'stage1')
        n_stage2_fail = sum(1 for r in self.results if r.get('fail_stage') == 'stage2')

        summary = {
            'model': self.model,
            'dataset': self.dataset,
            'experiment_type': 'twostage',
            'total': n_total,
            'correct': n_correct,
            'accuracy': n_correct / n_total if n_total > 0 else 0,
            'stage1_success': n_stage1_success,
            'stage1_success_rate': n_stage1_success / n_total if n_total > 0 else 0,
            'stage2_success': n_stage2_success,
            'stage2_success_rate': n_stage2_success / n_stage1_success if n_stage1_success > 0 else 0,
            'stage1_fail': n_stage1_fail,
            'stage2_fail': n_stage2_fail,
        }

        # Add breakdown by depth/logic_type for MultiLogiEval
        if self.dataset == 'multilogieval':
            by_depth = {}
            by_logic = {}
            for r in self.results:
                d = r.get('depth', 'unknown')
                lt = r.get('logic_type', 'unknown')
                for key, store in [(d, by_depth), (lt, by_logic)]:
                    if key not in store:
                        store[key] = {'total': 0, 'correct': 0, 's1_pass': 0, 's2_pass': 0}
                    store[key]['total'] += 1
                    if r.get('correct'):
                        store[key]['correct'] += 1
                    if r.get('stage1_success'):
                        store[key]['s1_pass'] += 1
                    if r.get('stage2_success'):
                        store[key]['s2_pass'] += 1
            summary['by_depth'] = by_depth
            summary['by_logic'] = by_logic

        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Update progress file
        with open(self.progress_file, 'a') as f:
            f.write("\n" + "=" * 70 + "\n")
            f.write("FINAL RESULTS\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Total: {n_total}\n")
            f.write(f"Accuracy: {n_correct}/{n_total} ({summary['accuracy']*100:.1f}%)\n")
            f.write(f"Stage 1 Success: {n_stage1_success}/{n_total} ({summary['stage1_success_rate']*100:.1f}%)\n")
            s2_rate = summary['stage2_success_rate'] * 100 if n_stage1_success > 0 else 0
            f.write(f"Stage 2 Success: {n_stage2_success}/{n_stage1_success} ({s2_rate:.1f}%)\n")
            f.write(f"Stage 1 Failures: {n_stage1_fail}\n")
            f.write(f"Stage 2 Failures: {n_stage2_fail}\n")
            f.write(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        print(f"\n{'='*70}")
        print("Results saved:")
        print(f"{'='*70}")
        print(f"Results:        {self.jsonl_file}")
        print(f"Summary:        {self.summary_file}")
        print(f"Progress log:   {self.progress_file}")
        print(f"Responses:      {self.responses_dir}/")
        print(f"{'='*70}")

        return summary
