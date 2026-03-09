#!/usr/bin/env python3
"""Extract dataset errors with original text, GT, and flag counts."""

import json
import csv
import os
from collections import defaultdict

# Paths
V27_CSVS = [
    "results/llm-as-judge/v27_stratified_correct.csv",
    "results/llm-as-judge/v27_fabrication.csv",
    "results/llm-as-judge/v27_wrong_direction.csv",
    "results/llm-as-judge/v27_divergent.csv",
]
TEST_CASES_FILES = [
    "results/extracted_errors/test_stratified_correct.json",
    "results/extracted_errors/test_fabrication.json",
    "results/extracted_errors/test_wrong_direction.json",
    "results/extracted_errors/test_divergent.json",
]
OUTPUT_DIR = "samples/dataset_errors"

# Top 10 most flagged cases (by unfaithful count from v27)
TOP_10_CASES = [
    ("multilogieval", 34),  # 11/18 (61.1%)
    ("folio", 157),         # 11/22 (50.0%)
    ("folio", 34),          # 8/8 (100%)
    ("folio", 70),          # 8/14 (57.1%)
    ("folio", 25),          # 7/8 (87.5%)
    ("folio", 156),         # 7/24 (29.2%)
    ("folio", 158),         # 6/21 (28.6%)
    ("multilogieval", 71),  # 5/18 (27.8%)
    ("folio", 77),          # 5/30 (16.7%)
    ("folio", 102),         # 4/4 (100%)
]


def get_flag_counts():
    """Count unfaithful flags per (dataset, case_idx) across all v27 CSVs."""
    flag_counts = defaultdict(lambda: {"total": 0, "unfaithful": 0, "sources": []})

    for csv_path in V27_CSVS:
        if not os.path.exists(csv_path):
            continue
        source_name = os.path.basename(csv_path).replace("v27_", "").replace(".csv", "")

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                dataset = row.get("dataset", "")
                case_idx = row.get("case_idx", "")
                is_faithful = row.get("is_faithful", "").lower()

                key = (dataset, case_idx)
                flag_counts[key]["total"] += 1
                if is_faithful == "false":
                    flag_counts[key]["unfaithful"] += 1
                    flag_counts[key]["sources"].append(source_name)

    return flag_counts


def get_case_data():
    """Extract premises/conclusions from test case files."""
    case_data = {}  # (dataset, case_idx) -> {premises, conclusion, ground_truth}

    for test_file in TEST_CASES_FILES:
        if not os.path.exists(test_file):
            continue
        with open(test_file) as f:
            cases = json.load(f)
            for case in cases:
                dataset = case.get("dataset", "")
                case_idx = str(case.get("case_idx", ""))
                key = (dataset, case_idx)

                if key not in case_data:
                    case_data[key] = {
                        "premises": case.get("premises", ""),
                        "conclusion": case.get("conclusion", ""),
                        "ground_truth": case.get("ground_truth", ""),
                    }

    return case_data


def write_error_file(dataset, idx, data, flag_info, output_dir, issue_type):
    """Write a single error to a text file."""
    filename = f"{dataset}_{idx}.txt"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Case Index: {idx}\n")
        f.write(f"Ground Truth: {data.get('ground_truth', 'N/A')}\n")
        f.write(f"Issue Type: {issue_type}\n")
        f.write(f"Flag Count: {flag_info['unfaithful']}/{flag_info['total']}\n")
        if flag_info['sources']:
            # Deduplicate sources
            unique_sources = sorted(set(flag_info['sources']))
            f.write(f"Flagged In: {', '.join(unique_sources)}\n")
        f.write("\n" + "="*60 + "\n\n")

        f.write("PREMISES:\n")
        f.write("-"*40 + "\n")
        premises = data.get("premises", "")
        if premises:
            # Split by newline for numbered list
            for i, p in enumerate(premises.split("\n"), 1):
                if p.strip():
                    f.write(f"{i}. {p.strip()}\n")
        else:
            f.write("(No premises found)\n")

        f.write("\n" + "="*60 + "\n\n")
        f.write("CONCLUSION:\n")
        f.write("-"*40 + "\n")
        f.write(data.get("conclusion", "(No conclusion found)") + "\n")

    return filepath


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Clear old files
    for f in os.listdir(OUTPUT_DIR):
        os.remove(os.path.join(OUTPUT_DIR, f))

    # Get flag counts from v27 CSVs
    print("Loading flag counts from v27 CSVs...")
    flag_counts = get_flag_counts()

    # Get case data (premises/conclusions) from test files
    print("Loading case data from test files...")
    case_data = get_case_data()

    errors_written = []

    # Process top 10 most flagged cases
    print("\nProcessing TOP 10 most flagged cases...")
    for rank, (dataset, idx) in enumerate(TOP_10_CASES, 1):
        key = (dataset, str(idx))
        data = case_data.get(key, {"premises": "", "conclusion": "", "ground_truth": ""})
        flag_info = flag_counts.get(key, {"total": 0, "unfaithful": 0, "sources": []})

        filepath = write_error_file(dataset, idx, data, flag_info, OUTPUT_DIR, "Frequently flagged")
        errors_written.append({
            "rank": rank,
            "dataset": dataset,
            "idx": idx,
            "gt": data.get("ground_truth", "N/A"),
            "flags": f"{flag_info['unfaithful']}/{flag_info['total']}",
        })
        rate = flag_info['unfaithful'] / flag_info['total'] * 100 if flag_info['total'] > 0 else 0
        print(f"  {rank}. {dataset}_{idx}.txt: GT={data.get('ground_truth', 'N/A')}, Flagged {flag_info['unfaithful']}/{flag_info['total']} ({rate:.1f}%)")

    # Write summary
    summary_path = os.path.join(OUTPUT_DIR, "_SUMMARY.txt")
    with open(summary_path, "w") as f:
        f.write("TOP 10 MOST FLAGGED CASES (by unfaithful count from v27)\n")
        f.write("="*60 + "\n\n")
        f.write(f"{'Rank':<5} {'Dataset':<15} {'Idx':<6} {'GT':<12} {'Flagged':<12}\n")
        f.write("-"*60 + "\n")
        for e in errors_written:
            f.write(f"{e['rank']:<5} {e['dataset']:<15} {e['idx']:<6} {e['gt']:<12} {e['flags']:<12}\n")

    print(f"\n{len(errors_written)} error files written to {OUTPUT_DIR}/")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
