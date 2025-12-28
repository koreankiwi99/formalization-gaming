"""Dataset loading utilities."""

import json
from pathlib import Path


def load_folio(folio_path: str = None) -> list:
    """Load FOLIO validation dataset."""
    if folio_path is None:
        folio_path = "data/folio/original/folio-validation.json"
    with open(folio_path, 'r') as f:
        data = json.load(f)

    cases = []
    for idx, entry in enumerate(data):
        cases.append({
            'idx': idx,
            'story_id': entry.get('story_id', 0),
            'example_id': entry.get('example_id', 0),
            'premises': entry.get('premises', ''),
            'conclusion': entry.get('conclusion', ''),
            'ground_truth': entry.get('label', 'Unknown')
        })
    return cases


def load_multilogieval(depths: list, logic_types: list) -> list:
    """Load MultiLogiEval dataset for specified depths and logic types."""
    data_dir = Path("data/multilogieval/original/data")
    cases = []

    for depth in depths:
        depth_dir = data_dir / f"{depth}_Data"
        if not depth_dir.exists():
            print(f"Warning: {depth_dir} does not exist")
            continue

        for logic_type in logic_types:
            logic_dir = depth_dir / logic_type
            if not logic_dir.exists():
                continue

            for json_file in logic_dir.glob("*.json"):
                try:
                    with open(json_file, encoding='utf-8') as f:
                        data = json.load(f)
                except UnicodeDecodeError:
                    with open(json_file, encoding='latin-1') as f:
                        data = json.load(f)

                rule = data.get('rule', json_file.stem)
                samples = data.get('samples', [])

                for sample in samples:
                    cases.append({
                        'id': sample.get('id'),
                        'context': sample.get('context', ''),
                        'question': sample.get('question', ''),
                        'ground_truth': sample.get('answer', '').lower(),
                        'logic_type': logic_type,
                        'depth': depth,
                        'rule': rule
                    })

    return cases


def load_multilogieval_sampled(data_file: str) -> list:
    """Load sampled MultiLogiEval dataset from JSON file."""
    with open(data_file, 'r') as f:
        data = json.load(f)

    cases = []
    for sample in data:
        cases.append({
            'idx': sample.get('idx'),  # Unique index from data file
            'id': sample.get('id'),
            'context': sample.get('context', ''),
            'question': sample.get('question', ''),
            'ground_truth': sample.get('answer', '').lower(),
            'logic_type': sample.get('logic', 'fol'),
            'depth': sample.get('depth', ''),
            'rule': sample.get('rule', '')
        })

    return cases
