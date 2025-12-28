"""Answer extraction and normalization utilities."""

import re


def normalize_answer(answer, answer_format="yes_no"):
    """Normalize answer to standard format.

    Args:
        answer: The answer to normalize
        answer_format: "yes_no", "true_false", "bidir_true", or "bidir_false"

    Returns:
        str: Normalized answer
    """
    if not answer:
        return 'Failure' if answer_format.startswith('bidir') else 'Uncertain'

    low = answer.lower().strip()

    if answer_format == "yes_no":
        # Multi-LogiEval uses Yes/No/Uncertain format
        if low in ['yes', 'y', 'true', 't', '1']:
            return 'Yes'
        elif low in ['no', 'n', 'false', 'f', '0']:
            return 'No'
        elif low in ['unknown', 'uncertain', 'u']:
            return 'Uncertain'
    elif answer_format == "true_false":
        # FOLIO uses True/False/Uncertain format
        if low in ['true', 't', 'yes', 'y']:
            return 'True'
        elif low in ['false', 'f', 'no', 'n']:
            return 'False'
        elif low in ['unknown', 'uncertain', 'u']:
            return 'Uncertain'
    elif answer_format == "bidir_true":
        # Bidirectional true: preserve Yes vs True based on input
        if low in ['yes', 'y']:
            return 'Yes'
        elif low in ['true', 't', 'success']:
            return 'True'
        elif low in ['failure', 'failed', 'fail', 'f']:
            return 'Failure'
    elif answer_format == "bidir_false":
        # Bidirectional false: preserve No vs False based on input
        if low in ['no', 'n']:
            return 'No'
        elif low in ['false', 'f', 'success']:
            return 'False'
        elif low in ['failure', 'failed', 'fail']:
            return 'Failure'

    return 'Failure' if answer_format.startswith('bidir') else 'Uncertain'


def parse_answer(response, answer_format="true_false"):
    """Extract answer from model response.

    Args:
        response: The model's text response
        answer_format: "true_false", "yes_no", "bidir_true", or "bidir_false"

    Returns:
        tuple: (answer, parse_status)
        Status: SUCCESS, FALLBACK, EMPTY, PARSE_FAILED
    """
    if not response or not response.strip():
        return None, "EMPTY"

    # Set patterns based on format
    if answer_format == "bidir_true":
        # Accept both True and Yes (dataset-specific)
        pattern = r'ANSWER:\s*(True|Yes|Failure|Failed)'
        fallback_pattern = r'\b(True|Yes|Failure|Failed)\b'
    elif answer_format == "bidir_false":
        # Accept both False and No (dataset-specific)
        pattern = r'ANSWER:\s*(False|No|Failure|Failed)'
        fallback_pattern = r'\b(False|No|Failure|Failed)\b'
    elif answer_format == "true_false":
        pattern = r'ANSWER:\s*(True|False|Unknown|Uncertain)'
        fallback_pattern = r'\b(True|False|Unknown|Uncertain)\b'
    else:  # yes_no
        # Accept both Yes/No and True/False (normalize later)
        pattern = r'ANSWER:\s*(Yes|No|True|False|Unknown|Uncertain)'
        fallback_pattern = r'\b(Yes|No|True|False|Unknown|Uncertain)\b'

    # Look for ANSWER: format first
    answer_match = re.search(pattern, response, re.IGNORECASE)
    if answer_match:
        answer = normalize_answer(answer_match.group(1), answer_format)
        return answer, "SUCCESS"

    # Fallback: find all matches and take last one
    all_answers = re.findall(fallback_pattern, response, re.IGNORECASE)
    if all_answers:
        answer = normalize_answer(all_answers[-1], answer_format)
        return answer, "FALLBACK"

    return None, "PARSE_FAILED"


# Aliases for backward compatibility
def parse_folio_answer(response, return_status=False):
    """Extract True/False/Unknown answer from FOLIO response."""
    answer, status = parse_answer(response, answer_format="true_false")
    if return_status:
        return answer, status
    return answer


def parse_multilogieval_answer(response):
    """Extract Yes/No/Unknown answer from MultiLogiEval response."""
    return parse_answer(response, answer_format="yes_no")
