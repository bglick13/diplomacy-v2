import re
from typing import Dict, List

# Regex patterns for our specific XML tags
# DOTALL means . matches newlines, enabling multi-line extraction
RE_ORDERS = re.compile(r"<orders>(.*?)</orders>", re.DOTALL | re.IGNORECASE)
RE_THOUGHT = re.compile(r"<analysis>(.*?)</analysis>", re.DOTALL | re.IGNORECASE)
RE_PRESS = re.compile(
    r"<communication>(.*?)</communication>", re.DOTALL | re.IGNORECASE
)
RE_TRUTH = re.compile(r"<truth_status>(.*?)</truth_status>", re.DOTALL | re.IGNORECASE)


def extract_orders(llm_output: str) -> List[str]:
    """
    Extracts orders from the <orders> block.
    Returns a clean list of strings.
    """
    match = RE_ORDERS.search(llm_output)
    if not match:
        return []

    raw_block = match.group(1)
    # Split by newlines and clean whitespace
    orders = [line.strip() for line in raw_block.split("\n") if line.strip()]
    return orders


def extract_metadata(llm_output: str) -> Dict[str, str]:
    """
    Extracts structured intent tags.
    """
    meta = {}

    thought_match = RE_THOUGHT.search(llm_output)
    if thought_match:
        meta["thought"] = thought_match.group(1).strip()

    press_match = RE_PRESS.search(llm_output)
    if press_match:
        # Inside communication, look for truth status
        comm_block = press_match.group(1)
        truth_match = RE_TRUTH.search(comm_block)
        if truth_match:
            meta["truth_status"] = truth_match.group(1).strip().upper()

    return meta
