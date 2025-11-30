import re
from typing import Dict, List

# Regex patterns for our specific XML tags
# DOTALL means . matches newlines, enabling multi-line extraction
RE_ORDERS = re.compile(r"<orders>(.*?)</orders>", re.DOTALL | re.IGNORECASE)
# Fallback: Match <orders> without closing tag (when stop token truncates)
RE_ORDERS_OPEN = re.compile(r"<orders>(.*?)$", re.DOTALL | re.IGNORECASE)
RE_THOUGHT = re.compile(r"<analysis>(.*?)</analysis>", re.DOTALL | re.IGNORECASE)
RE_PRESS = re.compile(
    r"<communication>(.*?)</communication>", re.DOTALL | re.IGNORECASE
)
RE_TRUTH = re.compile(r"<truth_status>(.*?)</truth_status>", re.DOTALL | re.IGNORECASE)

# Pattern to detect if text looks like Diplomacy orders (starts with unit type)
# A = Army, F = Fleet, B = Build, D = Disband
RE_ORDER_LINE = re.compile(r"^[AFBD]\s+[A-Z]{3}", re.MULTILINE)


def extract_orders(llm_output: str) -> List[str]:
    """
    Extracts orders from the <orders> block OR directly from output.

    Handles multiple formats:
    1. Complete <orders>...</orders> blocks
    2. Truncated responses where stop=["</orders>"] cut off closing tag
    3. Raw move strings when prompt was primed with <orders>

    NOTE: When prompt ends with '<orders>\n', the model output won't include
    the opening tag - it will be raw moves directly.
    """
    # Try complete pattern first
    match = RE_ORDERS.search(llm_output)
    if match:
        raw_block = match.group(1)
    else:
        # Fallback: try open-ended pattern (stop token truncated closing tag)
        match = RE_ORDERS_OPEN.search(llm_output)
        if match:
            raw_block = match.group(1)
        else:
            # Final fallback: if output looks like raw orders (no tags),
            # use it directly. This handles the primed prompt case.
            if RE_ORDER_LINE.search(llm_output):
                raw_block = llm_output
            else:
                return []

    # Split by newlines and clean whitespace
    orders = [line.strip() for line in raw_block.split("\n") if line.strip()]

    # Filter to only valid-looking order lines
    # A = Army, F = Fleet, B = Build (standalone), D = Disband (standalone)
    # Also accept WAIVE for adjustment phases
    valid_orders = []
    for o in orders:
        if not o:
            continue
        if o.upper() == "WAIVE":
            valid_orders.append("WAIVE")
        elif o[0] in ("A", "F"):
            valid_orders.append(o)

    return valid_orders


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
