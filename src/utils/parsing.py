import re

# Regex patterns for our specific XML tags
# DOTALL means . matches newlines, enabling multi-line extraction
RE_ORDERS = re.compile(r"<orders>(.*?)</orders>", re.DOTALL | re.IGNORECASE)
# Fallback: Match <orders> without closing tag (when stop token truncates)
RE_ORDERS_OPEN = re.compile(r"<orders>(.*?)$", re.DOTALL | re.IGNORECASE)
RE_THOUGHT = re.compile(r"<analysis>(.*?)</analysis>", re.DOTALL | re.IGNORECASE)
RE_PRESS = re.compile(r"<communication>(.*?)</communication>", re.DOTALL | re.IGNORECASE)
RE_TRUTH = re.compile(r"<truth_status>(.*?)</truth_status>", re.DOTALL | re.IGNORECASE)

# Pattern to detect if text looks like Diplomacy orders (starts with unit type)
# A = Army, F = Fleet
# Requires proper order format with action verb to avoid false positives like "AUM: Correct..."
# Uses lookahead patterns to properly detect each action type:
#   - Movement: "- " followed by destination
#   - Hold: "H" at end or followed by whitespace
#   - Support/Convoy/Retreat: "S ", "C ", "R " followed by unit
#   - Build/Disband: "B" or "D" at end or followed by whitespace
RE_ORDER_LINE = re.compile(
    r"^[AF]\s+[A-Z]{3}(/[A-Z]{2})?\s+"  # Unit + location (with optional coast)
    r"(-\s|H(?:\s|$)|S\s|C\s|B(?:\s|$)|D(?:\s|$)|R\s)",  # Action indicator
    re.MULTILINE | re.IGNORECASE,
)

# Pattern to validate a complete order line (stricter than detection)
# Matches: A PAR - BUR, A PAR H, A PAR S A BUR, F NTH C A LON - BRE, A PAR B, A PAR D, A PAR R BUR
RE_VALID_ORDER = re.compile(
    r"^[AF]\s+[A-Z]{3}(/[A-Z]{2})?\s+"  # Unit type + location (with optional coast)
    r"(-\s+[A-Z]{3}(/[A-Z]{2})?"  # Movement: - LOC
    r"|H"  # Hold
    r"|S\s+[AF]\s+[A-Z]{3}(/[A-Z]{2})?(\s+-\s+[A-Z]{3}(/[A-Z]{2})?)?"  # Support
    r"|C\s+[AF]\s+[A-Z]{3}(/[A-Z]{2})?\s+-\s+[A-Z]{3}(/[A-Z]{2})?"  # Convoy
    r"|B"  # Build
    r"|D"  # Disband
    r"|R\s+[A-Z]{3}(/[A-Z]{2})?"  # Retreat
    r")(\s+VIA)?$",  # Optional VIA for convoy routes
    re.IGNORECASE,
)


def extract_orders(llm_output: str) -> list[str]:
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

    # Filter to only valid-looking order lines using strict validation
    # This prevents false positives like "AUM: Correct..." from being extracted
    valid_orders = []
    for o in orders:
        if not o:
            continue
        o_stripped = o.strip()
        if o_stripped.upper() == "WAIVE":
            valid_orders.append("WAIVE")
        elif RE_VALID_ORDER.match(o_stripped):
            valid_orders.append(o_stripped)

    return valid_orders


def extract_metadata(llm_output: str) -> dict[str, str]:
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
