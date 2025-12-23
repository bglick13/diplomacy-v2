import pytest

from src.utils.parsing import RE_ORDER_LINE, RE_VALID_ORDER, extract_metadata, extract_orders

# Sample LLM Outputs
CLEAN_OUTPUT = """
<analysis>
Germany is weak. I should attack.
</analysis>

<communication>
  <truth_status>LIE</truth_status>
  <message>I am holding.</message>
</communication>

<orders>
A PAR - BUR
F BRE - MAO
</orders>
"""

MESSY_OUTPUT = """
<orders>
   A PAR - BUR

   F BRE H
</orders>
<analysis>Thinking...</analysis>
"""

NO_TAGS_OUTPUT = "I think I should move A PAR to BUR."

# Simulates when stop=["</orders>"] truncates the closing tag
TRUNCATED_OUTPUT = """<orders>
A PAR - BUR
F BRE - MAO
A MAR H
"""

# Very short response with no orders tag at all
SHORT_NO_TAG = "I will move north."

# Raw orders (when prompt was primed with <orders>\n)
RAW_ORDERS_OUTPUT = """A PAR - BUR
F BRE - MAO
A MAR H
"""


def test_extract_orders_clean():
    orders = extract_orders(CLEAN_OUTPUT)
    assert len(orders) == 2
    assert "A PAR - BUR" in orders
    assert "F BRE - MAO" in orders


def test_extract_orders_messy():
    """Test handling of extra whitespace and empty lines inside tags."""
    orders = extract_orders(MESSY_OUTPUT)
    assert len(orders) == 2
    assert "A PAR - BUR" in orders
    assert "F BRE H" in orders


def test_extract_orders_missing():
    orders = extract_orders(NO_TAGS_OUTPUT)
    assert orders == []


def test_extract_orders_truncated():
    """Test handling of truncated output (stop token cut off closing tag)."""
    orders = extract_orders(TRUNCATED_OUTPUT)
    assert len(orders) == 3
    assert "A PAR - BUR" in orders
    assert "F BRE - MAO" in orders
    assert "A MAR H" in orders


def test_extract_orders_short_no_tag():
    """Test very short response with no orders tag."""
    orders = extract_orders(SHORT_NO_TAG)
    assert orders == []


def test_extract_orders_raw():
    """Test raw orders without tags (primed prompt case)."""
    orders = extract_orders(RAW_ORDERS_OUTPUT)
    assert len(orders) == 3
    assert "A PAR - BUR" in orders
    assert "F BRE - MAO" in orders
    assert "A MAR H" in orders


def test_extract_metadata_full():
    meta = extract_metadata(CLEAN_OUTPUT)
    assert meta["thought"] == "Germany is weak. I should attack."
    assert meta["truth_status"] == "LIE"


def test_extract_metadata_partial():
    """Test when only some tags are present."""
    meta = extract_metadata(MESSY_OUTPUT)
    assert meta["thought"] == "Thinking..."
    assert "truth_status" not in meta


# =============================================================================
# Strict Order Validation Tests
# =============================================================================


def test_extract_orders_rejects_false_positives():
    """Test that text like 'AUM: Correct...' is NOT extracted as an order."""
    # This was a real bug where conversational text leaked through
    bad_output = """<orders>
AUM: Correct. I will monitor the map and prepare for the spring phase.
A PAR - BUR
</orders>"""
    orders = extract_orders(bad_output)
    # Should only get the valid order, not the "AUM:" text
    assert len(orders) == 1
    assert "A PAR - BUR" in orders
    assert not any("AUM:" in o for o in orders)


def test_extract_orders_rejects_conversational_text():
    """Test various conversational patterns that start with A/F."""
    # These should all be rejected as not valid orders
    bad_lines = [
        "AUM: Let me think about this.",
        "A good strategy would be...",
        "FRANCE: I will support you.",
        "A: Here are my orders:",
        "After consideration, I move...",
        "Finally, I hold.",
    ]
    for line in bad_lines:
        orders = extract_orders(line)
        assert orders == [], f"'{line}' should not produce orders"


def test_extract_orders_accepts_all_order_types():
    """Test that all valid Diplomacy order types are accepted."""
    all_orders = """<orders>
A PAR - BUR
F BRE - MAO
A MAR H
A PAR S A BUR
A PAR S A BUR - MAR
F NTH C A LON - BRE
A PAR B
F LON D
A MUN R TYR
WAIVE
</orders>"""
    orders = extract_orders(all_orders)
    assert len(orders) == 10
    assert "A PAR - BUR" in orders
    assert "F BRE - MAO" in orders
    assert "A MAR H" in orders
    assert "A PAR S A BUR" in orders
    assert "A PAR S A BUR - MAR" in orders
    assert "F NTH C A LON - BRE" in orders
    assert "A PAR B" in orders
    assert "F LON D" in orders
    assert "A MUN R TYR" in orders
    assert "WAIVE" in orders


def test_extract_orders_coastal_locations():
    """Test orders with coastal notation."""
    coastal_orders = """<orders>
F NWG - STP/NC
F AEG - BUL/SC
</orders>"""
    orders = extract_orders(coastal_orders)
    assert len(orders) == 2
    assert "F NWG - STP/NC" in orders
    assert "F AEG - BUL/SC" in orders


def test_extract_orders_via_convoy():
    """Test movement orders with VIA convoy."""
    via_orders = """<orders>
A LON - BRE VIA
</orders>"""
    orders = extract_orders(via_orders)
    assert len(orders) == 1
    assert "A LON - BRE VIA" in orders


# =============================================================================
# RE_ORDER_LINE Detection Tests (Raw Order Fallback Path)
# =============================================================================
# These tests verify the regex that detects if text looks like Diplomacy orders.
# This is critical for the fallback path when <orders> tags are missing
# (e.g., when prompt is primed with '<orders>\n' and model outputs raw orders).


class TestOrderLineDetection:
    """Tests for RE_ORDER_LINE regex pattern matching."""

    @pytest.mark.parametrize(
        "order",
        [
            "A PAR - BUR",
            "F BRE - MAO",
            "A VIE - GAL",
            "F NTH - NWY",
        ],
    )
    def test_detects_movement_orders(self, order):
        """Movement orders (unit - destination) must be detected."""
        assert RE_ORDER_LINE.search(order), f"Failed to detect movement: '{order}'"

    @pytest.mark.parametrize(
        "order",
        [
            "F NWG - STP/NC",
            "F AEG - BUL/SC",
            "F MAO - SPA/SC",
        ],
    )
    def test_detects_movement_with_coast(self, order):
        """Movement orders with coastal notation must be detected."""
        assert RE_ORDER_LINE.search(order), f"Failed to detect coastal movement: '{order}'"

    @pytest.mark.parametrize(
        "order",
        [
            "A PAR H",
            "F BRE H",
            "A MUN H",
        ],
    )
    def test_detects_hold_orders(self, order):
        """Hold orders must be detected."""
        assert RE_ORDER_LINE.search(order), f"Failed to detect hold: '{order}'"

    @pytest.mark.parametrize(
        "order",
        [
            "A PAR S A BUR",
            "F MAO S F BRE",
            "A MUN S A BER - SIL",
            "F NTH S A YOR - NWY",
        ],
    )
    def test_detects_support_orders(self, order):
        """Support orders must be detected."""
        assert RE_ORDER_LINE.search(order), f"Failed to detect support: '{order}'"

    @pytest.mark.parametrize(
        "order",
        [
            "F NTH C A LON - BRE",
            "F ENG C A WAL - PIC",
            "F MAO C A POR - NAF",
        ],
    )
    def test_detects_convoy_orders(self, order):
        """Convoy orders must be detected."""
        assert RE_ORDER_LINE.search(order), f"Failed to detect convoy: '{order}'"

    @pytest.mark.parametrize(
        "order",
        [
            "A PAR B",
            "F BRE B",
            "A MUN B",
            "F LON B",
        ],
    )
    def test_detects_build_orders(self, order):
        """Build orders must be detected."""
        assert RE_ORDER_LINE.search(order), f"Failed to detect build: '{order}'"

    @pytest.mark.parametrize(
        "order",
        [
            "A PAR D",
            "F BRE D",
            "A MUN D",
        ],
    )
    def test_detects_disband_orders(self, order):
        """Disband orders must be detected."""
        assert RE_ORDER_LINE.search(order), f"Failed to detect disband: '{order}'"

    @pytest.mark.parametrize(
        "order",
        [
            "A MUN R TYR",
            "F NTH R NWG",
            "A BUR R PAR",
        ],
    )
    def test_detects_retreat_orders(self, order):
        """Retreat orders must be detected."""
        assert RE_ORDER_LINE.search(order), f"Failed to detect retreat: '{order}'"

    def test_detects_via_convoy(self):
        """Movement with VIA convoy must be detected."""
        assert RE_ORDER_LINE.search("A LON - BRE VIA")

    @pytest.mark.parametrize(
        "text",
        [
            "AUM: Correct. I will monitor the map.",
            "A good strategy would be...",
            "FRANCE: I will support you.",
            "A: Here are my orders:",
            "After consideration, I move...",
            "Finally, I hold.",
            "Austria should attack Italy.",
            "For the alliance, we need...",
        ],
    )
    def test_rejects_conversational_text(self, text):
        """Conversational text starting with A/F must NOT be detected as orders."""
        assert not RE_ORDER_LINE.search(text), f"False positive: '{text}'"


class TestRawOrderExtraction:
    """Tests for extract_orders() with raw orders (no <orders> tags).

    This tests the fallback path that was broken when RE_ORDER_LINE
    failed to detect movement orders.
    """

    def test_raw_movement_orders(self):
        """Raw movement orders without tags must be extracted."""
        raw = "A PAR - BUR\nF BRE - MAO\nA MAR - SPA"
        orders = extract_orders(raw)
        assert len(orders) == 3
        assert "A PAR - BUR" in orders
        assert "F BRE - MAO" in orders
        assert "A MAR - SPA" in orders

    def test_raw_hold_orders(self):
        """Raw hold orders without tags must be extracted."""
        raw = "A PAR H\nF BRE H"
        orders = extract_orders(raw)
        assert len(orders) == 2
        assert "A PAR H" in orders
        assert "F BRE H" in orders

    def test_raw_support_orders(self):
        """Raw support orders without tags must be extracted."""
        raw = "A PAR S A BUR\nA MUN S A BER - SIL"
        orders = extract_orders(raw)
        assert len(orders) == 2
        assert "A PAR S A BUR" in orders
        assert "A MUN S A BER - SIL" in orders

    def test_raw_convoy_orders(self):
        """Raw convoy orders without tags must be extracted."""
        raw = "F NTH C A LON - BRE"
        orders = extract_orders(raw)
        assert len(orders) == 1
        assert "F NTH C A LON - BRE" in orders

    def test_raw_build_orders(self):
        """Raw build orders without tags must be extracted."""
        raw = "A PAR B\nF BRE B"
        orders = extract_orders(raw)
        assert len(orders) == 2
        assert "A PAR B" in orders
        assert "F BRE B" in orders

    def test_raw_disband_orders(self):
        """Raw disband orders without tags must be extracted."""
        raw = "A MUN D\nF NTH D"
        orders = extract_orders(raw)
        assert len(orders) == 2
        assert "A MUN D" in orders
        assert "F NTH D" in orders

    def test_raw_retreat_orders(self):
        """Raw retreat orders without tags must be extracted."""
        raw = "A MUN R TYR\nF NTH R NWG"
        orders = extract_orders(raw)
        assert len(orders) == 2
        assert "A MUN R TYR" in orders
        assert "F NTH R NWG" in orders

    def test_raw_mixed_orders(self):
        """Mix of all order types without tags must be extracted."""
        raw = """A PAR - BUR
F BRE H
A MUN S A BER
F NTH C A LON - NWY
A MAR - SPA"""
        orders = extract_orders(raw)
        assert len(orders) == 5
        assert "A PAR - BUR" in orders
        assert "F BRE H" in orders
        assert "A MUN S A BER" in orders
        assert "F NTH C A LON - NWY" in orders
        assert "A MAR - SPA" in orders

    def test_raw_coastal_orders(self):
        """Raw orders with coastal notation must be extracted."""
        raw = "F NWG - STP/NC\nF AEG - BUL/SC"
        orders = extract_orders(raw)
        assert len(orders) == 2
        assert "F NWG - STP/NC" in orders
        assert "F AEG - BUL/SC" in orders

    def test_raw_via_convoy(self):
        """Raw movement with VIA must be extracted."""
        raw = "A LON - BRE VIA"
        orders = extract_orders(raw)
        assert len(orders) == 1
        assert "A LON - BRE VIA" in orders

    def test_raw_waive_with_other_orders(self):
        """WAIVE mixed with other raw orders must be extracted."""
        # WAIVE alone won't trigger fallback detection (doesn't start with A/F),
        # but it should work when mixed with regular orders
        raw = "A PAR B\nWAIVE"
        orders = extract_orders(raw)
        assert len(orders) == 2
        assert "A PAR B" in orders
        assert "WAIVE" in orders

    def test_waive_alone_needs_tags(self):
        """WAIVE alone requires <orders> tags since it doesn't look like a unit order."""
        # This is expected behavior - "WAIVE" alone doesn't trigger fallback detection
        raw = "WAIVE"
        orders = extract_orders(raw)
        assert orders == []  # No detection without tags

        # But with tags it works
        tagged = "<orders>\nWAIVE\n</orders>"
        orders = extract_orders(tagged)
        assert len(orders) == 1
        assert "WAIVE" in orders


class TestValidOrderRegex:
    """Tests for RE_VALID_ORDER regex (stricter validation)."""

    @pytest.mark.parametrize(
        "order",
        [
            "A PAR - BUR",
            "F BRE - MAO",
            "A PAR H",
            "A PAR S A BUR",
            "A PAR S A BUR - MAR",
            "F NTH C A LON - BRE",
            "A PAR B",
            "F LON D",
            "A MUN R TYR",
            "F NWG - STP/NC",
            "A LON - BRE VIA",
        ],
    )
    def test_validates_correct_orders(self, order):
        """Valid orders must pass RE_VALID_ORDER."""
        assert RE_VALID_ORDER.match(order), f"Failed to validate: '{order}'"

    @pytest.mark.parametrize(
        "text",
        [
            "AUM: Correct",
            "A good idea",
            "A PAR",  # Incomplete - no action
            "A PAR -",  # Incomplete - no destination
            "A PAR S",  # Incomplete - no support target
        ],
    )
    def test_rejects_invalid_orders(self, text):
        """Invalid text must NOT pass RE_VALID_ORDER."""
        assert not RE_VALID_ORDER.match(text), f"False positive: '{text}'"
