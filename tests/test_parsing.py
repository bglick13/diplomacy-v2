from src.utils.parsing import extract_metadata, extract_orders

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


def test_extract_metadata_full():
    meta = extract_metadata(CLEAN_OUTPUT)
    assert meta["thought"] == "Germany is weak. I should attack."
    assert meta["truth_status"] == "LIE"


def test_extract_metadata_partial():
    """Test when only some tags are present."""
    meta = extract_metadata(MESSY_OUTPUT)
    assert meta["thought"] == "Thinking..."
    assert "truth_status" not in meta
