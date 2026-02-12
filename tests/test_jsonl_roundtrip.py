import json


def test_jsonl_roundtrip_preserves_inner_quotes() -> None:
    """JSONL must escape quotes on disk, but decoding must restore the original text.

    This guards against accidental double-escaping when writing training JSONL.
    """

    original = 'rx_id = "RX-7328466"\npatient_mrn = "MRN-740510"'
    row = {"query": "Produce a TOML document for a prescription.", "output": original, "output_type": "TOML"}

    # What is written to JSONL (a single JSON line)
    line = json.dumps(row, ensure_ascii=False)
    assert '\\"RX-7328466\\"' in line  # escaped *in the file representation*

    # What training/eval code reads
    decoded = json.loads(line)
    assert decoded["output"] == original
    assert '\\"' not in decoded["output"]  # no stray backslashes inside the content
