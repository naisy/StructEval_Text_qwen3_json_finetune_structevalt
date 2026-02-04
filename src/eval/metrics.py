from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalCounts:
    """Simple counter container for eval metrics.

    Canonical field is `total`. We also keep a few aliases for backward
    compatibility across template revisions (`n`, `constraints_ok`).
    """
    total: int = 0
    json_parse_ok: int = 0
    json_only_ok: int = 0
    schema_ok: int = 0

    # Strict parsing of the requested output format (JSON/YAML/TOML/XML/CSV)
    format_parse_ok: int = 0
    # "Only" heuristics for the requested format (mostly same as strict parse; JSON also checks output-only)
    format_only_ok: int = 0

    # Example constraint counter(s)
    authors_len_ok: int = 0

    # ---- Backward-compatible aliases ----
    @property
    def n(self) -> int:
        return self.total

    @n.setter
    def n(self, v: int) -> None:
        self.total = v

    @property
    def constraints_ok(self) -> int:
        # Historically we used a generic constraint counter; map it to a concrete one.
        return self.authors_len_ok

    @constraints_ok.setter
    def constraints_ok(self, v: int) -> None:
        self.authors_len_ok = v


def safe_div(a: int, b: int) -> float:
    return float(a) / float(b) if b else 0.0
