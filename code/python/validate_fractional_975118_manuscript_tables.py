"""Validate all table fragments actually compiled by the manuscript.

Every expected fragment is reconstructed from the authoritative
machine-readable results in a temporary directory. Both stored copies must
then equal that reconstruction byte for byte.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import prepare_fractional_975118_revision as prepare


ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "data").is_dir() and (parent / "manuscript").is_dir())
OUT = ROOT / "output" / "fractional_975118_8am"
OUTPUT_TABLES = OUT / "tables"
MANUSCRIPT_TABLES = ROOT / "manuscript" / "generated_tables" / "fractional_975118_8am"
MANUSCRIPT = ROOT / "manuscript" / "jhb_comparison.tex"

TABLES = (
    "manuscript_main_30.tex",
    "manuscript_coefficients_30.tex",
    "manuscript_thresholds.tex",
    "manuscript_prediction_winners_30.tex",
    "manuscript_diagnostics_30.tex",
    "manuscript_one_variable_30.tex",
    "manuscript_ratio_restrictions_30.tex",
    "manuscript_prediction_all_30.tex",
)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="fractional_975118_tables_") as temporary:
        expected = Path(temporary) / "expected"
        expected_mirror = Path(temporary) / "mirror"
        expected.mkdir()
        expected_mirror.mkdir()

        original_t = prepare.T
        original_manuscript_tables = prepare.MANUSCRIPT_TABLES
        try:
            prepare.T = expected
            prepare.MANUSCRIPT_TABLES = expected_mirror
            prepare.tables()
        finally:
            prepare.T = original_t
            prepare.MANUSCRIPT_TABLES = original_manuscript_tables

        for name in TABLES:
            expected_bytes = (expected / name).read_bytes()
            output_bytes = (OUTPUT_TABLES / name).read_bytes()
            compiled_bytes = (MANUSCRIPT_TABLES / name).read_bytes()
            if output_bytes != expected_bytes:
                raise AssertionError(f"Analysis-output table is stale or inconsistent: {name}")
            if compiled_bytes != expected_bytes:
                raise AssertionError(f"Compiled manuscript table is stale or inconsistent: {name}")

    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    for name in TABLES:
        if name not in manuscript:
            raise AssertionError(f"Compiled manuscript does not input table fragment: {name}")

    print(
        "Validation passed: all eight compiled fragments exactly match tables "
        "reconstructed from the authoritative machine-readable results."
    )


if __name__ == "__main__":
    main()
