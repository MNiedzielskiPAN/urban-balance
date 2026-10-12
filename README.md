# Jobs--housing balance measure comparison in Warsaw

This repository contains the complete, current replication materials for the
Warsaw jobs--housing balance measure-comparison paper. It includes the current
general manuscript and a separately framed submission version for
*Environment and Planning B: Urban Analytics and City Science*. It
intentionally excludes superseded manuscript and analysis versions.

## Contents

- `code/python/`: the analysis, table generator, and validators;
- `data/`: the two 8am network travel-time matrices;
- `data/input/`: the fractional full-period OD matrix, zone translation, and
  spatial centroid files;
- `output/fractional_975118_8am/`: machine-readable results,
  publication tables, diagnostics, fold assignments, and MRC allocations.

## Tested environment

CPython 3.12.14 (64-bit) with the exact package versions in
`code/python/requirements.txt`. Create a virtual environment and
install those requirements before running the analysis.

## Reproduce the analysis

Run from this repository's top directory:

```powershell
python code/python/run_fractional_975118_8am_analysis.py --recompute-mrc
python code/python/validate_fractional_975118_8am_analysis.py
python code/python/prepare_fractional_975118_revision.py
python code/python/validate_fractional_975118_manuscript_tables.py
```

Omit `--recompute-mrc` to use the supplied MRC allocation caches. All generated
CSV files use semicolon delimiters and decimal points.


## Integrity and citation

`REPOSITORY_MANIFEST.csv` is a short guide to the essential components;
`FILE_HASHES.csv` records the size and SHA-256 hash of every distributed file.
`VERIFICATION.md` records the final clean run. Once the GitHub repository
has a permanent URL or DOI, cite that repository in the manuscript's data and
code availability statement.

No data or code license has been assigned automatically. Confirm redistribution
rights and add an appropriate `LICENSE` file before making the repository public.
