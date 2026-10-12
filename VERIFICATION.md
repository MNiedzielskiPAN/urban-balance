# Clean-repository verification

Date: 2026-09-24  
Python: CPython 3.12.14, 64-bit  
Dependencies: NumPy 2.5.2; pandas 3.0.1; SciPy 1.18.1; pyshp 3.1.6

The complete analysis was run from this directory using only the inputs stored
under `data/`. Both MRC allocations were reconstructed without reading the
supplied caches:

```powershell
python code/python/run_fractional_975118_8am_analysis.py --recompute-mrc
```

The uncached run completed successfully in 73.2 seconds. After converting the
input manifest to portable repository-relative paths, the analysis was run once
more and completed in 55.1 seconds. The final outputs were checked with:

```powershell
python code/python/validate_fractional_975118_8am_analysis.py
python code/python/prepare_fractional_975118_revision.py
python code/python/validate_fractional_975118_manuscript_tables.py
```

Both validators passed. All eight manuscript table fragments were reconstructed
from the machine-readable results and matched the fragments compiled by the
manuscript byte for byte. Principal model results, cross-validation results,
fold-level results, MRC accounting, rounded zonal totals, and all eight tables
also matched the corresponding authoritative working-repository files.

The repository contains no local absolute filesystem paths, no superseded
manuscript or analysis variants, and no file larger than 100 MB.
