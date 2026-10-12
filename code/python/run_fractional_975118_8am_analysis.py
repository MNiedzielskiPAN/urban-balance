"""Fractional full-period OD flows and controlled-rounded margins at 8am.

This is an isolated revision pipeline. It does not modify raw inputs or any
previous output. CSV files use semicolon separators and decimal points.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "data").is_dir() and (parent / "manuscript").is_dir())
LOCAL_DEPS = ROOT / ".python_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
import pandas as pd
from scipy.stats import chi2

import run_robustness as base


SELF_CONTAINED_INPUTS = ROOT / "data" / "input"
LEGACY = SELF_CONTAINED_INPUTS if SELF_CONTAINED_INPUTS.exists() else ROOT.parent
OUT = ROOT / "output" / "fractional_975118_8am"
MACHINE = OUT / "machine_readable"
TABLES = OUT / "tables"
CACHE = OUT / "cache"
for directory in (OUT, MACHINE, TABLES, CACHE):
    directory.mkdir(parents=True, exist_ok=True)

N = 895
THRESHOLDS = (20, 30, 45, 60)
MODES = ("car", "pt")
OUTCOMES = ("pairwise", "out_commute", "in_commute")
MODE_LABELS = {"car": "Car", "pt": "Public transport"}
OUTCOME_LABELS = {
    "pairwise": "Pairwise network time",
    "out_commute": "Out-commute",
    "in_commute": "In-commute",
}
MODEL_ORDER = ("mrc", "jwr", "separate_access", "access_ratio", "2sfca")
RANDOM_SEED = 20260826

FLOW_FILE = LEGACY / "macierze_obs_flow" / "Macierze_modes_8am.csv"
TRANSLATION_FILE = LEGACY / "waw_zoneID_translation.csv"
GEOMETRY_FILE = LEGACY / "gis" / "warszawa_zones_centroids_weighted.shp"
GEOMETRY_COMPONENTS = tuple(GEOMETRY_FILE.with_suffix(extension)
                            for extension in (".shp", ".shx", ".dbf", ".prj"))
TRAVEL_FILES = {
    "car": ROOT / "data" / "TomTom_S60_0511_v4_adjTT_newzoneID.dta",
    "pt": ROOT / "data" / "Waw_GTFS_H5_0511_v2_newzoneID.dta",
}


def csv_write(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, sep=";", decimal=".", index=False, float_format="%.12g")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_total_flow() -> tuple[np.ndarray, dict]:
    raw = pd.read_csv(FLOW_FILE, sep=";", decimal=",", usecols=["From", "To", "No. 101"])
    raw[["From", "To"]] = raw[["From", "To"]].astype(int)
    raw = raw[(raw.From != 797) & (raw.To != 797)].copy()
    translation = pd.read_csv(TRANSLATION_FILE, sep=";", usecols=["zoneID1", "zoneID2"])
    mapping = dict(zip(translation.zoneID2.astype(int), translation.zoneID1.astype(int)))
    raw["fromid"] = raw.From.map(mapping)
    raw["toid"] = raw.To.map(mapping)
    if raw[["fromid", "toid"]].isna().any().any():
        raise ValueError("Unmapped legacy zone ID in total OD flow")
    raw[["fromid", "toid"]] = raw[["fromid", "toid"]].astype(int)
    if len(raw) != N * N or raw.duplicated(["fromid", "toid"]).any():
        raise ValueError("Mapped total OD flow is not a complete unique 895 x 895 matrix")
    raw["flow"] = pd.to_numeric(raw["No. 101"], errors="raise")
    if (raw.flow < 0).any() or not np.isfinite(raw.flow).all():
        raise ValueError("Invalid total OD flow")
    matrix = raw.sort_values(["fromid", "toid"]).flow.to_numpy(float).reshape(N, N)
    details = {
        "source_column": "No. 101",
        "source_interpretation": "full-period fractional commuting OD pattern",
        "removed_legacy_zone": 797,
        "mapped_pairs": int(matrix.size),
        "positive_pairs": int((matrix > 0).sum()),
        "raw_mapped_total": float(matrix.sum()),
    }
    return matrix, details


def largest_remainder(values: np.ndarray, target: int) -> np.ndarray:
    """Controlled rounding with deterministic zone-ID tie breaking."""
    floors = np.floor(values).astype(np.int64)
    seats = int(target - floors.sum())
    if seats < 0 or seats > len(values):
        raise ValueError("Largest-remainder target is incompatible with the margins")
    remainders = values - floors
    order = np.lexsort((np.arange(len(values)), -remainders))
    rounded = floors.copy()
    rounded[order[:seats]] += 1
    if rounded.sum() != target or not np.all((rounded == floors) | (rounded == floors + 1)):
        raise AssertionError("Controlled rounding failed")
    return rounded


def derive_integer_margins(flow: np.ndarray) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    """Aggregate fractional cells first, then round both margins to one total."""
    fractional_workers = flow.sum(axis=1)
    fractional_jobs = flow.sum(axis=0)
    if not np.isclose(fractional_workers.sum(), fractional_jobs.sum()):
        raise ValueError("Origin and destination fractional totals differ")
    target = int(np.rint(flow.sum()))
    workers = largest_remainder(fractional_workers, target)
    jobs = largest_remainder(fractional_jobs, target)
    frame = pd.DataFrame({
        "newzoneid": np.arange(1, N + 1),
        "fractional_workers": fractional_workers,
        "workers": workers,
        "worker_rounding_change": workers - fractional_workers,
        "fractional_jobs": fractional_jobs,
        "jobs": jobs,
        "job_rounding_change": jobs - fractional_jobs,
    })
    details = {
        "method": "aggregate fractional OD cells to zonal margins, then deterministic largest-remainder rounding",
        "fractional_total": float(flow.sum()),
        "integer_target": target,
        "integer_worker_total": int(workers.sum()),
        "integer_job_total": int(jobs.sum()),
        "maximum_absolute_worker_rounding_change": float(np.max(np.abs(workers - fractional_workers))),
        "maximum_absolute_job_rounding_change": float(np.max(np.abs(jobs - fractional_jobs))),
        "od_cells_modified": 0,
    }
    return jobs.astype(float), workers.astype(float), frame, details


def weighted_outcomes(times: np.ndarray, flow: np.ndarray) -> dict[str, np.ndarray]:
    row = flow.sum(axis=1)
    col = flow.sum(axis=0)
    commuter_minutes = flow * times
    out = np.divide(commuter_minutes.sum(axis=1), row,
                    out=np.full(N, np.nan), where=row > 0)
    inc = np.divide(commuter_minutes.sum(axis=0), col,
                    out=np.full(N, np.nan), where=col > 0)
    return {
        "pairwise": base.safe_log(times.ravel()),
        "out_commute": base.safe_log(out),
        "in_commute": base.safe_log(inc),
        "out_minutes": out,
        "in_minutes": inc,
    }


def mrc(times: np.ndarray, jobs: np.ndarray, workers: np.ndarray, mode: str,
        recompute: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
    cache = CACHE / f"mrc_{mode}_8am_full_period.npz"
    if cache.exists() and not recompute:
        saved = np.load(cache)
        allocation = saved["allocation"]
        objective = float(saved["objective"])
    else:
        rhs = np.concatenate([workers, jobs])[:-1]
        result = base.linprog(times.ravel(), A_eq=base.transport_constraints(), b_eq=rhs,
                              bounds=(0, None), method="highs")
        if not result.success:
            raise RuntimeError(f"MRC failed for {mode}: {result.message}")
        allocation = result.x.reshape(N, N)
        objective = float(result.fun)
        np.savez_compressed(cache, allocation=allocation, objective=objective)
    costs = allocation * times
    origin = np.divide(costs.sum(axis=1), workers, out=np.full(N, np.nan), where=workers > 0)
    destination = np.divide(costs.sum(axis=0), jobs, out=np.full(N, np.nan), where=jobs > 0)
    details = {
        "mode": mode,
        "objective_commuter_minutes": objective,
        "assigned_commuters": float(allocation.sum()),
        "max_worker_margin_error": float(np.max(np.abs(allocation.sum(axis=1) - workers))),
        "max_job_margin_error": float(np.max(np.abs(allocation.sum(axis=0) - jobs))),
        "origin_quantity": "mean assigned time by origin under the metropolitan minimum-cost assignment",
        "destination_quantity": "mean assigned time by destination under the metropolitan minimum-cost assignment",
    }
    return origin, destination, details


def metrics(times: np.ndarray, jobs: np.ndarray, workers: np.ndarray,
            threshold: int) -> tuple[dict[str, dict[str, np.ndarray]], list[dict]]:
    """Calculate directional accessibility and symmetric directional 2SFCA.

    Origin 2SFCA: R_j=E_j/sum_i(w_ij W_i); A_i=sum_j(w_ij R_j).
    Destination 2SFCA: Q_i=W_i/sum_j(w_ij E_j); B_j=sum_i(w_ij Q_i).
    The destination construction therefore reverses aggregation direction and
    exchanges the jobs/workers roles. Ratios are never logged before summing.
    """
    w = (times <= threshold).astype(float)
    origin_jobs = w @ jobs
    origin_workers = w @ workers
    destination_jobs = w.T @ jobs
    destination_workers = w.T @ workers

    origin_competition = w.T @ workers
    origin_ratio = np.divide(jobs, origin_competition, out=np.zeros_like(jobs),
                             where=origin_competition > 0)
    origin_2sfca = w @ origin_ratio

    destination_competition = w @ jobs
    destination_ratio = np.divide(workers, destination_competition,
                                  out=np.zeros_like(workers), where=destination_competition > 0)
    destination_2sfca = w.T @ destination_ratio

    output = {
        "origin": {
            "access_jobs": origin_jobs,
            "access_workers": origin_workers,
            "access_ratio": np.divide(origin_jobs, origin_workers,
                                      out=np.full(N, np.nan), where=origin_workers > 0),
            "2sfca": origin_2sfca,
        },
        "destination": {
            "access_jobs": destination_jobs,
            "access_workers": destination_workers,
            "access_ratio": np.divide(destination_jobs, destination_workers,
                                      out=np.full(N, np.nan), where=destination_workers > 0),
            "2sfca": destination_2sfca,
        },
    }
    accounting = [
        {
            "threshold_minutes": threshold,
            "direction": "origin",
            "ratio_numerator": "jobs at destination j",
            "competition_denominator": "sum of workers i able to reach destination j in forward direction",
            "aggregation": "sum unlogged destination ratios over destinations reachable from origin i",
            "zero_denominator_treatment": "ratio set to zero; completed index logged only after summation",
            "zero_competition_denominators": int((origin_competition == 0).sum()),
            "zero_completed_indices": int((origin_2sfca == 0).sum()),
            "reverses_direction": False,
            "exchanges_jobs_workers": False,
        },
        {
            "threshold_minutes": threshold,
            "direction": "destination",
            "ratio_numerator": "workers at origin i",
            "competition_denominator": "sum of jobs j reachable from origin i in forward direction",
            "aggregation": "sum unlogged origin ratios over origins able to reach destination j",
            "zero_denominator_treatment": "ratio set to zero; completed index logged only after summation",
            "zero_competition_denominators": int((destination_competition == 0).sum()),
            "zero_completed_indices": int((destination_2sfca == 0).sum()),
            "reverses_direction": True,
            "exchanges_jobs_workers": True,
        },
    ]
    return output, accounting


def predictor_sets(mrc_o: np.ndarray, mrc_d: np.ndarray, jwr: np.ndarray,
                   access: dict[str, dict[str, np.ndarray]], outcome: str) -> dict[str, np.ndarray]:
    if outcome == "pairwise":
        return {
            "mrc": np.column_stack([base.safe_log(mrc_o).repeat(N), np.tile(base.safe_log(mrc_d), N)]),
            "jwr": np.column_stack([base.safe_log(jwr).repeat(N), np.tile(base.safe_log(jwr), N)]),
            "separate_access": np.column_stack([
                base.safe_log(access["origin"]["access_jobs"]).repeat(N),
                base.safe_log(access["origin"]["access_workers"]).repeat(N),
                np.tile(base.safe_log(access["destination"]["access_jobs"]), N),
                np.tile(base.safe_log(access["destination"]["access_workers"]), N),
            ]),
            "access_ratio": np.column_stack([
                base.safe_log(access["origin"]["access_ratio"]).repeat(N),
                np.tile(base.safe_log(access["destination"]["access_ratio"]), N),
            ]),
            "2sfca": np.column_stack([
                base.safe_log(access["origin"]["2sfca"]).repeat(N),
                np.tile(base.safe_log(access["destination"]["2sfca"]), N),
            ]),
        }
    direction = "origin" if outcome == "out_commute" else "destination"
    mrc_value = mrc_o if outcome == "out_commute" else mrc_d
    return {
        "mrc": base.safe_log(mrc_value)[:, None],
        "jwr": base.safe_log(jwr)[:, None],
        "separate_access": np.column_stack([
            base.safe_log(access[direction]["access_jobs"]),
            base.safe_log(access[direction]["access_workers"]),
        ]),
        "access_ratio": base.safe_log(access[direction]["access_ratio"])[:, None],
        "2sfca": base.safe_log(access[direction]["2sfca"])[:, None],
    }


def common_valid(y: np.ndarray, predictors: dict[str, np.ndarray]) -> np.ndarray:
    valid = np.isfinite(y)
    for x in predictors.values():
        valid &= np.isfinite(x).all(axis=1)
    return valid


def spatial_fit(y: np.ndarray, x: np.ndarray, valid: np.ndarray, outcome: str,
                coords: np.ndarray) -> tuple[base.Fit, str, float, float]:
    if outcome == "pairwise":
        origins = np.repeat(np.arange(N), N)[valid]
        destinations = np.tile(np.arange(N), N)[valid]
        fit = base.fit_linear(y[valid], x[valid], clusters=(origins, destinations))
        return fit, "two-way origin/destination clustered", np.nan, np.nan
    fit = base.fit_linear(y[valid], x[valid])
    moran, moran_p = base.moran_i(fit.residual, coords[valid])
    covariance = base.conley_covariance(y[valid], x[valid], fit, coords[valid])
    fit.covariance = covariance
    fit.se = np.sqrt(np.maximum(np.diag(covariance), 0))
    return fit, "Conley spatial HAC (10 km Bartlett kernel)", moran, moran_p


def restriction_test(fit: base.Fit, outcome: str) -> dict:
    """Wald test of the equal-and-opposite restriction imposed by access ratio."""
    if outcome == "pairwise":
        # beta: intercept, origin jobs, origin workers, destination jobs, destination workers
        restriction = np.array([[0, 1, 1, 0, 0], [0, 0, 0, 1, 1]], dtype=float)
    else:
        restriction = np.array([[0, 1, 1]], dtype=float)
    difference = restriction @ fit.beta
    variance = restriction @ fit.covariance @ restriction.T
    statistic = float(difference.T @ np.linalg.pinv(variance) @ difference)
    df = int(restriction.shape[0])
    return {
        "restriction": "jobs coefficient + workers coefficient = 0 at each represented endpoint",
        "wald_chi2": statistic,
        "df": df,
        "p_value": float(chi2.sf(statistic, df)),
        "restriction_rejected_5pct": bool(chi2.sf(statistic, df) < 0.05),
    }


def cross_validate(y: np.ndarray, x: np.ndarray, valid: np.ndarray, outcome: str,
                   folds: np.ndarray) -> tuple[dict, list[dict]]:
    """Five geographic blocks; metrics are pooled from fold-level sums.

    Pairwise tests contain only within-held-out-block connections. Training
    contains only connections whose two endpoints are outside that block.
    Cross-block connections are unused in that fold.
    """
    origins = np.repeat(np.arange(N), N) if outcome == "pairwise" else None
    destinations = np.tile(np.arange(N), N) if outcome == "pairwise" else None
    rows = []
    total_sse = total_sae = total_sst_training = total_sst_heldout = 0.0
    total_n = 0
    for fold in sorted(np.unique(folds)):
        if outcome == "pairwise":
            test = valid & (folds[origins] == fold) & (folds[destinations] == fold)
            train = valid & (folds[origins] != fold) & (folds[destinations] != fold)
            scope = "within-held-out-block pairs; cross-block pairs unused"
        else:
            test = valid & (folds == fold)
            train = valid & (folds != fold)
            scope = "zones in held-out geographic block"
        fit = base.fit_linear(y[train], x[train])
        pred = np.column_stack([np.ones(test.sum()), x[test]]) @ fit.beta
        error = y[test] - pred
        training_mean = float(y[train].mean())
        heldout_mean = float(y[test].mean())
        sse = float(np.sum(error ** 2))
        sae = float(np.sum(np.abs(error)))
        sst_training = float(np.sum((y[test] - training_mean) ** 2))
        sst_heldout = float(np.sum((y[test] - heldout_mean) ** 2))
        rows.append({
            "fold": int(fold), "train_N": int(train.sum()), "test_N": int(test.sum()),
            "test_scope": scope, "training_mean_log_outcome": training_mean,
            "heldout_mean_log_outcome": heldout_mean, "sse": sse, "sae": sae,
            "sst_training_mean_benchmark": sst_training,
            "sst_heldout_mean_benchmark": sst_heldout,
            "rmse_log_units": float(np.sqrt(sse / test.sum())),
            "mae_log_units": float(sae / test.sum()),
            "oos_r2_training_mean": float(1 - sse / sst_training) if sst_training > 0 else np.nan,
            "oos_r2_heldout_mean": float(1 - sse / sst_heldout) if sst_heldout > 0 else np.nan,
        })
        total_sse += sse
        total_sae += sae
        total_sst_training += sst_training
        total_sst_heldout += sst_heldout
        total_n += int(test.sum())
    aggregate = {
        "cv_test_N": total_n,
        "rmse_log_units": float(np.sqrt(total_sse / total_n)),
        "mae_log_units": float(total_sae / total_n),
        "oos_r2_training_mean": float(1 - total_sse / total_sst_training) if total_sst_training > 0 else np.nan,
        "oos_r2_heldout_mean": float(1 - total_sse / total_sst_heldout) if total_sst_heldout > 0 else np.nan,
        "aggregation": "pooled held-out SSE/SAE; both R2 variants use pooled SST around the fold training mean or fold held-out mean",
    }
    return aggregate, rows


def latex(frame: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    shown = frame.copy()
    for col in shown.columns:
        shown[col] = shown[col].map(lambda v: "" if pd.isna(v) else
                                    (f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)))
    lines = ["\\begin{table}[htbp]", "\\centering", "\\scriptsize",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\begin{tabular}{" + "l" * len(shown.columns) + "}", "\\toprule",
             " & ".join(c.replace("_", "\\_") for c in shown.columns) + " \\\\", "\\midrule"]
    for row in shown.itertuples(index=False, name=None):
        lines.append(" & ".join(str(v).replace("_", "\\_") for v in row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(recompute_mrc: bool = False) -> None:
    started = time.time()
    raw_flow, raw_flow_info = load_total_flow()
    flow = raw_flow
    jobs, workers, totals, rounding_info = derive_integer_margins(flow)
    coords = base.coordinates()
    folds = base.kmeans_folds(coords, folds=5, seed=RANDOM_SEED)
    csv_write(pd.DataFrame({
        "zoneid": np.arange(1, N + 1), "x_projected_m": coords[:, 0],
        "y_projected_m": coords[:, 1], "geographic_block": folds,
    }), MACHINE / "spatial_blocks.csv")

    input_rows = []
    required_inputs = {
        "full_period_fractional_od": FLOW_FILE,
        "legacy_to_895_zone_translation": TRANSLATION_FILE,
        "car_8am_travel_times": TRAVEL_FILES["car"],
        "public_transport_8am_travel_times": TRAVEL_FILES["pt"],
    }
    for component in GEOMETRY_COMPONENTS:
        required_inputs[f"spatial_centroid_geometry_{component.suffix[1:]}"] = component
    for role, path in required_inputs.items():
        resolved = path.resolve()
        try:
            displayed_path = resolved.relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            displayed_path = str(resolved)
        input_rows.append({"role": role, "path": displayed_path, "bytes": path.stat().st_size,
                           "sha256": sha256(path)})
    csv_write(pd.DataFrame(input_rows), MACHINE / "input_manifest.csv")

    results, accounting, restrictions = [], [], []
    cv_results, cv_folds = [], []
    onevar, residuals, correlations, vifs = [], [], [], []
    metric_rows, sfca_rows, mrc_rows, outcome_rows = [], [], [], []

    for mode in MODES:
        print(f"Running {mode} fractional OD with 975118 rounded margins and 8am times", flush=True)
        times = base.canonical_times(mode, 8)
        outcomes = weighted_outcomes(times, flow)
        mrc_o, mrc_d, mrc_info = mrc(times, jobs, workers, mode, recompute=recompute_mrc)
        mrc_rows.append(mrc_info)
        jwr = np.divide(jobs, workers, out=np.full(N, np.nan), where=workers > 0)
        outcome_rows.extend([
            {"mode": mode, "outcome": "out_commute", "represented_commuters": flow.sum(),
             "positive_denominator_zones": int((flow.sum(axis=1) > 0).sum()),
             "zero_denominator_zones": int((flow.sum(axis=1) == 0).sum()),
             "minimum_minutes": float(np.nanmin(outcomes["out_minutes"])),
             "maximum_minutes": float(np.nanmax(outcomes["out_minutes"]))},
            {"mode": mode, "outcome": "in_commute", "represented_commuters": flow.sum(),
             "positive_denominator_zones": int((flow.sum(axis=0) > 0).sum()),
             "zero_denominator_zones": int((flow.sum(axis=0) == 0).sum()),
             "minimum_minutes": float(np.nanmin(outcomes["in_minutes"])),
             "maximum_minutes": float(np.nanmax(outcomes["in_minutes"]))},
        ])

        by_threshold = {}
        valid_by_threshold = {}
        for threshold in THRESHOLDS:
            access, sfca_info = metrics(times, jobs, workers, threshold)
            for row in sfca_info:
                sfca_rows.append({"mode": mode, **row})
            by_threshold[threshold] = {}
            valid_by_threshold[threshold] = {}
            for direction in ("origin", "destination"):
                for name, values in access[direction].items():
                    for zone, value in enumerate(values, start=1):
                        metric_rows.append({"mode": mode, "threshold_minutes": threshold,
                                            "direction": direction, "zoneid": zone,
                                            "metric": name, "value_before_log": value})
            for outcome in OUTCOMES:
                predictors = predictor_sets(mrc_o, mrc_d, jwr, access, outcome)
                by_threshold[threshold][outcome] = predictors
                valid_by_threshold[threshold][outcome] = common_valid(outcomes[outcome], predictors)

        all_threshold_valid = {
            outcome: np.logical_and.reduce([valid_by_threshold[t][outcome] for t in THRESHOLDS])
            for outcome in OUTCOMES
        }

        for threshold in THRESHOLDS:
            for outcome in OUTCOMES:
                sample_variants = {"threshold_common": valid_by_threshold[threshold][outcome]}
                if mode == "pt":
                    sample_variants["all_threshold_intersection"] = all_threshold_valid[outcome]
                for sample_variant, valid in sample_variants.items():
                    y = outcomes[outcome]
                    predictors = by_threshold[threshold][outcome]
                    for model in MODEL_ORDER:
                        x = predictors[model]
                        fit, inference, moran, moran_p = spatial_fit(y, x, valid, outcome, coords)
                        row = {
                            "mode": mode, "threshold_minutes": threshold, "outcome": outcome,
                            "sample_variant": sample_variant, "model": model, "N": fit.n,
                            "represented_fractional_flow": float(flow.sum()), "r2": fit.r2, "adj_r2": fit.adj_r2,
                            "rmse_in_sample_log_units": fit.rmse,
                            "mae_in_sample_log_units": fit.mae, "inference": inference,
                        }
                        for index, (coef, se) in enumerate(zip(fit.beta, fit.se)):
                            row[f"beta_{index}"] = coef
                            row[f"se_{index}"] = se
                        results.append(row)
                        residuals.append({
                            "mode": mode, "threshold_minutes": threshold, "outcome": outcome,
                            "sample_variant": sample_variant, "model": model, "N": fit.n,
                            "moran_i": moran, "moran_permutation_p": moran_p,
                            **base.residual_summary(fit),
                        })
                        model_valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
                        accounting.append({
                            "mode": mode, "threshold_minutes": threshold, "outcome": outcome,
                            "sample_variant": sample_variant, "model": model,
                            "starting_observations": len(y),
                            "dropped_invalid_outcome": int((~np.isfinite(y)).sum()),
                            "dropped_invalid_model_predictors": int((np.isfinite(y) & ~np.isfinite(x).all(axis=1)).sum()),
                            "dropped_for_five_model_common_sample": int(model_valid.sum() - valid_by_threshold[threshold][outcome].sum()),
                            "additional_dropped_for_all_threshold_intersection": int(valid_by_threshold[threshold][outcome].sum() - valid.sum()),
                            "final_N": int(valid.sum()),
                        })
                        cv, fold_rows = cross_validate(y, x, valid, outcome, folds)
                        cv_results.append({"mode": mode, "threshold_minutes": threshold,
                                           "outcome": outcome, "sample_variant": sample_variant,
                                           "model": model, **cv})
                        for fold_row in fold_rows:
                            cv_folds.append({"mode": mode, "threshold_minutes": threshold,
                                             "outcome": outcome, "sample_variant": sample_variant,
                                             "model": model, **fold_row})
                        if model == "separate_access":
                            restrictions.append({"mode": mode, "threshold_minutes": threshold,
                                                 "outcome": outcome, "sample_variant": sample_variant,
                                                 "N": fit.n, "covariance_used": inference,
                                                 **restriction_test(fit, outcome)})

                # Diagnostics and one-variable models use each threshold's five-model common sample.
                valid = valid_by_threshold[threshold][outcome]
                access = by_threshold[threshold][outcome]
                separate = access["separate_access"]
                if outcome == "pairwise":
                    # A one-variable accessibility model retains the same variable
                    # at both endpoints, matching the pairwise model structure.
                    for name, columns in (("access_jobs", [0, 2]), ("access_workers", [1, 3])):
                        x = separate[:, columns]
                        own_valid = np.isfinite(outcomes[outcome]) & np.isfinite(x).all(axis=1)
                        fit, inference, _, _ = spatial_fit(outcomes[outcome], x, own_valid, outcome, coords)
                        onevar.append({
                            "mode": mode, "threshold_minutes": threshold, "outcome": outcome,
                            "variable": name, "N": fit.n,
                            "beta_origin": fit.beta[1], "se_origin": fit.se[1],
                            "beta_destination": fit.beta[2], "se_destination": fit.se[2],
                            "adj_r2": fit.adj_r2, "inference": inference,
                        })
                else:
                    for col, name in enumerate(("access_jobs", "access_workers")):
                        x = separate[:, [col]]
                        own_valid = np.isfinite(outcomes[outcome]) & np.isfinite(x).all(axis=1)
                        fit, inference, _, _ = spatial_fit(outcomes[outcome], x, own_valid, outcome, coords)
                        onevar.append({
                            "mode": mode, "threshold_minutes": threshold, "outcome": outcome,
                            "variable": name, "N": fit.n, "beta": fit.beta[1],
                            "standard_error": fit.se[1], "adj_r2": fit.adj_r2,
                            "inference": inference,
                        })

                if outcome != "pairwise":
                    x_valid = separate[np.isfinite(separate).all(axis=1)]
                    for name, value in zip(("access_jobs", "access_workers"), base.vif_values(x_valid)):
                        vifs.append({"mode": mode, "threshold_minutes": threshold,
                                     "outcome": outcome, "variable": name, "vif": value})
                    direction = "origin" if outcome == "out_commute" else "destination"
                    metric_values = {
                        "mrc": mrc_o if outcome == "out_commute" else mrc_d,
                        "jwr": jwr,
                        "access_jobs": (by_threshold[threshold][outcome]["separate_access"][:, 0]),
                        "access_workers": (by_threshold[threshold][outcome]["separate_access"][:, 1]),
                        "access_ratio": by_threshold[threshold][outcome]["access_ratio"][:, 0],
                        "2sfca": by_threshold[threshold][outcome]["2sfca"][:, 0],
                    }
                    # Predictor arrays above are already logged; MRC/JWR are logged here for consistency.
                    logged = pd.DataFrame({
                        "mrc": base.safe_log(metric_values["mrc"]), "jwr": base.safe_log(metric_values["jwr"]),
                        "access_jobs": metric_values["access_jobs"], "access_workers": metric_values["access_workers"],
                        "access_ratio": metric_values["access_ratio"], "2sfca": metric_values["2sfca"],
                    })
                    corr = logged.corr()
                    for left in corr.columns:
                        for right in corr.columns:
                            correlations.append({"mode": mode, "threshold_minutes": threshold,
                                                 "outcome": outcome, "variable_1": left,
                                                 "variable_2": right, "correlation": corr.loc[left, right]})

    frames = {
        "results": pd.DataFrame(results), "sample_accounting": pd.DataFrame(accounting),
        "restriction_tests": pd.DataFrame(restrictions), "cv_results": pd.DataFrame(cv_results),
        "cv_fold_results": pd.DataFrame(cv_folds), "one_variable_regressions": pd.DataFrame(onevar),
        "residual_diagnostics": pd.DataFrame(residuals), "correlations": pd.DataFrame(correlations),
        "vif": pd.DataFrame(vifs), "2sfca_accounting": pd.DataFrame(sfca_rows),
        "mrc_accounting": pd.DataFrame(mrc_rows), "weighted_outcome_accounting": pd.DataFrame(outcome_rows),
        "zonal_metric_values": pd.DataFrame(metric_rows),
    }
    for name, frame in frames.items():
        csv_write(frame, MACHINE / f"{name}.csv")
    csv_write(pd.DataFrame([{**raw_flow_info, **rounding_info}]), MACHINE / "margin_rounding_accounting.csv")
    csv_write(totals, MACHINE / "fractional_and_integer_zonal_totals.csv")

    result_frame = frames["results"]
    main = result_frame[(result_frame.threshold_minutes == 30) &
                        (result_frame.sample_variant == "threshold_common")]
    latex(main[["mode", "outcome", "model", "N", "adj_r2"]], TABLES / "main_30min_all_models.tex",
          "Thirty-minute common-sample comparison: all five models", "tab:total-main")
    fixed_pt = result_frame[(result_frame["mode"] == "pt") &
                            (result_frame.sample_variant == "all_threshold_intersection")]
    latex(fixed_pt[["threshold_minutes", "outcome", "model", "N", "adj_r2"]],
          TABLES / "public_transport_fixed_all_threshold_sample.tex",
          "Public-transport threshold comparison on the all-threshold intersection", "tab:pt-fixed")
    latex(frames["cv_results"][["mode", "threshold_minutes", "outcome", "sample_variant", "model",
                                "cv_test_N", "rmse_log_units", "mae_log_units",
                                "oos_r2_training_mean", "oos_r2_heldout_mean"]],
          TABLES / "predictive_appendix_all_models_all_thresholds.tex",
          "Spatially blocked prediction for all five models and thresholds", "tab:cv-all")
    latex(frames["one_variable_regressions"], TABLES / "one_variable_regressions.tex",
          "One-variable accessibility regressions", "tab:onevar")
    latex(frames["restriction_tests"], TABLES / "accessibility_ratio_restriction_tests.tex",
          "Tests of the equal-and-opposite accessibility-ratio restriction", "tab:ratio-test")

    metadata = {
        "analysis_variant": "original full-period fractional OD weights and aggregate-then-rounded 975118 margins; 8am travel times",
        "zones": N, "thresholds_minutes": list(THRESHOLDS), "random_seed": RANDOM_SEED,
        "travel_time_columns": {"car": "t830 (the study's 8am car matrix)", "pt": "t8"},
        "diagonal_rule": "replace every diagonal with 0.60 times nearest positive outbound interzonal time",
        "opportunity_universe": "all 895 zones and full-period totals; never recalculated after sample restriction",
        "jobs_total": float(jobs.sum()), "workers_total": float(workers.sum()),
        "flow_and_rounding": {**raw_flow_info, **rounding_info},
        "2sfca_destination": "reverses aggregation direction and exchanges jobs/workers; ratios with zero competition are zero; completed sums are logged",
        "public_transport_threshold_sample": "additional all-threshold intersection reported with predictors always calculated on all 895 zones",
        "rmse_mae_scale": "natural-log dependent-variable units",
        "cv_blocks": "five deterministic k-means clusters of projected population-weighted zone centroids",
        "pairwise_cv_scope": "test only within-held-out-block connections; train only pairs with both endpoints outside block; cross-block pairs unused",
        "oos_r2": "both training-mean and held-out-mean fold benchmarks are exported from identical predictions",
        "csv_dialect": "semicolon separator; decimal point",
    }
    (MACHINE / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    elapsed = time.time() - started
    write_report(frames, raw_flow_info, rounding_info, elapsed)
    print(f"Completed in {elapsed:.1f} seconds", flush=True)


def write_report(frames: dict[str, pd.DataFrame], raw_flow: dict, rounding: dict, elapsed: float) -> None:
    results = frames["results"]
    winners = results.loc[results.groupby(["mode", "threshold_minutes", "outcome", "sample_variant"])["adj_r2"].idxmax(),
                          ["mode", "threshold_minutes", "outcome", "sample_variant", "model", "N", "adj_r2"]]
    cv = frames["cv_results"]
    cv_winners = cv.loc[cv.groupby(["mode", "threshold_minutes", "outcome", "sample_variant"])["rmse_log_units"].idxmin(),
                        ["mode", "threshold_minutes", "outcome", "sample_variant", "model", "cv_test_N",
                         "rmse_log_units", "mae_log_units", "oos_r2_training_mean",
                         "oos_r2_heldout_mean"]]
    def md(frame: pd.DataFrame) -> str:
        shown = frame.copy()
        for column in shown.columns:
            shown[column] = shown[column].map(
                lambda value: "" if pd.isna(value) else
                (f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)))
        header = "| " + " | ".join(shown.columns) + " |"
        rule = "| " + " | ".join(["---"] * len(shown.columns)) + " |"
        rows = ["| " + " | ".join(map(str, row)) + " |"
                for row in shown.itertuples(index=False, name=None)]
        return "\n".join([header, rule, *rows])
    report = f"""# Fractional full-period OD flows and rounded 975,118 margins with 8am travel times

## Inputs and controlled rounding

The full fractional OD pattern is the `No. 101` column in `Macierze_modes_8am.csv`; that column is repeated unchanged in the hourly files. After removal of obsolete legacy zone 797 and translation to the 895-zone identifiers, it totals {raw_flow['raw_mapped_total']:.6f}. Fractional cells are first aggregated to origin and destination margins. Each margin is then rounded by the deterministic largest-remainder method to the common rounded grand total of {rounding['integer_target']:,}. Each integer zonal value is a floor or ceiling of its corresponding fractional aggregate, and no OD cell is altered.

Car and public transport calculations use the same 975,118 integer job and worker margins and the same original fractional commuting pattern. Travel costs come from the mode-specific 8am matrices (`t830` in the car file and `t8` in the public-transport file). Every accessibility measure is calculated on all 895 zones before any estimation-sample restriction.

## Outcomes

Pairwise network time is the unweighted natural logarithm of modeled network time for every valid origin--destination connection; it is not an observed commute outcome. Out-commute and in-commute are natural logarithms of mean mode-specific 8am network time evaluated for the same original fractional commuting pattern. They are not separate averages for observed car and public transport commuters.

## Directional 2SFCA

For origins, $R_j=E_j/\\sum_i w_{{ij}}W_i$ and $A_i=\\sum_j w_{{ij}}R_j$. For destinations, roles are exchanged and aggregation direction is reversed: $Q_i=W_i/\\sum_j w_{{ij}}E_j$ and $B_j=\\sum_i w_{{ij}}Q_i$. A ratio with a zero competition denominator is set to zero. Destination or origin ratios are summed in levels, and only the completed index is logged for regression. Counts of zero denominators and completed indices are in `2sfca_accounting.csv`.

## MRC

MRC is reconstructed by linear programming for each 8am mode matrix using the fixed full-period margins. The zonal origin and destination quantities are mean assigned travel times under the metropolitan minimum-cost assignment. They are not interpreted as each zone's individually shortest travel time. Allocation totals and numerical margin errors are in `mrc_accounting.csv`; compressed allocation matrices are in `cache/`.

## Samples and inference

All five models use a common valid sample within each mode, threshold, and outcome. In addition, every public-transport threshold is estimated on the intersection valid at all four thresholds. Restricting the estimation sample never changes the underlying 895-zone opportunity universe. Pairwise standard errors are clustered two ways by origin and destination. Zonal standard errors use a 10-km Bartlett-kernel Conley covariance; Moran diagnostics are reported separately.

## Model comparison winners

{md(winners)}

Threshold-specific rows are in-sample comparisons. The `all_threshold_intersection` public-transport rows make threshold fit directly comparable on a fixed sample.

The table above is generated directly from the authoritative result objects. Threshold-specific rows are descriptive in-sample comparisons; model rankings do not identify destination competition or any other causal mechanism.

## Accessibility-ratio restriction

The ratio model imposes equal-and-opposite coefficients on logged jobs and workers accessibility. `restriction_tests.csv` reports robust Wald tests of that restriction using the separate-accessibility model. Rejection supports retaining separate scales statistically; it does not identify destination competition or any other mechanism.

## Spatial validation

Five geographic blocks are deterministic k-means clusters of projected population-weighted zone centroids; exact assignments are in `spatial_blocks.csv`. Zonal validation holds out one block. Pairwise validation tests only connections whose origin and destination are both inside the held-out block and trains only on connections whose two endpoints are outside it. Cross-block connections are unused in that fold.

RMSE and MAE are reported in natural-log dependent-variable units. Fold statistics are pooled by summing squared or absolute errors. Two out-of-sample $R^2$ values are calculated from identical predictions: one uses each fold's training-outcome mean and the other its held-out-outcome mean. All five models, thresholds, and fold-level components are exported.

{md(cv_winners)}

The two $R^2$ columns differ only in their reference mean. Changes in those scores therefore are not evidence of changed prediction unless RMSE and MAE also change under otherwise identical folds and observations.

## Files

Machine-readable files use semicolon delimiters and decimal points. `input_manifest.csv` gives exact paths, sizes, roles, and SHA-256 hashes for every required input. `one_variable_regressions.csv` and its LaTeX table report the previously mentioned regressions. `cv_results.csv` and `cv_fold_results.csv` provide aggregate and fold-level validation for every model and threshold.

Elapsed time: {elapsed:.1f} seconds.
"""
    (OUT / "analysis_report.md").write_text(report, encoding="utf-8")
    relative_code_dir = Path(__file__).resolve().parent.relative_to(ROOT).as_posix()
    (OUT / "README.md").write_text(
        "# Fractional OD and controlled-rounded 975,118-margin analysis\n\n"
        "Run from the repository root with the bundled or compatible Python environment:\n\n"
        f"```powershell\npython {relative_code_dir}/run_fractional_975118_8am_analysis.py\n"
        f"python {relative_code_dir}/validate_fractional_975118_8am_analysis.py\n```\n\n"
        "Required inputs and hashes are listed in `machine_readable/input_manifest.csv`. "
        "No raw input or prior result is modified.\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recompute-mrc",
        action="store_true",
        help="Ignore existing MRC allocation caches and reconstruct both allocations.",
    )
    args = parser.parse_args()
    main(recompute_mrc=args.recompute_mrc)
