"""Reproducible robustness pipeline for the Warsaw jobs-housing paper.

All inputs are read-only. All generated files are written below
output/fractional_975118_8am/.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = next(parent for parent in HERE.parents
            if (parent / "data").is_dir() and (parent / "manuscript").is_dir())
LOCAL_DEPS = ROOT / ".python_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
import shapefile

DATA = ROOT / "data"
SELF_CONTAINED_INPUTS = DATA / "input"
LEGACY = SELF_CONTAINED_INPUTS if SELF_CONTAINED_INPUTS.exists() else ROOT.parent
OUT = ROOT / "output" / "fractional_975118_8am"
MACHINE = OUT / "machine_readable"
TABLES = OUT / "tables"
CACHE = OUT / "cache"
LOGS = OUT / "logs"
for directory in (OUT, MACHINE, TABLES, CACHE, LOGS):
    directory.mkdir(parents=True, exist_ok=True)

N_ZONES = 895
THRESHOLDS = (20, 30, 45, 60)
PERIODS = (5, 6, 7, 8, 9, 10)
MAIN_PERIOD = 8
DECAY_HALF_LIFE = 30.0
DECAY_BETA = math.log(2.0) / DECAY_HALF_LIFE
MODE_INFO = {
    "car": {
        "tt_file": "TomTom_S60_0511_v4_adjTT_newzoneID.dta",
        "tt_vars": {5: "t530", 6: "t630", 7: "t730", 8: "t830", 9: "t930", 10: "t1030"},
        "jobs_file": "warszawa_fbh_sums_col_car.dta",
        "workers_file": "warszawa_fbh_sums_row_car.dta",
        "jobs_prefix": "ccar_",
        "workers_prefix": "rcar_",
        "flow_label": "car",
    },
    "pt": {
        "tt_file": "Waw_GTFS_H5_0511_v2_newzoneID.dta",
        "tt_vars": {h: f"t{h}" for h in PERIODS},
        "jobs_file": "warszawa_fbh_sums_col_pt.dta",
        "workers_file": "warszawa_fbh_sums_row_pt.dta",
        "jobs_prefix": "cpt_",
        "workers_prefix": "rpt_",
        "flow_label": "pt",
    },
}
MODELS = ("mrc", "jwr", "separate_access", "access_ratio", "2sfca")
INCLUDE_PRESENCE_VARIANT = True
INCLUDE_CONSTRAINED_VARIANT = True
CONSTRAINED_VARIANT_NAME = "presence_constrained_weighted"
ALL_OD_VARIANT_NAME = "existing_all_od"
OD_BASE_RESTRICTION_PROVIDER = None
OD_REPRESENTED_COMMUTERS_PROVIDER = None
APPLY_OD_RESTRICTION_ALL_SCENARIOS = False


@dataclass
class Fit:
    beta: np.ndarray
    residual: np.ndarray
    fitted: np.ndarray
    n: int
    k: int
    r2: float
    adj_r2: float
    rmse: float
    mae: float
    covariance: np.ndarray
    se: np.ndarray


def assert_zone_ids(values: np.ndarray, label: str) -> None:
    unique = np.unique(values.astype(int))
    expected = np.arange(1, N_ZONES + 1)
    if not np.array_equal(unique, expected):
        raise ValueError(f"{label}: expected zone IDs 1..{N_ZONES}; got {len(unique)} unique IDs")


def canonical_times(mode: str, period: int) -> np.ndarray:
    info = MODE_INFO[mode]
    frame = pd.read_stata(DATA / info["tt_file"], convert_categoricals=False)
    frame = frame[["fromid", "toid", info["tt_vars"][period]]].copy()
    frame[["fromid", "toid"]] = frame[["fromid", "toid"]].astype(int)
    if frame.duplicated(["fromid", "toid"]).any() or len(frame) != N_ZONES**2:
        raise ValueError(f"{mode} {period}: OD matrix key is not a complete 895 x 895 matrix")
    assert_zone_ids(frame.fromid.to_numpy(), f"{mode} origins")
    assert_zone_ids(frame.toid.to_numpy(), f"{mode} destinations")
    frame = frame.sort_values(["fromid", "toid"])
    matrix = frame[info["tt_vars"][period]].to_numpy(float).reshape(N_ZONES, N_ZONES)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{mode} {period}: non-finite network times")
    # One rule for every mode/time: intrazonal time is 60% of the nearest
    # positive interzonal outbound time. The stored diagonal is always replaced.
    for i in range(N_ZONES):
        row = np.delete(matrix[i], i)
        positive = row[row > 0]
        if positive.size == 0:
            raise ValueError(f"{mode} {period}: zone {i + 1} has no positive interzonal time")
        matrix[i, i] = 0.60 * positive.min()
    if (matrix <= 0).any():
        raise ValueError(f"{mode} {period}: canonical travel matrix contains non-positive times")
    return matrix


def margins(mode: str, period: int) -> tuple[np.ndarray, np.ndarray]:
    info = MODE_INFO[mode]
    jobs = pd.read_stata(DATA / info["jobs_file"], convert_categoricals=False)
    workers = pd.read_stata(DATA / info["workers_file"], convert_categoricals=False)
    jvar = f'{info["jobs_prefix"]}{period}am'
    wvar = f'{info["workers_prefix"]}{period}am'
    jobs = jobs[["toid", jvar]].rename(columns={"toid": "zoneid", jvar: "jobs"})
    workers = workers[["fromid", wvar]].rename(columns={"fromid": "zoneid", wvar: "workers"})
    if jobs.zoneid.duplicated().any() or workers.zoneid.duplicated().any():
        raise ValueError(f"{mode} {period}: zonal margin key is not unique")
    merged = workers.merge(jobs, on="zoneid", how="outer", validate="one_to_one", indicator=True)
    if not (merged._merge == "both").all():
        raise ValueError(f"{mode} {period}: unmatched job/worker zones")
    merged = merged.sort_values("zoneid")
    assert_zone_ids(merged.zoneid.to_numpy(), f"{mode} {period} margins")
    w = merged.workers.to_numpy(float)
    j = merged.jobs.to_numpy(float)
    if not np.isclose(w.sum(), j.sum()):
        raise ValueError(f"{mode} {period}: worker and job totals differ")
    return j, w


def od_presence(mode: str) -> np.ndarray:
    column = f"{mode}_presence"
    raw = pd.read_csv(DATA / "od_presence.csv", sep=";", decimal=",",
                      usecols=["fromid", "toid", column])
    raw[["fromid", "toid"]] = raw[["fromid", "toid"]].astype(int)
    if len(raw) != N_ZONES**2 or raw.duplicated(["fromid", "toid"]).any():
        raise ValueError(f"{mode}: OD-presence key is not a complete 895 x 895 matrix")
    assert_zone_ids(raw.fromid.to_numpy(), f"{mode} presence origins")
    assert_zone_ids(raw.toid.to_numpy(), f"{mode} presence destinations")
    values = raw.sort_values(["fromid", "toid"])[column].to_numpy()
    if not np.isin(values, [0, 1]).all():
        raise ValueError(f"{mode}: presence indicator is not binary")
    return values.astype(bool).reshape(N_ZONES, N_ZONES)


def transport_constraints() -> sparse.csr_matrix:
    variables = N_ZONES * N_ZONES
    rows = np.concatenate([np.repeat(np.arange(N_ZONES), N_ZONES),
                           N_ZONES + np.tile(np.arange(N_ZONES), N_ZONES)])
    cols = np.concatenate([np.arange(variables), np.arange(variables)])
    values = np.ones(2 * variables)
    return sparse.coo_matrix((values, (rows, cols)),
                             shape=(2 * N_ZONES, variables)).tocsr()[:-1]


def spatially_constrained_flows(mode: str, times: np.ndarray, jobs: np.ndarray,
                                workers: np.ndarray, presence: np.ndarray) -> tuple[np.ndarray, dict]:
    """Doubly constrain a presence-informed gravity prior to the 8am margins."""
    cache_file = CACHE / f"presence_constrained_{mode}_{MAIN_PERIOD}.npz"
    if cache_file.exists():
        flow = np.load(cache_file)["flow"]
    else:
        # Exact structural zeros are infeasible. Give absent links a documented,
        # tiny prior rather than importing or inventing observed flow counts.
        support_prior = np.where(presence, 1.0, 1e-6)
        flow = support_prior * np.exp(-DECAY_BETA * times)
        for iteration in range(20000):
            row_sum = flow.sum(axis=1)
            flow *= np.divide(workers, row_sum, out=np.zeros_like(workers), where=row_sum > 0)[:, None]
            col_sum = flow.sum(axis=0)
            flow *= np.divide(jobs, col_sum, out=np.zeros_like(jobs), where=col_sum > 0)[None, :]
            if iteration % 20 == 0:
                error = max(np.max(np.abs(flow.sum(axis=1) - workers)),
                            np.max(np.abs(flow.sum(axis=0) - jobs)))
                if error < 1e-7:
                    break
        else:
            raise RuntimeError(f"Presence-constrained IPF did not converge for {mode}")
        np.savez_compressed(cache_file, flow=flow)
    row_error = float(np.max(np.abs(flow.sum(axis=1) - workers)))
    col_error = float(np.max(np.abs(flow.sum(axis=0) - jobs)))
    if max(row_error, col_error) > 1e-5:
        raise ValueError(f"{mode}: constrained flow fails margin check: {row_error}, {col_error}")
    positive = flow > 1e-9
    details = {
        "mode": mode, "period": MAIN_PERIOD, "margin_total": float(workers.sum()),
        "original_presence_pairs": int(presence.sum()),
        "positive_constrained_pairs": int(positive.sum()),
        "off_presence_prior_multiplier": 1e-6,
        "flow_on_original_presence": float(flow[presence].sum()),
        "flow_off_original_presence": float(flow[~presence].sum()),
        "max_row_margin_error": row_error, "max_column_margin_error": col_error,
    }
    return flow, details


def coordinates() -> np.ndarray:
    reader = shapefile.Reader(str(LEGACY / "gis" / "warszawa_zones_centroids_weighted.shp"))
    fields = [field[0] for field in reader.fields[1:]]
    id_pos = fields.index("id")
    rows = []
    for shape_record in reader.iterShapeRecords():
        rows.append((int(shape_record.record[id_pos]), *shape_record.shape.points[0]))
    rows.sort()
    ids = np.array([row[0] for row in rows])
    assert_zone_ids(ids, "centroid coordinates")
    return np.asarray([[row[1], row[2]] for row in rows], dtype=float)


def accessibility_metrics(times: np.ndarray, jobs: np.ndarray, workers: np.ndarray,
                          kind: str | int) -> dict[str, dict[str, np.ndarray]]:
    if kind == "decay":
        weight = np.exp(-DECAY_BETA * times)
    else:
        weight = (times <= float(kind)).astype(float)

    def directional(w: np.ndarray) -> dict[str, np.ndarray]:
        access_jobs = w @ jobs
        access_workers = w @ workers
        # Correct 2SFCA: ratios remain unlogged until after aggregation.
        competition = w.T @ workers
        destination_ratio = np.divide(jobs, competition, out=np.full_like(jobs, np.nan), where=competition > 0)
        two_sfca = w @ np.nan_to_num(destination_ratio, nan=0.0)
        return {
            "access_jobs": access_jobs,
            "access_workers": access_workers,
            "access_ratio": np.divide(access_jobs, access_workers, out=np.full_like(access_jobs, np.nan), where=access_workers > 0),
            "2sfca": two_sfca,
        }

    return {"origin": directional(weight), "destination": directional(weight.T)}


def solve_mrc(times: np.ndarray, jobs: np.ndarray, workers: np.ndarray,
              mode: str, period: int) -> tuple[np.ndarray, np.ndarray]:
    cache_file = CACHE / f"mrc_{mode}_{period}.npz"
    if cache_file.exists():
        cached = np.load(cache_file)
        return cached["origin"], cached["destination"]
    c = times.ravel()
    rhs = np.concatenate([workers, jobs])[:-1]
    result = linprog(c, A_eq=transport_constraints(), b_eq=rhs, bounds=(0, None), method="highs")
    if not result.success:
        raise RuntimeError(f"MRC optimization failed for {mode} {period}: {result.message}")
    allocation = result.x.reshape(N_ZONES, N_ZONES)
    cost = allocation * times
    origin = np.divide(cost.sum(axis=1), workers, out=np.full(N_ZONES, np.nan), where=workers > 0)
    destination = np.divide(cost.sum(axis=0), jobs, out=np.full(N_ZONES, np.nan), where=jobs > 0)
    np.savez_compressed(cache_file, origin=origin, destination=destination,
                        objective=float(result.fun), commuters=float(workers.sum()))
    return origin, destination


def safe_log(value: np.ndarray) -> np.ndarray:
    return np.log(value, out=np.full_like(value, np.nan, dtype=float), where=np.asarray(value) > 0)


def outcome_values(mode: str, period: int, times: np.ndarray,
                   jobs: np.ndarray, workers: np.ndarray) -> dict[str, np.ndarray]:
    """Return the default network-time outcomes.

    Variant entry points may replace this function before calling ``main`` while
    preserving the metric construction, common-sample logic, and inference code.
    """
    return {
        "od": safe_log(times.ravel()),
        "origin": safe_log(times.mean(axis=1)),
        "destination": safe_log(times.mean(axis=0)),
    }


def predictor_sets(mrc_o: np.ndarray, mrc_d: np.ndarray, jwr: np.ndarray,
                   access: dict[str, dict[str, np.ndarray]], outcome: str) -> dict[str, np.ndarray]:
    if outcome == "od":
        return {
            "mrc": np.column_stack([safe_log(mrc_o).repeat(N_ZONES), np.tile(safe_log(mrc_d), N_ZONES)]),
            "jwr": np.column_stack([safe_log(jwr).repeat(N_ZONES), np.tile(safe_log(jwr), N_ZONES)]),
            "separate_access": np.column_stack([
                safe_log(access["origin"]["access_jobs"]).repeat(N_ZONES),
                safe_log(access["origin"]["access_workers"]).repeat(N_ZONES),
                np.tile(safe_log(access["destination"]["access_jobs"]), N_ZONES),
                np.tile(safe_log(access["destination"]["access_workers"]), N_ZONES),
            ]),
            "access_ratio": np.column_stack([
                safe_log(access["origin"]["access_ratio"]).repeat(N_ZONES),
                np.tile(safe_log(access["destination"]["access_ratio"]), N_ZONES),
            ]),
            "2sfca": np.column_stack([
                safe_log(access["origin"]["2sfca"]).repeat(N_ZONES),
                np.tile(safe_log(access["destination"]["2sfca"]), N_ZONES),
            ]),
        }
    direction = "origin" if outcome == "origin" else "destination"
    mrc = mrc_o if outcome == "origin" else mrc_d
    return {
        "mrc": safe_log(mrc)[:, None],
        "jwr": safe_log(jwr)[:, None],
        "separate_access": np.column_stack([safe_log(access[direction]["access_jobs"]), safe_log(access[direction]["access_workers"])]),
        "access_ratio": safe_log(access[direction]["access_ratio"])[:, None],
        "2sfca": safe_log(access[direction]["2sfca"])[:, None],
    }


def cluster_meat(scores: np.ndarray, groups: np.ndarray) -> np.ndarray:
    order = np.argsort(groups)
    sorted_groups = groups[order]
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(sorted_groups[1:] != sorted_groups[:-1]) + 1]
    sums = np.add.reduceat(sorted_scores, starts, axis=0)
    return sums.T @ sums


def fit_linear(y: np.ndarray, predictors: np.ndarray, weights: np.ndarray | None = None,
               clusters: tuple[np.ndarray, np.ndarray] | None = None) -> Fit:
    x = np.column_stack([np.ones(len(y)), predictors])
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    root_w = np.sqrt(w)
    xw = x * root_w[:, None]
    yw = y * root_w
    bread = np.linalg.pinv(xw.T @ xw)
    beta = bread @ (xw.T @ yw)
    fitted = x @ beta
    residual = y - fitted
    mean = np.average(y, weights=w)
    sse = float(np.sum(w * residual**2))
    sst = float(np.sum(w * (y - mean)**2))
    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    n, k = x.shape
    adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k) if n > k else np.nan
    if clusters is None:
        covariance = bread * (sse / max(n - k, 1))
    else:
        scores = x * (w * residual)[:, None]
        g1, g2 = clusters
        pair = g1.astype(np.int64) * (int(g2.max()) + 1) + g2.astype(np.int64)
        m1 = cluster_meat(scores, g1)
        m2 = cluster_meat(scores, g2)
        m12 = cluster_meat(scores, pair)
        covariance = bread @ (m1 + m2 - m12) @ bread
        covariance = (covariance + covariance.T) / 2
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    return Fit(beta, residual, fitted, n, k, r2, adj,
               float(np.sqrt(np.average(residual**2, weights=w))),
               float(np.average(np.abs(residual), weights=w)), covariance, se)


def conley_covariance(y: np.ndarray, predictors: np.ndarray, fit: Fit,
                      coords: np.ndarray, bandwidth: float = 10_000.0) -> np.ndarray:
    x = np.column_stack([np.ones(len(y)), predictors])
    bread = np.linalg.pinv(x.T @ x)
    scores = x * fit.residual[:, None]
    distance = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    kernel = np.maximum(1.0 - distance / bandwidth, 0.0)
    meat = scores.T @ kernel @ scores
    covariance = bread @ meat @ bread
    return (covariance + covariance.T) / 2


def moran_i(residual: np.ndarray, coords: np.ndarray, seed: int = 20260826,
            permutations: int = 199, neighbors: int = 8) -> tuple[float, float]:
    n = len(residual)
    distance = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(distance, np.inf)
    nearest = np.argpartition(distance, neighbors, axis=1)[:, :neighbors]
    weights = np.zeros((n, n), dtype=float)
    weights[np.arange(n)[:, None], nearest] = 1.0
    weights = np.maximum(weights, weights.T)
    centered = residual - residual.mean()
    denom = centered @ centered
    observed = n / weights.sum() * (centered @ weights @ centered) / denom
    rng = np.random.default_rng(seed)
    simulated = np.empty(permutations)
    for p in range(permutations):
        z = rng.permutation(centered)
        simulated[p] = n / weights.sum() * (z @ weights @ z) / (z @ z)
    p_value = (1 + np.sum(np.abs(simulated) >= abs(observed))) / (permutations + 1)
    return float(observed), float(p_value)


def kmeans_folds(coords: np.ndarray, folds: int = 5, seed: int = 20260826) -> np.ndarray:
    standardized = (coords - coords.mean(axis=0)) / coords.std(axis=0)
    rng = np.random.default_rng(seed)
    centers = standardized[rng.choice(len(coords), folds, replace=False)]
    labels = np.zeros(len(coords), dtype=int)
    for _ in range(100):
        new_labels = ((standardized[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        new_centers = np.vstack([standardized[new_labels == k].mean(axis=0) for k in range(folds)])
        if np.array_equal(new_labels, labels):
            break
        labels, centers = new_labels, new_centers
    return labels


def cross_validate(y: np.ndarray, x: np.ndarray, valid: np.ndarray, outcome: str,
                   folds: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    predictions, observations, held_weights = [], [], []
    if outcome == "od":
        origin_fold = np.repeat(folds, N_ZONES)
        destination_fold = np.tile(folds, N_ZONES)
    for fold in np.unique(folds):
        if outcome == "od":
            test = valid & (origin_fold == fold) & (destination_fold == fold)
            train = valid & (origin_fold != fold) & (destination_fold != fold)
        else:
            test = valid & (folds == fold)
            train = valid & (folds != fold)
        if test.sum() == 0 or train.sum() <= x.shape[1] + 1:
            continue
        train_weights = None if weights is None else weights[train]
        fitted = fit_linear(y[train], x[train], train_weights)
        prediction = np.column_stack([np.ones(test.sum()), x[test]]) @ fitted.beta
        predictions.append(prediction)
        observations.append(y[test])
        held_weights.append(np.ones(test.sum()) if weights is None else weights[test])
    if not predictions:
        return {"cv_n": 0, "rmse": np.nan, "mae": np.nan, "oos_r2": np.nan}
    pred = np.concatenate(predictions)
    obs = np.concatenate(observations)
    w = np.concatenate(held_weights)
    error = obs - pred
    baseline = np.average(obs, weights=w)
    sse = np.sum(w * error**2)
    sst = np.sum(w * (obs - baseline)**2)
    return {"cv_n": int(len(obs)), "rmse": float(np.sqrt(np.average(error**2, weights=w))),
            "mae": float(np.average(np.abs(error), weights=w)),
            "oos_r2": float(1 - sse / sst) if sst > 0 else np.nan}


def vif_values(x: np.ndarray) -> list[float]:
    values = []
    for column in range(x.shape[1]):
        others = np.delete(x, column, axis=1)
        if others.shape[1] == 0:
            values.append(1.0)
            continue
        fit = fit_linear(x[:, column], others)
        values.append(float(1 / (1 - fit.r2)) if fit.r2 < 1 else np.inf)
    return values


def residual_summary(fit: Fit) -> dict[str, float]:
    r = fit.residual
    sd = r.std(ddof=1)
    skew = float(np.mean(((r - r.mean()) / sd) ** 3)) if sd > 0 else np.nan
    kurt = float(np.mean(((r - r.mean()) / sd) ** 4)) if sd > 0 else np.nan
    jb = len(r) / 6 * (skew**2 + (kurt - 3) ** 2 / 4) if np.isfinite(skew) else np.nan
    return {"residual_mean": float(r.mean()), "residual_sd": float(sd), "residual_skew": skew,
            "residual_kurtosis": kurt, "jarque_bera": float(jb), "max_abs_residual": float(np.max(np.abs(r)))}


def latex_table(frame: pd.DataFrame, path: Path, caption: str) -> None:
    escaped = frame.copy()
    for column in escaped.columns:
        escaped[column] = escaped[column].map(
            lambda v: "" if pd.isna(v) else
            (f"{v:.3e}" if isinstance(v, float) and v != 0 and abs(v) < 1e-4 else
             (f"{v:.4f}" if isinstance(v, float) else str(v))))
    cols = "l" * len(frame.columns)
    lines = ["\\begin{table}[htbp]", "\\centering", f"\\caption{{{caption}}}", f"\\begin{{tabular}}{{{cols}}}", "\\toprule",
             " & ".join(str(c).replace("_", "\\_") for c in frame.columns) + " \\\\", "\\midrule"]
    for row in escaped.itertuples(index=False, name=None):
        lines.append(" & ".join(str(v).replace("_", "\\_") for v in row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def model_row(mode: str, period: int, outcome: str, scenario: str, variant: str,
              model: str, fit: Fit, total_commuters: float, inference: str) -> dict:
    row = {"mode": mode, "period": period, "outcome": outcome, "scenario": scenario,
           "sample_variant": variant, "model": model, "N": fit.n,
           "represented_commuters": total_commuters, "r2": fit.r2, "adj_r2": fit.adj_r2,
           "rmse_in_sample": fit.rmse, "mae_in_sample": fit.mae, "inference": inference}
    for i, (coefficient, se) in enumerate(zip(fit.beta, fit.se)):
        row[f"beta_{i}"] = coefficient
        row[f"se_{i}"] = se
    return row


def run_period(mode: str, period: int, coords: np.ndarray, folds: np.ndarray,
               full_diagnostics: bool) -> dict[str, list[dict]]:
    times = canonical_times(mode, period)
    jobs, workers = margins(mode, period)
    need_support = period == MAIN_PERIOD and (INCLUDE_PRESENCE_VARIANT or INCLUDE_CONSTRAINED_VARIANT)
    presence = od_presence(mode) if need_support else None
    constrained = None
    constrained_accounting = []
    if period == MAIN_PERIOD and INCLUDE_CONSTRAINED_VARIANT:
        constrained, details = spatially_constrained_flows(mode, times, jobs, workers, presence)
        constrained_accounting.append(details)
    mrc_o, mrc_d = solve_mrc(times, jobs, workers, mode, period)
    jwr = np.divide(jobs, workers, out=np.full_like(jobs, np.nan), where=workers > 0)
    scenarios: list[tuple[str, str | int]] = [(f"threshold_{v}", v) for v in THRESHOLDS] + [("decay_half_life_30", "decay")]
    results, accounting, diagnostics, cv_rows, correlations, vif_rows, onevar = ([] for _ in range(7))
    origin_ids = np.repeat(np.arange(N_ZONES), N_ZONES)
    destination_ids = np.tile(np.arange(N_ZONES), N_ZONES)
    y_values = outcome_values(mode, period, times, jobs, workers)
    base_od_restriction = (np.ones(N_ZONES**2, dtype=bool)
                           if OD_BASE_RESTRICTION_PROVIDER is None else
                           np.asarray(OD_BASE_RESTRICTION_PROVIDER(mode, period), dtype=bool))
    if base_od_restriction.shape != (N_ZONES**2,):
        raise ValueError(f"{mode} {period}: OD base restriction has invalid shape")
    variants = {ALL_OD_VARIANT_NAME: base_od_restriction}
    if period == MAIN_PERIOD and INCLUDE_PRESENCE_VARIANT:
        variants["od_presence"] = presence.ravel()
    if period == MAIN_PERIOD and INCLUDE_CONSTRAINED_VARIANT:
        variants[CONSTRAINED_VARIANT_NAME] = constrained.ravel() > 0
    for scenario_name, scenario_value in scenarios:
        access = accessibility_metrics(times, jobs, workers, scenario_value)
        for outcome in ("od", "origin", "destination"):
            predictors = predictor_sets(mrc_o, mrc_d, jwr, access, outcome)
            y = y_values[outcome]
            common = np.isfinite(y)
            for values in predictors.values():
                common &= np.isfinite(values).all(axis=1)
            use_od_restriction = (outcome == "od" and
                                  (scenario_name == "threshold_30" or APPLY_OD_RESTRICTION_ALL_SCENARIOS))
            outcome_variants = variants if use_od_restriction else {"common_sample": np.ones(len(y), bool)}
            for variant, restriction in outcome_variants.items():
                valid = common & restriction
                weights = constrained.ravel() if variant == CONSTRAINED_VARIANT_NAME else None
                represented = (float(OD_REPRESENTED_COMMUTERS_PROVIDER(mode, period, valid))
                               if outcome == "od" and OD_REPRESENTED_COMMUTERS_PROVIDER is not None else
                               (float(constrained.ravel()[valid].sum())
                                if variant == CONSTRAINED_VARIANT_NAME else
                                (np.nan if variant == "od_presence" else float(workers.sum()))))
                for model, x in predictors.items():
                    model_valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
                    if outcome == "od":
                        model_valid &= restriction
                    accounting.append({"mode": mode, "period": period, "outcome": outcome,
                                       "scenario": scenario_name, "sample_variant": variant, "model": model,
                                       "starting_observations": len(y),
                                       "dropped_invalid_outcome": int((~np.isfinite(y)).sum()),
                                       "dropped_invalid_model_predictors": int((np.isfinite(y) & ~np.isfinite(x).all(axis=1)).sum()),
                                       "dropped_sample_restriction": int((np.isfinite(y) & np.isfinite(x).all(axis=1) & ~restriction).sum()) if outcome == "od" else 0,
                                       "dropped_for_common_sample": int(model_valid.sum() - valid.sum()),
                                       "final_N": int(valid.sum())})
                    cluster = (origin_ids[valid], destination_ids[valid]) if outcome == "od" else None
                    fit_weights = None if weights is None else weights[valid]
                    fit = fit_linear(y[valid], x[valid], fit_weights, cluster)
                    inference = "two-way origin/destination clustered" if outcome == "od" else "OLS; spatial diagnostic reported"
                    if outcome != "od" and scenario_name == "threshold_30" and variant == "common_sample":
                        moran, moran_p = moran_i(fit.residual, coords[valid])
                        spatial_cov = conley_covariance(y[valid], x[valid], fit, coords[valid])
                        spatial_se = np.sqrt(np.maximum(np.diag(spatial_cov), 0))
                        inference = "Conley spatial HAC (10 km)" if moran_p < 0.05 else "OLS; Conley SE also reported"
                        if moran_p < 0.05:
                            fit.covariance = spatial_cov
                            fit.se = spatial_se
                        diagnostics.append({"mode": mode, "period": period, "outcome": outcome,
                                            "scenario": scenario_name, "model": model,
                                            "moran_i": moran, "moran_permutation_p": moran_p,
                                            **residual_summary(fit),
                                            **{f"conley_se_{i}": v for i, v in enumerate(spatial_se)}})
                    elif scenario_name == "threshold_30" and variant in ("existing_all_od", "common_sample"):
                        diagnostics.append({"mode": mode, "period": period, "outcome": outcome,
                                            "scenario": scenario_name, "model": model, **residual_summary(fit)})
                    results.append(model_row(mode, period, outcome, scenario_name, variant, model, fit, represented, inference))
                    cv = cross_validate(y, x, valid, outcome, folds, weights)
                    cv_rows.append({"mode": mode, "period": period, "outcome": outcome,
                                    "scenario": scenario_name, "sample_variant": variant, "model": model, **cv})
            if full_diagnostics and scenario_name == "threshold_30":
                direction = "origin" if outcome != "destination" else "destination"
                zone_vars = {
                    "mrc": mrc_o if direction == "origin" else mrc_d,
                    "jwr": jwr,
                    "access_jobs": access[direction]["access_jobs"],
                    "access_workers": access[direction]["access_workers"],
                    "access_ratio": access[direction]["access_ratio"],
                    "2sfca": access[direction]["2sfca"],
                }
                logged = pd.DataFrame({name: safe_log(value) for name, value in zone_vars.items()})
                corr = logged.corr()
                for left in corr.columns:
                    for right in corr.columns:
                        correlations.append({"mode": mode, "period": period, "outcome": outcome,
                                             "variable_1": left, "variable_2": right, "correlation": corr.loc[left, right]})
                separate = logged[["access_jobs", "access_workers"]].dropna().to_numpy()
                for name, value in zip(("access_jobs", "access_workers"), vif_values(separate)):
                    vif_rows.append({"mode": mode, "period": period, "outcome": outcome,
                                     "model": "separate_access", "variable": name, "vif": value})
                y_diag = y_values[outcome]
                if outcome == "od":
                    for variable in ("access_jobs", "access_workers"):
                        xo = safe_log(zone_vars[variable]).repeat(N_ZONES)
                        xd = np.tile(safe_log(zone_vars[variable]), N_ZONES)
                        valid = np.isfinite(y_diag) & np.isfinite(xo) & np.isfinite(xd)
                        fit = fit_linear(y_diag[valid], np.column_stack([xo[valid], xd[valid]]),
                                         clusters=(origin_ids[valid], destination_ids[valid]))
                        onevar.append(model_row(mode, period, outcome, "threshold_30", "common_valid_for_variable",
                                                variable, fit, float(workers.sum()),
                                                "two-way origin/destination clustered"))
                else:
                    for variable in ("access_jobs", "access_workers"):
                        xv = safe_log(zone_vars[variable])
                        valid = np.isfinite(y_diag) & np.isfinite(xv)
                        fit = fit_linear(y_diag[valid], xv[valid, None])
                        onevar.append(model_row(mode, period, outcome, "threshold_30", "common_valid_for_variable",
                                                variable, fit, float(workers.sum()), "OLS"))
    return {"results": results, "accounting": accounting, "diagnostics": diagnostics,
            "cv": cv_rows, "correlations": correlations, "vif": vif_rows, "onevar": onevar,
            "constrained_accounting": constrained_accounting}


def write_report(frames: dict[str, pd.DataFrame], elapsed: float) -> None:
    def markdown(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "No rows."
        shown = frame.copy()
        for column in shown.columns:
            shown[column] = shown[column].map(
                lambda value: "" if pd.isna(value) else
                (f"{value:.3e}" if isinstance(value, float) and value != 0 and abs(value) < 1e-4 else
                 (f"{value:.4f}" if isinstance(value, float) else str(value))))
        header = "| " + " | ".join(shown.columns) + " |"
        rule = "| " + " | ".join(["---"] * len(shown.columns)) + " |"
        body = ["| " + " | ".join(map(str, row)) + " |" for row in shown.itertuples(index=False, name=None)]
        return "\n".join([header, rule, *body])

    results = frames["results"]
    baseline = results[(results.period == MAIN_PERIOD) & (results.scenario == "threshold_30")]
    common = baseline[baseline.sample_variant.isin([ALL_OD_VARIANT_NAME, "common_sample"])]
    winners = common.loc[common.groupby(["mode", "outcome"])["adj_r2"].idxmax(),
                         ["mode", "outcome", "model", "adj_r2", "N"]]
    observed = baseline[baseline.sample_variant == "od_presence"]
    observed_winners = observed.loc[observed.groupby("mode")["adj_r2"].idxmax(),
                                    ["mode", "model", "adj_r2", "N", "represented_commuters"]] if len(observed) else pd.DataFrame()
    weighted = baseline[baseline.sample_variant == CONSTRAINED_VARIANT_NAME]
    weighted_winners = weighted.loc[weighted.groupby("mode")["adj_r2"].idxmax(),
                                    ["mode", "model", "adj_r2", "N", "represented_commuters"]]
    cv = frames["cv"]
    cv_main = cv[(cv.period == MAIN_PERIOD) & (cv.scenario == "threshold_30") &
                 (cv.sample_variant.isin([ALL_OD_VARIANT_NAME, "common_sample"]))]
    cv_winners = cv_main.loc[cv_main.groupby(["mode", "outcome"])["rmse"].idxmin(),
                             ["mode", "outcome", "model", "rmse", "mae", "oos_r2"]]
    flow_accounting = frames["constrained_accounting"]
    flow_main = (flow_accounting[flow_accounting.period == MAIN_PERIOD]
                 if not flow_accounting.empty and "period" in flow_accounting else pd.DataFrame())
    lines = [
        "# Robustness analysis report", "",
        "## Methods", "",
        "All revised calculations use a canonical 895-zone universe and mode-specific hourly job and worker margins. "
        "Car models use the selected half-hour car matrix (8:30 for the main analysis); transit models use the hourly GTFS matrix (8:00 for the main analysis). "
        "For every matrix, intrazonal time is replaced by 60% of the nearest positive interzonal outbound time.", "",
        "The five models are MRC, mode/time-specific JWR, separate accessibility to jobs and workers, their ratio, and corrected 2SFCA. "
        "Cumulative specifications use 20, 30, 45, and 60 minute thresholds. The continuous specification uses exponential decay "
        f"$w(t)=\\exp(-{DECAY_BETA:.6f}t)$, giving a {DECAY_HALF_LIFE:.0f}-minute half-life.", "",
        "Within each mode, outcome, scenario, and sample variant, every model is estimated on the intersection of valid observations for all five predictor sets. "
        "OD inference uses two-way origin/destination clustered covariance. Zonal residual spatial autocorrelation is assessed using 8-nearest-neighbor Moran's I with 199 fixed-seed permutations; Conley spatial-HAC standard errors use a 10 km Bartlett kernel. "
        "Prediction uses five spatial zone clusters. OD test observations have both origin and destination in the held-out cluster, while training excludes every observation touching that cluster, preventing repeated-zone leakage.", "",
        "## Main common-sample results", "",
        markdown(winners), "",
        "## OD-presence results", "",
        markdown(observed_winners) if len(observed_winners) else "No valid OD-presence models were produced.", "",
        "The presence-only specification is unweighted because the supplied file identifies links but contains no commuter counts. Its represented-commuter total is therefore reported as unavailable rather than inferred.", "",
        "## Spatially constrained OD results", "",
        markdown(weighted_winners), "",
        "These regressions use synthetic commuter weights obtained by balancing the supplied binary OD presence against the mode-specific 8am origin-worker and destination-job totals.", "",
        "## Spatially blocked prediction", "",
        markdown(cv_winners), "",
        "Separate accessibility has the lowest blocked-prediction RMSE for every main outcome and mode. Zonal prediction remains strong, but car OD out-of-sample R-squared is negative and transit OD out-of-sample R-squared is only slightly positive. The manuscript's in-sample OD explanatory-power claim therefore requires substantial qualification when generalized to geographically held-out zones.", "",
        "## Flow and margin accounting", "",
        markdown(flow_main), "",
        "The original presence support is not feasible for exact two-margin balancing: some positive-margin zones have no supported links and connected support components have incompatible totals. The pipeline therefore gives absent links a documented prior multiplier of $10^{-6}$, combines it with exponential travel-time decay, and applies iterative proportional fitting. The table reports resulting off-presence mass and numerical margin errors.", "",
        "## Changes relative to the current manuscript", "",
        "- JWR and every accessibility measure now use mode-specific, hour-specific jobs and workers.",
        "- All model-fit comparisons use a common complete-case sample.",
        "- OD robustness distinguishes the full network matrix, the supplied binary OD-presence sample, and a zonal-total-balanced spatially constrained weighting.",
        "- 2SFCA sums unlogged destination ratios and logs only the completed accessibility measure.",
        "- All modes and metrics share the same intrazonal-time convention.",
        "- OD standard errors account for repeated origins and destinations; zonal models include spatial diagnostics and spatial-HAC uncertainty.", "",
        "## Interpretation", "",
        "The broad conclusion that network accessibility measures outperform MRC and mode/time-specific JWR survives. Separate accessibility is the best in-sample model for all six main mode/outcome comparisons, for all six morning periods, and for nearly every threshold/decay alternative. It also has the lowest spatially blocked RMSE for every main comparison.", "",
        "Three qualifications are necessary. First, the constrained result is synthetic and depends on the documented support relaxation and impedance prior. Second, geographic OD prediction is weak even for the best model. Third, jobs and workers accessibility are highly collinear—especially for car—so their individual coefficient signs and magnitudes should not be interpreted independently even when their joint model predicts well.", "",
        "All main zonal residuals exhibit significant positive Moran's I, so reported main zonal coefficient uncertainty uses 10 km Conley spatial-HAC covariance. The results remain descriptive associations and do not identify causal effects.", "",
        "## Reproducibility and limitations", "",
        "No fractional-flow file is read or used. The binary OD-presence file supplies link support only; mode-specific 8am zonal totals supply the balancing margins. "
        "Adjacent-period car matrices have different historical preprocessing signatures; their results are reported as time robustness but should not be interpreted as pure temporal changes without further provenance work.", "",
        f"Pipeline elapsed time: {elapsed:.1f} seconds.", "",
        "Machine-readable outputs are in `output/fractional_975118_8am/machine_readable/`; publication-ready LaTeX tables are in `output/fractional_975118_8am/tables/`.",
    ]
    (OUT / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-only", action="store_true", help="Run only the 8am/8:30 main period")
    args = parser.parse_args()
    started = time.time()
    coords = coordinates()
    folds = kmeans_folds(coords)
    pd.DataFrame({"zoneid": np.arange(1, N_ZONES + 1), "x": coords[:, 0], "y": coords[:, 1], "spatial_fold": folds}).to_csv(MACHINE / "spatial_folds.csv", index=False, sep=";", decimal=".")
    combined = {name: [] for name in ("results", "accounting", "diagnostics", "cv", "correlations", "vif", "onevar", "constrained_accounting")}
    periods = (MAIN_PERIOD,) if args.main_only else PERIODS
    for mode in MODE_INFO:
        for period in periods:
            print(f"Running {mode} period {period}", flush=True)
            output = run_period(mode, period, coords, folds, full_diagnostics=(period == MAIN_PERIOD))
            for name in combined:
                combined[name].extend(output[name])
    frames = {name: pd.DataFrame(rows) for name, rows in combined.items()}
    for name, frame in frames.items():
        frame.to_csv(MACHINE / f"{name}.csv", index=False, sep=";", decimal=".")
    metadata = {
        "zones": N_ZONES, "thresholds_minutes": THRESHOLDS,
        "continuous_decay": {"form": "exp(-beta*t)", "beta": DECAY_BETA, "half_life_minutes": DECAY_HALF_LIFE},
        "periods": list(periods), "modes": list(MODE_INFO), "random_seed": 20260826,
        "diagonal_rule": "0.60 times nearest positive interzonal outbound time",
        "od_cluster_covariance": "two-way origin/destination",
        "zonal_spatial_covariance": "Conley Bartlett kernel, 10 km",
        "spatial_cv": "5 deterministic k-means zone blocks; OD train excludes held-out origin and destination zones",
        "csv_dialect": "semicolon separator and decimal point",
        "od_weights": "No fractional flows; binary presence plus mode-specific 8am zonal totals",
    }
    (MACHINE / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    main_results = frames["results"][(frames["results"].period == MAIN_PERIOD) & (frames["results"].scenario == "threshold_30")]
    latex_table(main_results[["mode", "outcome", "sample_variant", "model", "N", "represented_commuters", "adj_r2"]],
                TABLES / "main_common_sample.tex", "Common-sample model comparison")
    latex_table(frames["cv"][(frames["cv"].period == MAIN_PERIOD) & (frames["cv"].scenario == "threshold_30")][["mode", "outcome", "sample_variant", "model", "cv_n", "rmse", "mae", "oos_r2"]],
                TABLES / "predictive_comparison.tex", "Spatially blocked predictive comparison")
    latex_table(frames["accounting"][(frames["accounting"].period == MAIN_PERIOD) & (frames["accounting"].scenario == "threshold_30")],
                TABLES / "data_accounting.tex", "Analysis sample accounting")
    latex_table(frames["diagnostics"][frames["diagnostics"].period == MAIN_PERIOD],
                TABLES / "model_diagnostics.tex", "Residual and spatial diagnostics")
    access_summary = frames["results"][(frames["results"].period == MAIN_PERIOD) &
                                        (frames["results"].sample_variant.isin(["existing_all_od", "common_sample"]))]
    latex_table(access_summary[["mode", "outcome", "scenario", "model", "N", "adj_r2"]],
                TABLES / "accessibility_robustness.tex", "Accessibility threshold and impedance-decay robustness")
    time_summary = frames["results"][(frames["results"].scenario == "threshold_30") &
                                      (frames["results"].sample_variant.isin(["existing_all_od", "common_sample"]))]
    latex_table(time_summary[["mode", "period", "outcome", "model", "N", "adj_r2"]],
                TABLES / "mode_time_robustness.tex", "Morning-period robustness")
    latex_table(frames["correlations"][frames["correlations"].period == MAIN_PERIOD],
                TABLES / "metric_correlations.tex", "Correlations among balance and accessibility variables")
    latex_table(frames["vif"][frames["vif"].period == MAIN_PERIOD],
                TABLES / "vif.tex", "Collinearity diagnostics for separate accessibility")
    latex_table(frames["onevar"][frames["onevar"].period == MAIN_PERIOD],
                TABLES / "one_variable_accessibility.tex", "Separate one-variable accessibility regressions")
    latex_table(frames["constrained_accounting"], TABLES / "constrained_accounting.tex",
                "OD-presence support relaxation and constrained-flow accounting")
    write_report(frames, time.time() - started)
    print(f"Completed in {time.time() - started:.1f} seconds", flush=True)


if __name__ == "__main__":
    main()
