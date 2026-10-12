"""Independent checks for the fractional-OD, rounded-margin analysis."""

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "data").is_dir() and (parent / "manuscript").is_dir())
OUT = ROOT / "output" / "fractional_975118_8am"
M = OUT / "machine_readable"


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(M / name, sep=";", decimal=".")


def main() -> None:
    metadata = json.loads((M / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["jobs_total"] == 975118.0
    assert metadata["workers_total"] == 975118.0
    rounding = metadata["flow_and_rounding"]
    assert abs(rounding["fractional_total"] - 975117.8441822712) < 1e-5
    assert rounding["integer_target"] == 975118
    assert rounding["od_cells_modified"] == 0
    totals = read("fractional_and_integer_zonal_totals.csv")
    assert totals.workers.sum() == totals.jobs.sum() == 975118
    assert (np.abs(totals.worker_rounding_change) < 1).all()
    assert (np.abs(totals.job_rounding_change) < 1).all()

    results = read("results.csv")
    expected = {"car", "pt"}, {20, 30, 45, 60}, {"pairwise", "out_commute", "in_commute"}
    assert set(results["mode"]) == expected[0]
    assert set(results["threshold_minutes"]) == expected[1]
    assert set(results["outcome"]) == expected[2]
    assert np.allclose(results["represented_fractional_flow"], 975117.8441822712)
    assert not results.duplicated(["mode", "threshold_minutes", "outcome", "sample_variant", "model"]).any()
    counts = results.groupby(["mode", "threshold_minutes", "outcome", "sample_variant"]).size()
    assert (counts == 5).all()

    fixed = results[(results["mode"] == "pt") &
                    (results["sample_variant"] == "all_threshold_intersection")]
    for outcome, group in fixed.groupby("outcome"):
        assert group["N"].nunique() == 1, f"Public-transport fixed sample varies for {outcome}"

    sfca = read("2sfca_accounting.csv")
    assert set(sfca.direction) == {"origin", "destination"}
    assert (sfca.groupby(["mode", "threshold_minutes", "direction"]).size() == 1).all()
    destination = sfca[sfca.direction == "destination"]
    assert destination.reverses_direction.astype(bool).all()
    assert destination.exchanges_jobs_workers.astype(bool).all()
    assert sfca.zero_denominator_treatment.str.contains("logged only after summation").all()

    restrictions = read("restriction_tests.csv")
    assert not restrictions.p_value.isna().any()
    assert ((restrictions.p_value >= 0) & (restrictions.p_value <= 1)).all()

    cv = read("cv_results.csv")
    assert not cv.duplicated(["mode", "threshold_minutes", "outcome", "sample_variant", "model"]).any()
    assert (cv.groupby(["mode", "threshold_minutes", "outcome", "sample_variant"]).size() == 5).all()
    assert cv.aggregation.str.contains("pooled").all()
    assert not cv[["oos_r2_training_mean", "oos_r2_heldout_mean"]].isna().any().any()
    folds = read("cv_fold_results.csv")
    assert (folds.groupby(["mode", "threshold_minutes", "outcome", "sample_variant", "model"]).size() == 5).all()
    pairwise = folds[folds.outcome == "pairwise"]
    assert pairwise.test_scope.str.contains("within-held-out-block").all()

    onevar = read("one_variable_regressions.csv")
    assert len(onevar) > 0
    zonal_onevar = onevar[onevar.outcome != "pairwise"]
    pairwise_onevar = onevar[onevar.outcome == "pairwise"]
    assert not zonal_onevar.beta.isna().any()
    assert not pairwise_onevar[["beta_origin", "beta_destination"]].isna().any().any()
    assert set(pairwise_onevar.variable) == {"access_jobs", "access_workers"}
    accounting = read("sample_accounting.csv")
    assert (accounting.final_N > 0).all()
    assert np.all(accounting.starting_observations - accounting.dropped_invalid_outcome >= accounting.final_N)

    print("Validation passed: 975118 margins, unchanged fractional cells, samples, 2SFCA, restrictions, dual-benchmark CV, and accounting.")


if __name__ == "__main__":
    main()
