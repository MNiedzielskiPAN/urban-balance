"""Create manuscript tables directly from authoritative outputs."""

from __future__ import annotations

from pathlib import Path

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "data").is_dir() and (parent / "manuscript").is_dir())

import numpy as np
import pandas as pd

NEW = ROOT / "output" / "fractional_975118_8am"
M = NEW / "machine_readable"
T = NEW / "tables"
T.mkdir(parents=True, exist_ok=True)
MANUSCRIPT_TABLES = ROOT / "manuscript" / "generated_tables" / "fractional_975118_8am"
MANUSCRIPT_TABLES.mkdir(parents=True, exist_ok=True)

MODE = {"car": "Car", "pt": "Public transport"}
OUTCOME = {"pairwise": "Pairwise network time", "out_commute": "Out-commute", "in_commute": "In-commute"}
MODEL = {"mrc": "MRC", "jwr": "JWR", "separate_access": "Separate access", "access_ratio": "Access ratio", "2sfca": "2SFCA"}
ORDER = ["mrc", "jwr", "separate_access", "access_ratio", "2sfca"]


def read(folder: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(folder / "machine_readable" / name, sep=";", decimal=".")


def write(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(M / name, sep=";", decimal=".", index=False, float_format="%.12g")


def fmt(value, digits=3) -> str:
    if pd.isna(value):
        return "--"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return f"{float(value):.{digits}f}"


def table(path: Path, caption: str, label: str, columns: list[str], rows: list[list[str]],
          landscape=False, note: str | None = None) -> None:
    prefix = ["\\begin{landscape}"] if landscape else []
    suffix = ["\\end{landscape}"] if landscape else []
    lines = prefix + ["\\begin{table}[htbp]", "\\centering", "\\scriptsize",
                      f"\\caption{{{caption}}}", f"\\label{{{label}}}",
                      "\\begin{tabular}{" + "l" * len(columns) + "}", "\\toprule",
                      " & ".join(columns) + " \\\\", "\\midrule"]
    lines.extend(" & ".join(row) + " \\\\" for row in rows)
    lines += ["\\bottomrule", "\\end{tabular}"]
    if note:
        lines += ["\\par\\vspace{2pt}",
                  "\\begin{minipage}{0.96\\linewidth}\\footnotesize\\textit{Notes:} " + note,
                  "\\end{minipage}"]
    lines += ["\\end{table}"] + suffix
    contents = "\n".join(lines) + "\n"
    path.write_text(contents, encoding="utf-8")
    # Keep a manuscript-local copy so the TeX source can be compiled from the
    # manuscript directory or uploaded with that directory as a self-contained
    # submission package. Both copies are generated from the same results.
    if path.parent == T:
        (MANUSCRIPT_TABLES / path.name).write_text(contents, encoding="utf-8")


def tables() -> None:
    results = read(NEW, "results.csv")
    main = results[(results.threshold_minutes == 30) & (results.sample_variant == "threshold_common")]
    rows = []
    for (mode, outcome), group in main.groupby(["mode", "outcome"], sort=False):
        values = group.set_index("model")
        rows.append([MODE[mode], OUTCOME[outcome], fmt(int(group.N.iloc[0]), 0)] +
                    [fmt(values.loc[m, "adj_r2"]) for m in ORDER])
    table(T / "manuscript_main_30.tex", "Common-sample model comparison at 8am and 30 minutes",
          "tab:r2summary", ["Mode", "Outcome", "$N$", "MRC", "JWR", "Separate access", "Access ratio", "2SFCA"], rows,
          note="Cells report adjusted $R^2$ from OLS models in natural-log units. Within each mode--outcome row, all five models use the same valid observations.")

    selected = pd.concat([
        results[(results["mode"] == "car") & (results.sample_variant == "threshold_common")],
        results[(results["mode"] == "pt") & (results.sample_variant == "all_threshold_intersection")],
    ])
    rows = []
    for (mode, outcome, threshold), group in selected.groupby(["mode", "outcome", "threshold_minutes"], sort=False):
        values = group.set_index("model")
        rows.append([MODE[mode], OUTCOME[outcome], f"{int(threshold)} min", fmt(int(group.N.iloc[0]), 0)] +
                    [fmt(values.loc[m, "adj_r2"]) for m in ORDER])
    table(T / "manuscript_thresholds.tex", "8am accessibility-threshold robustness: common-sample adjusted $R^2$",
          "tab:thresholds", ["Mode", "Outcome", "Threshold", "$N$", "MRC", "JWR", "Separate access", "Access ratio", "2SFCA"], rows, True,
          note="Cells report adjusted $R^2$ from OLS models in natural-log units. Each row uses a five-model common sample. Public transport uses the same all-threshold intersection at every cutoff; its accessibility measures nevertheless retain the full 895-zone opportunity universe.")

    separate = main[main.model == "separate_access"]
    rows = []
    for row in separate.itertuples():
        if row.outcome == "pairwise":
            values = [f"{row.beta_1:.3f} ({row.se_1:.3f})", f"{row.beta_2:.3f} ({row.se_2:.3f})",
                      f"{row.beta_3:.3f} ({row.se_3:.3f})", f"{row.beta_4:.3f} ({row.se_4:.3f})"]
        elif row.outcome == "out_commute":
            values = [f"{row.beta_1:.3f} ({row.se_1:.3f})", f"{row.beta_2:.3f} ({row.se_2:.3f})", "--", "--"]
        else:
            values = ["--", "--", f"{row.beta_1:.3f} ({row.se_1:.3f})", f"{row.beta_2:.3f} ({row.se_2:.3f})"]
        rows.append([MODE[row.mode], OUTCOME[row.outcome], fmt(row.N, 0)] + values)
    table(T / "manuscript_coefficients_30.tex", "Separate-accessibility coefficients at 8am and 30 minutes",
          "tab:separatecoeff", ["Mode", "Outcome", "$N$", "Origin jobs", "Origin workers", "Destination jobs", "Destination workers"], rows,
          note="Standard errors are in parentheses. Pairwise standard errors use two-way origin/destination clustering; zonal standard errors use 10-km Bartlett-kernel Conley spatial-HAC covariance.")

    cv = read(NEW, "cv_results.csv")
    cv30 = cv[(cv.threshold_minutes == 30) & (cv.sample_variant == "threshold_common")]
    winners = cv30.loc[cv30.groupby(["mode", "outcome"]).rmse_log_units.idxmin()]
    rows = [[MODE[r.mode], OUTCOME[r.outcome], MODEL[r.model], fmt(r.cv_test_N, 0), fmt(r.rmse_log_units),
             fmt(r.mae_log_units), fmt(r.oos_r2_training_mean), fmt(r.oos_r2_heldout_mean)] for r in winners.itertuples()]
    table(T / "manuscript_prediction_winners_30.tex", "Best spatially blocked predictive model at 8am and 30 minutes",
          "tab:prediction", ["Mode", "Outcome", "Model", "Test $N$", "RMSE", "MAE", "$R^2_{train}$", "$R^2_{test}$"], rows,
          note="The selected model has the lowest pooled held-out RMSE. RMSE and MAE are in natural-log outcome units. $R^2_{train}$ benchmarks each fold against its training-outcome mean; $R^2_{test}$ uses its held-out-outcome mean. Sums of squares are pooled across folds.")

    rows = [[MODE[r.mode], OUTCOME[r.outcome], MODEL[r.model], fmt(r.cv_test_N, 0), fmt(r.rmse_log_units),
             fmt(r.mae_log_units), fmt(r.oos_r2_training_mean), fmt(r.oos_r2_heldout_mean)]
            for r in cv30.itertuples()]
    table(T / "manuscript_prediction_all_30.tex", "Spatially blocked prediction for all five models at 30 minutes",
          "tab:cvall30", ["Mode", "Outcome", "Model", "Test $N$", "RMSE", "MAE", "$R^2_{train}$", "$R^2_{test}$"], rows, True,
          note="RMSE and MAE are pooled across held-out observations and expressed in natural-log outcome units. $R^2_{train}$ uses each fold's training-outcome mean as the reference; $R^2_{test}$ uses the held-out-outcome mean. Pairwise test observations have both endpoints in the held-out block.")

    residual = read(NEW, "residual_diagnostics.csv")
    residual = residual[(residual.threshold_minutes == 30) &
                        (residual.sample_variant == "threshold_common") &
                        (residual.model == "separate_access") &
                        (residual.outcome != "pairwise")]
    vif = read(NEW, "vif.csv")
    vif = vif[(vif.threshold_minutes == 30) & (vif.variable == "access_jobs")]
    corr = read(NEW, "correlations.csv")
    corr = corr[(corr.threshold_minutes == 30) & (corr.variable_1 == "access_jobs") &
                (corr.variable_2 == "access_workers")]
    rows = []
    for r in residual.itertuples():
        v = vif[(vif["mode"] == r.mode) & (vif.outcome == r.outcome)].iloc[0]
        c = corr[(corr["mode"] == r.mode) & (corr.outcome == r.outcome)].iloc[0]
        rows.append([MODE[r.mode], OUTCOME[r.outcome], fmt(r.moran_i), fmt(r.moran_permutation_p),
                     fmt(c.correlation), fmt(v.vif, 2)])
    table(T / "manuscript_diagnostics_30.tex", "Spatial-dependence and collinearity diagnostics at 30 minutes",
          "tab:diagnostics", ["Mode", "Outcome", "Moran's $I$", "Permutation $p$", "$\\mathrm{corr}(A_E,A_W)$", "VIF"], rows,
          note="Moran's $I$ is calculated for separate-accessibility residuals on the eight-nearest-neighbour graph using 199 fixed-seed permutations. VIF is reported for jobs accessibility; the two-variable specification gives the same VIF for workers accessibility.")

    one = read(NEW, "one_variable_regressions.csv")
    one = one[one.threshold_minutes == 30]
    rows = []
    for r in one.itertuples():
        if r.outcome == "pairwise":
            origin = f"{r.beta_origin:.3f} ({r.se_origin:.3f})"
            destination = f"{r.beta_destination:.3f} ({r.se_destination:.3f})"
        elif r.outcome == "out_commute":
            origin, destination = f"{r.beta:.3f} ({r.standard_error:.3f})", "--"
        else:
            origin, destination = "--", f"{r.beta:.3f} ({r.standard_error:.3f})"
        rows.append([MODE[r.mode], OUTCOME[r.outcome], "Jobs" if r.variable == "access_jobs" else "Workers",
                     fmt(r.N, 0), origin, destination, fmt(r.adj_r2)])
    table(T / "manuscript_one_variable_30.tex", "One-variable accessibility regressions at 30 minutes",
          "tab:onevariable", ["Mode", "Outcome", "Variable", "$N$", "Origin coefficient", "Destination coefficient", "Adjusted $R^2$"], rows,
          note="Standard errors are in parentheses; pairwise models use two-way origin/destination clustering and zonal models use 10-km Conley spatial-HAC covariance. Adjusted $R^2$ is calculated for OLS models in natural-log units. Each one-variable regression uses the observations valid for that variable, so its $N$ can exceed the five-model common sample in Table~\\ref{tab:r2summary}; for example, public transport in-commute has $N=886$ here and $N=794$ there. Fits are therefore not directly comparable across those tables.")

    restriction = read(NEW, "restriction_tests.csv")
    restriction = restriction[(restriction.threshold_minutes == 30) &
                              (restriction.sample_variant == "threshold_common")]
    rows = [[MODE[r.mode], OUTCOME[r.outcome], fmt(r.N, 0), fmt(r.wald_chi2, 2), str(int(r.df)),
             "<0.001" if r.p_value < 0.001 else fmt(r.p_value)] for r in restriction.itertuples()]
    table(T / "manuscript_ratio_restrictions_30.tex", "Wald tests of the accessibility-ratio restriction at 30 minutes",
          "tab:ratiorestriction", ["Mode", "Outcome", "$N$", "$\\chi^2$", "df", "$p$"], rows,
          note="Robust Wald tests are based on the separate-accessibility model estimated on the five-model common sample. Pairwise covariance is clustered two ways by origin and destination; zonal covariance is Conley spatial-HAC.")

    # Export the exact source rows used by every manuscript fragment for automated checking.
    write(main, "manuscript_source_main_30.csv")
    write(selected, "manuscript_source_thresholds.csv")
    write(cv30, "manuscript_source_prediction_30.csv")
    write(one, "manuscript_source_one_variable_30.csv")
    write(restriction, "manuscript_source_ratio_restrictions_30.csv")


if __name__ == "__main__":
    tables()
    print("Authoritative manuscript table fragments created.")
