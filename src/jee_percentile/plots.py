"""Every chart in the README, rendered from data + reports/metrics.json.

Run `python -m jee_percentile figures` (after `evaluate`).
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from jee_percentile.config import FIGURES_DIR, PERCENTILES, REPORTS_DIR  # noqa: E402
from jee_percentile.curves import normal_cdf_pct  # noqa: E402
from jee_percentile.data import load_curves, load_dataset  # noqa: E402
from jee_percentile.evaluate import spearman_table  # noqa: E402
from jee_percentile.train import train  # noqa: E402

INK, MUTED, GRID = "#1E293B", "#64748B", "#E2E8F0"
INDIGO, TEAL, AMBER, ROSE, SKY = "#4F46E5", "#0D9488", "#F59E0B", "#E11D48", "#0EA5E9"
COHORTS = {
    (2023, "January"): ("2023 · January", "#94A3B8"),
    (2025, "January"): ("2025 · January", TEAL),
    (2025, "April"): ("2025 · April", AMBER),
    (2026, "January"): ("2026 · January", INDIGO),
}
SEQ = LinearSegmentedColormap.from_list("jee", ["#E0E7FF", "#818CF8", "#4338CA", "#1E1B4B"])
DIV = LinearSegmentedColormap.from_list("jee_div", [ROSE, "#FDF2F8", "#F8FAFC", "#EEF2FF", INDIGO])

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.25,
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 26,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
    "legend.frameon": False, "axes.prop_cycle": plt.cycler(color=[INDIGO, TEAL, AMBER, ROSE, SKY]),
})


def _subtitle(ax, text):
    ax.text(0, 1.015, text, transform=ax.transAxes, color=MUTED, fontsize=9.5, va="bottom")


def _save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def _metrics():
    return json.loads((REPORTS_DIR / "metrics.json").read_text())


# ------------------------------------------------------------------ figures

def percentile_curves():
    curves = load_curves()
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for (year, session), grp in curves.groupby(["year", "session"]):
        label, color = COHORTS[(year, session)]
        for i, (_, shift) in enumerate(grp.groupby("shift_id")):
            shift = shift.sort_values("raw_score")
            ax.plot(shift["raw_score"], shift["percentile"], color=color, lw=1.6, alpha=0.75,
                    marker="o", ms=2.5, label=label if i == 0 else None)
    ax.set(xlabel="Raw marks (out of 300)", ylabel="Percentile", ylim=(89.5, 100.2))
    ax.set_title("Same marks, different percentile")
    _subtitle(ax, "Published marks-vs-percentile tables for 41 JEE Main shifts. "
                  "The 95th-percentile cutoff alone spans 96 to 147 marks.")
    ax.legend(loc="lower right", ncol=2)
    return _save(fig, "percentile_curves.png")


def stage1_fits():
    curves = load_curves()
    ids = ["2026_22_JAN_2", "2025_28_JAN_1", "2025_04_APR_2", "2023_25_JAN_1"]
    ids = [i for i in ids if i in set(curves["shift_id"])] or list(curves["shift_id"].unique()[:4])
    from jee_percentile.curves import fit_normal_cdf
    fig, axes = plt.subplots(1, len(ids), figsize=(13, 3.6), sharey=True)
    for ax, sid in zip(np.atleast_1d(axes), ids):
        s = curves[curves["shift_id"] == sid].sort_values("raw_score")
        fit = fit_normal_cdf(s["raw_score"].to_numpy(), s["percentile"].to_numpy())
        xs = np.linspace(s["raw_score"].min() - 10, s["raw_score"].max() + 10, 200)
        color = COHORTS[(s["year"].iat[0], s["session"].iat[0])][1]
        ax.plot(xs, normal_cdf_pct(xs, fit["mu"], fit["sigma"]), color=INK, lw=1.4, zorder=1)
        ax.scatter(s["raw_score"], s["percentile"], color=color, s=26, zorder=2, edgecolor="white", lw=0.6)
        ax.set_title(sid.replace("_", " "), fontsize=11)
        ax.text(0.97, 0.06, f"μ={fit['mu']:.0f}  σ={fit['sigma']:.0f}\nRMSE {fit['rmse_pct']:.2f} pts",
                transform=ax.transAxes, ha="right", fontsize=9, color=MUTED)
        ax.set_xlabel("Raw marks")
    np.atleast_1d(axes)[0].set_ylabel("Percentile")
    fig.suptitle("Stage 1 · every shift compresses to a two-number Normal-CDF fingerprint",
                 x=0.06, ha="left", fontweight="bold", color=INK, y=1.08)
    return _save(fig, "stage1_fits.png")


def difficulty_signal():
    df = load_dataset()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, (feat, target, xlabel) in zip(axes, [
        ("phy_easy", "score_95.0", "Physics questions labelled easy (%)"),
        ("phy_hard", "score_99.0", "Physics questions labelled hard (%)"),
    ]):
        for (year, session), grp in df.groupby(["year", "session"]):
            label, color = COHORTS[(year, session)]
            ax.scatter(grp[feat], grp[target], s=46, color=color, label=label, edgecolor="white", lw=0.8, zorder=3)
        slope, intercept = np.polyfit(df[feat], df[target], 1)
        xs = np.linspace(df[feat].min(), df[feat].max(), 50)
        ax.plot(xs, slope * xs + intercept, color=INK, lw=1.2, ls="--", zorder=2)
        rho = df[[feat, target]].corr(method="spearman").iloc[0, 1]
        ax.set(xlabel=xlabel, ylabel=f"Marks needed for {target[6:-2]}th percentile")
        ypos = 0.93 if feat == "phy_easy" else 0.04
        ax.text(0.03, ypos, f"Spearman ρ = {rho:+.2f}", transform=ax.transAxes, color=INK, fontweight="bold")
    axes[0].set_title("Easier Physics → higher cutoffs")
    axes[1].set_title("Harder Physics → lower cutoffs at the top")
    axes[1].legend(loc="upper right", fontsize=9)
    return _save(fig, "difficulty_signal.png")


def correlation_heatmap():
    tab = spearman_table(load_dataset())
    pivot = tab.pivot(index="feature", columns="target", values="rho")
    order = ["phy_easy", "phy_hard", "chem_easy", "chem_hard", "math_easy", "math_hard",
             "overall_easy", "overall_hard", "is_evening"]
    pivot = pivot.loc[order, ["marks@90", "marks@95", "marks@99"]]
    pvals = tab.pivot(index="feature", columns="target", values="p_value").loc[pivot.index, pivot.columns]
    fig, ax = plt.subplots(figsize=(6.2, 6))
    im = ax.imshow(pivot.values, cmap=DIV, vmin=-0.8, vmax=0.8, aspect="auto")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            star = "★" if pvals.values[i, j] < 0.05 else ""
            ax.text(j, i, f"{pivot.values[i, j]:+.2f}{star}", ha="center", va="center",
                    color=INK if abs(pivot.values[i, j]) < 0.55 else "white", fontsize=10)
    ax.set_xticks(range(3), ["90th", "95th", "99th"])
    ax.set_yticks(range(len(order)), [o.replace("_", " ") for o in order])
    ax.grid(False)
    ax.set_title("Which difficulty signal moves the cutoff?")
    _subtitle(ax, "Spearman ρ vs marks needed per percentile · ★ p < 0.05 · n = 29 shifts")
    fig.colorbar(im, ax=ax, shrink=0.75, label="Spearman ρ")
    return _save(fig, "correlation_heatmap.png")


def loso_parity():
    pred = pd.read_csv(REPORTS_DIR / "loso_predictions.csv")
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(pred["actual_marks"], pred["predicted_marks"], c=pred["percentile"], cmap=SEQ,
                    s=26, edgecolor="white", lw=0.5, zorder=3)
    lo, hi = pred[["actual_marks", "predicted_marks"]].min().min() - 5, pred[["actual_marks", "predicted_marks"]].max().max() + 5
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--", zorder=2)
    ax.fill_between([lo, hi], [lo - 10, hi - 10], [lo + 10, hi + 10], color=INDIGO, alpha=0.06, zorder=1)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Actual marks needed", ylabel="Predicted marks needed (held-out)")
    ax.set_title("Out-of-sample cutoffs")
    _subtitle(ax, "Leave-one-shift-out · shaded band = ±10 marks")
    fig.colorbar(sc, ax=ax, shrink=0.75, label="Percentile")
    return _save(fig, "loso_parity.png")


def model_comparison():
    m = _metrics()
    rows = m["leaderboard"][:2] + m["ablation"]
    names = [r["name"] for r in rows]
    mae = np.array([r["pct_mae"] for r in rows])
    ci = np.array([r["pct_mae_ci"] for r in rows])
    order = np.argsort(-mae)
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    colors = [INDIGO if "deployed" in names[i] else ("#CBD5E1" if i < 2 else "#A5B4FC") for i in order]
    y = np.arange(len(order))
    ax.barh(y, mae[order], color=colors, height=0.62, zorder=2)
    ax.errorbar(mae[order], y, xerr=[mae[order] - ci[order, 0], ci[order, 1] - mae[order]],
                fmt="none", ecolor=INK, elinewidth=1, capsize=3, zorder=3)
    for yi, i in zip(y, order):
        ax.text(ci[i, 1] + 0.03, yi, f"{mae[i]:.3f}", va="center", color=INK, fontsize=9.5)
    ax.set_yticks(y, [names[i] for i in order])
    ax.set_xlabel("Leave-one-shift-out percentile MAE (lower is better) · bars = 95% bootstrap CI")
    ax.grid(axis="y", visible=False)
    ax.set_title("Baselines and feature ablation")
    _subtitle(ax, "Grey = no-difficulty baselines · Physics + session time scores slightly better "
                  "but needs an input practice tests don't have")
    return _save(fig, "model_comparison.png")


def diagnostics():
    m = _metrics()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 3.9))
    per = m["per_percentile_mae"]
    a1.bar([f"{float(k):g}" for k in per], list(per.values()), color=TEAL, width=0.65, zorder=2)
    a1.set(xlabel="Percentile tier", ylabel="MAE (percentile points)")
    a1.set_title("Error by tier")
    a1.grid(axis="x", visible=False)
    sweep = pd.DataFrame(m["alpha_sweep"])
    a2.plot(sweep["alpha"], sweep["pct_mae"], color=INDIGO, marker="o", lw=2)
    best = sweep.loc[sweep["pct_mae"].idxmin()]
    a2.scatter([best["alpha"]], [best["pct_mae"]], s=140, facecolor="none", edgecolor=ROSE, lw=2, zorder=4)
    a2.annotate(f"α = {best['alpha']:g}", (best["alpha"], best["pct_mae"]), xytext=(10, 18),
                textcoords="offset points", color=ROSE, fontweight="bold")
    a2.set_xscale("log")
    a2.set(xlabel="Ridge α (log scale)", ylabel="LOSO percentile MAE")
    a2.set_title("Regularisation sweep")
    return _save(fig, "diagnostics.png")


def prediction_band():
    model = train()
    marks = np.linspace(60, 260, 400)
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, ph, pe, color in [("Easy paper · Physics 0% hard, 56% easy", 0, 56, TEAL),
                                 ("Hard paper · Physics 12% hard, 28% easy", 12, 28, ROSE)]:
        q = model.query(ph, pe)
        curve = model.predict_curve(q)[0]
        bands = np.array([model.predict_interval(x, q) for x in marks])
        inside = (marks > curve[0]) & (marks < curve[-1])  # the model only covers 90-99
        ax.fill_between(marks, bands[:, 0], bands[:, 2], where=inside, color=color, alpha=0.15, lw=0)
        ax.plot(marks[inside], bands[inside, 1], color=color, lw=2.4, label=label)
    ax.axvline(150, color=MUTED, lw=1, ls=":")
    ax.set(xlabel="Raw marks (out of 300)", ylabel="Predicted percentile", ylim=(89.7, 99.3), xlim=(60, 200))
    ax.set_title("150 marks can mean very different things")
    _subtitle(ax, "Predicted curve with calibrated 80% band (back-tested coverage 80%) for two papers")
    ax.legend(loc="lower right")
    return _save(fig, "prediction_band.png")


def whatif_heatmap():
    model = train()
    hard = np.arange(0, 12.1, 0.5)  # stay inside the observed range
    easy = np.arange(20, 64.1, 2)
    grid = np.array([[model.predict_curve(model.query(h, e))[0][-1] for h in hard] for e in easy])
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap=SEQ,
                   extent=[hard[0] - 0.5, hard[-1] + 0.5, easy[0] - 1, easy[-1] + 1])
    cs = ax.contour(hard, easy, grid, levels=8, colors="white", linewidths=0.9)
    ax.clabel(cs, fmt="%d", fontsize=8.5, colors="white")
    df = load_dataset()
    ax.scatter(df["phy_hard"], df["phy_easy"], s=18, color=AMBER, edgecolor=INK, lw=0.5, label="observed shifts")
    ax.set(xlabel="Physics questions labelled hard (%)", ylabel="Physics questions labelled easy (%)")
    ax.grid(False)
    ax.set_title("What-if: marks needed for the 99th percentile")
    _subtitle(ax, "Model surface over the Physics difficulty space · dots = real shifts used for training")
    ax.legend(loc="lower left", fontsize=9, facecolor="white", frameon=True, framealpha=0.9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Marks for 99th percentile")
    return _save(fig, "whatif_heatmap.png")


def render_all():
    return [f() for f in (percentile_curves, stage1_fits, difficulty_signal, correlation_heatmap,
                          loso_parity, model_comparison, diagnostics, prediction_band, whatif_heatmap)]


if __name__ == "__main__":
    for p in render_all():
        print(p)
