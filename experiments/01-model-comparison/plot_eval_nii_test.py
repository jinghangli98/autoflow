"""Publication figure + summary tables from evaluate_nii_test.py CSVs.

Reads one or more per-slice metric CSVs (shards are concatenated), averages
slices to per-volume scores, and renders a metric-panel figure:

  rows = task (raw = artifact -> fully sampled, md = -> denoised+biascorrected)
  cols = PSNR / SSIM / LPIPS / FSIM by default (--metrics; MAE is still in
         the summary table, just not plotted -- it tracks PSNR)
  x    = artifact family (undersampled / spike / aniso)
  hue  = condition: input (degraded, no model), cfg0 (unconditioned),
         cfg1 (conditioned), plus the 2D baselines swinir / realesrgan when
         their CSVs (evaluate_baseline_nii_test.py) are passed alongside --
         box + jittered per-volume points. Duplicate "input" volumes across
         CSVs collapse in the per-volume aggregation (identical slice scores).

Styling follows Nature figure guidelines: Helvetica/Arial, 7 pt, no top/right
spines, 183 mm double-column width, vector PDF + 600 dpi PNG. A mean+-sd
summary table is written next to the figure. `--stats` adds paired Wilcoxon
cfg1-vs-input significance stars per (task, artifact) group.

Usage:
    python plot_eval_nii_test.py eval_nii_test_shard*.csv \
        --out_prefix figures/eval_nii_test [--stats]
    python plot_eval_nii_test.py eval_nii_test_fg5.csv \
        eval_nii_test_swinir_{raw,denoised}.csv \
        eval_nii_test_realesrgan_{raw,denoised}.csv \
        --out_prefix figures/eval_nii_test_methods --stats
"""

import argparse
import os

import numpy as np
import pandas as pd

METRICS = ["psnr", "ssim", "mae", "lpips", "fsim"]        # CSV columns
PLOT_METRICS = ["psnr", "ssim", "lpips", "fsim"]         # figure columns
METRIC_LABEL = {"psnr": "PSNR (dB) ↑", "ssim": "SSIM ↑", "mae": "MAE ↓",
                "lpips": "LPIPS ↓", "fsim": "FSIM ↑"}
TASK_LABEL = {"raw": "Artifact → fully sampled",
              "md": "→ denoised + bias-corrected"}
ARTIFACT_ORDER = ["undersampled", "spike", "aniso"]
ARTIFACT_LABEL = {"undersampled": "GRAPPA", "spike": "Spike",
                  "aniso": "Aniso."}
# Plot/legend order: baselines between the input anchor and the flow model.
CONDITION_ORDER = ["input", "realesrgan", "swinir", "cfg0", "cfg1"]
CONDITION_LABEL = {"input": "Input (no model)",
                   "cfg0": "Unconditioned (CFG 0)",
                   "cfg1": "Conditioned (CFG 1)",
                   "swinir": "SwinIR (2D)",
                   "realesrgan": "Real-ESRGAN (2D)"}
# Validated categorical palette (dataviz reference slots 1-5 in the documented
# order: slots 1-3 all-pairs PASS, slots 4-5 pass the adjacent-pair gates).
CONDITION_COLOR = {"input": "#2a78d6", "cfg0": "#eb6834", "cfg1": "#1baf7a",
                   "swinir": "#eda100", "realesrgan": "#e87ba4"}

VOLUME_KEYS = ["anatomy", "acquisition", "subject", "input", "artifact",
               "severity", "task", "condition"]


def aggregate_per_volume(df):
    """Per-slice rows -> one row per scored volume (slice-mean per metric)."""
    df = df.copy()
    df["severity"] = df["severity"].fillna("")
    return (df.groupby(VOLUME_KEYS, dropna=False)[METRICS]
              .mean().reset_index())


def summary_table(agg):
    """mean +- sd (and n volumes) per (task, artifact, condition)."""
    g = agg.groupby(["task", "artifact", "condition"])[METRICS]
    out = g.agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out["n_volumes"] = g.size()
    return out.reset_index()


def _style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6.5,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def _stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


def make_figure(agg, out_prefix, stats=False, metrics=PLOT_METRICS):
    import matplotlib.pyplot as plt

    _style()
    tasks = [t for t in ("raw", "md") if t in set(agg["task"])]
    fams_by_task = {
        t: [a for a in ARTIFACT_ORDER
            if a in set(agg.loc[agg["task"] == t, "artifact"])]
        for t in tasks
    }
    conds = [c for c in CONDITION_ORDER if c in set(agg["condition"])]

    mm = 1 / 25.4
    fig, axes = plt.subplots(len(tasks), len(metrics),
                             figsize=(183 * mm, 55 * mm * len(tasks)),
                             squeeze=False)
    rng = np.random.default_rng(0)
    # Fit all conditions into ~80% of the unit group spacing so neighbouring
    # artifact groups stay visibly separated however many conditions plot.
    gap = 0.03
    box_w = 0.8 / max(1, len(conds)) - gap

    for r, task in enumerate(tasks):
        fams = fams_by_task[task]
        tdf = agg[agg["task"] == task]
        for c, metric in enumerate(metrics):
            ax = axes[r][c]
            for fi, fam in enumerate(fams):
                fdf = tdf[tdf["artifact"] == fam]
                for ci, cond in enumerate(conds):
                    vals = fdf.loc[fdf["condition"] == cond, metric].dropna().values
                    if len(vals) == 0:
                        continue
                    x = fi + (ci - (len(conds) - 1) / 2) * (box_w + gap)
                    color = CONDITION_COLOR[cond]
                    bp = ax.boxplot(
                        vals, positions=[x], widths=box_w, patch_artist=True,
                        showfliers=False,
                        boxprops=dict(facecolor=color, alpha=0.45,
                                      edgecolor="#333333", linewidth=0.6),
                        whiskerprops=dict(color="#333333", linewidth=0.6),
                        capprops=dict(color="#333333", linewidth=0.6),
                        medianprops=dict(color="#111111", linewidth=0.9),
                    )
                    ax.scatter(x + rng.uniform(-box_w / 4, box_w / 4, len(vals)),
                               vals, s=2.2, color=color, edgecolors="none",
                               alpha=0.75, zorder=3)
                if stats and {"input", "cfg1"} <= set(conds):
                    _annotate_stat(ax, fdf, fi, metric, box_w, gap, conds)
            ax.set_xticks(range(len(fams)))
            ax.set_xticklabels([ARTIFACT_LABEL.get(f, f) for f in fams],
                               rotation=30, ha="right", rotation_mode="anchor")
            ax.set_xlim(-0.55, len(fams) - 0.45)
            if c == 0:
                ax.set_ylabel(TASK_LABEL.get(task, task), fontsize=7)
            if r == 0:
                ax.set_title(METRIC_LABEL[metric], fontsize=7, pad=3)
            if stats:  # headroom for significance brackets
                lo, hi = ax.get_ylim()
                ax.set_ylim(lo, hi + 0.10 * (hi - lo))
            ax.grid(axis="y", color="#dddddd", linewidth=0.4, alpha=0.8)
            ax.set_axisbelow(True)

    handles = [plt.matplotlib.patches.Patch(
        facecolor=CONDITION_COLOR[c], alpha=0.6, edgecolor="#333333",
        linewidth=0.6, label=CONDITION_LABEL[c]) for c in conds]
    fig.legend(handles=handles, loc="lower center", ncol=len(conds),
               frameon=False, bbox_to_anchor=(0.5, -0.015))
    fig.tight_layout(rect=(0, 0.035, 1, 1), h_pad=1.4, w_pad=0.9)

    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)) or ".", exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        path = f"{out_prefix}.{ext}"
        fig.savefig(path, bbox_inches="tight", **kw)
        print(f"wrote {path}")
    return fig


def _annotate_stat(ax, fdf, fi, metric, box_w, gap, conds):
    """Paired Wilcoxon cfg1 vs input on volumes present in both conditions."""
    from scipy.stats import wilcoxon
    keys = ["anatomy", "acquisition", "subject", "input", "severity"]
    a = fdf[fdf["condition"] == "input"].set_index(keys)[metric]
    b = fdf[fdf["condition"] == "cfg1"].set_index(keys)[metric]
    common = a.index.intersection(b.index)
    if len(common) < 5:
        return
    av, bv = a.loc[common].values, b.loc[common].values
    if np.allclose(av, bv):
        return
    p = wilcoxon(av, bv).pvalue
    top = max(np.nanmax(av), np.nanmax(bv))
    span = (box_w + gap) * (len(conds) - 1)
    ax.plot([fi - span / 2, fi + span / 2], [top * 1.04] * 2,
            color="#333333", linewidth=0.6)
    ax.text(fi, top * 1.045, _stars(p), ha="center", va="bottom", fontsize=6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csvs", nargs="+", help="per-slice CSV(s) from evaluate_nii_test.py")
    p.add_argument("--out_prefix", default="figures/eval_nii_test")
    p.add_argument("--metrics", nargs="+", default=PLOT_METRICS, choices=METRICS,
                   help="Metric columns to plot (summary CSV keeps all).")
    p.add_argument("--stats", action="store_true",
                   help="paired Wilcoxon cfg1-vs-input stars per group")
    args = p.parse_args()

    df = pd.concat([pd.read_csv(c) for c in args.csvs], ignore_index=True)
    print(f"{len(df)} slice rows from {len(args.csvs)} csv(s)")
    # Older CSVs may still contain the clean fully-sampled input group.
    df = df[df["artifact"] != "clean"]
    agg = aggregate_per_volume(df)
    print(f"{len(agg)} volumes after slice-averaging")

    summ = summary_table(agg)
    summ_path = f"{args.out_prefix}_summary.csv"
    os.makedirs(os.path.dirname(os.path.abspath(summ_path)) or ".", exist_ok=True)
    summ.to_csv(summ_path, index=False)
    print(f"wrote {summ_path}")
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(summ.round(4).to_string(index=False))

    make_figure(agg, args.out_prefix, stats=args.stats, metrics=args.metrics)


if __name__ == "__main__":
    main()
