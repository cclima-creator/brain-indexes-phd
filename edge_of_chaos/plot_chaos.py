# -*- coding: utf-8 -*-
"""
Figures of the edge-of-chaos analysis, from <out>/chaos_results.pkl (no re-analysis):

  chaos_boxplots.png        one panel per measure: subjects by group + Mann-Whitney p
  chaos_distributions.png   distribution of the channel values (all subjects) per group
  chaos_topomaps_<k>.png    group maps + difference map (FDR-marked channels), 4 measures per figure
  chaos_mse_curves.png      MSE vs scale, mean +- SEM per group, significant scales marked
  chaos_dfa_<band>.png      DFA fluctuation functions (log-log), mean per group

    python edge_of_chaos/plot_chaos.py <out>/chaos_results.pkl
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from descriptive_analysis.plots import paper_style, _box_by_group, _grid, _save, _label
from descriptive_analysis.topography import plot_group_topomaps
from edge_of_chaos.io_chaos import load_results
from edge_of_chaos.report_chaos import metric_columns

LABELS = {"lzc": "Lempel-Ziv complexity", "katz_fd": "Katz FD", "renyi_h2": "Rényi H$_2$ (nats)",
          "mse_auc": "MSE (mean over scales)", "aperiodic_exponent": "aperiodic exponent β",
          "lambda_max": "λ$_{max}$ (Dahmen)", "cov_width": "covariance width",
          "entropy_production": "entropy production (nats/step)",
          "entropy_production_per_s": "entropy production (nats/s)"}


def label(m):
    if m in LABELS:
        return LABELS[m]
    for pre, name in (("k_lp", "0-1 test K, low-pass {} Hz"), ("dfa_", "DFA α, {}"), ("fei_", "fE/I, {}")):
        if m.startswith(pre):
            return name.format(m[len(pre):])
    return m


def plot_boxplots(subj, colors, out_png):
    metrics = [m for m in metric_columns(subj) if np.isfinite(subj[m]).any()]   # skip all-NaN
    fig, axes = _grid(len(metrics), ncols=4, h=3.8, w=3.6)
    for ax, m in zip(axes, metrics):
        _box_by_group(ax, subj, m, colors, label(m))
    _save(fig, out_png)


def plot_distributions(chan, colors, out_png, bins=30):
    metrics = [m for m in metric_columns(chan) if np.isfinite(chan[m]).any()]
    fig, axes = _grid(len(metrics), ncols=4, h=3.2, w=3.8)
    for ax, m in zip(axes, metrics):
        x_all = chan[m].replace([np.inf, -np.inf], np.nan).dropna()
        if x_all.empty:
            ax.axis("off")
            continue
        edges = np.histogram_bin_edges(x_all, bins)
        for g, col in colors.items():
            x = chan.loc[chan.group == g, m].replace([np.inf, -np.inf], np.nan).dropna()
            if len(x):
                ax.hist(x, edges, density=True, histtype="stepfilled", alpha=0.35, color=col,
                        label=_label(g))
                ax.axvline(x.median(), color=col, lw=2)
        ax.set_xlabel(label(m))
        ax.set_ylabel("density")
    axes[0].legend()
    _save(fig, out_png)


def plot_topomaps(chan, colors, fig_dir, per_fig=4):
    metrics = metric_columns(chan)
    for k in range(0, len(metrics), per_fig):
        ms = [(m, label(m)) for m in metrics[k:k + per_fig]]
        plot_group_topomaps(chan, ms, colors, fig_dir / f"chaos_topomaps_{k // per_fig + 1}.png",
                            "Edge of chaos")


def plot_mse_curves(mse_long, tests, colors, out_png, alpha=0.05):
    if mse_long is None or mse_long.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for g, col in colors.items():
        d = mse_long[mse_long.group == g].groupby("scale")["mse"]
        if not len(d):
            continue
        m, se = d.mean(), d.sem()
        ax.plot(m.index, m.values, color=col, label=_label(g))
        ax.fill_between(m.index, m - se, m + se, color=col, alpha=0.25, lw=0)
    if tests is not None and len(tests):
        sig = tests.loc[tests.p_fdr < alpha, "scale"]
        y = ax.get_ylim()[1]
        ax.scatter(sig, np.full(len(sig), y), marker="*", color="k", s=40, label=f"FDR q < {alpha}")
    ax.set_xlabel("scale factor")
    ax.set_ylabel("sample entropy")
    ax.legend()
    _save(fig, out_png)


def plot_dfa_curves(details, subj, band, colors, out_png):
    key = f"dfa_{band}"
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for g, col in colors.items():
        curves, w = [], None
        for sid in subj.loc[subj.group == g, "subject"]:
            d = details.get(sid, {}).get(key)
            if d is None or d["fluct"].size == 0:
                continue
            w = d["windows_s"]
            curves.append(np.log10(d["fluct"]).mean(0) if d["fluct"].ndim == 2 else np.log10(d["fluct"]))
        if not curves or w is None:
            continue
        curves = np.array([c for c in curves if c.size == w.size])
        m = curves.mean(0)
        ax.plot(np.log10(w), m, "o-", color=col, ms=3, label=f"{_label(g)} (n={len(curves)})")
        for c in curves:
            ax.plot(np.log10(w), c, color=col, alpha=0.15, lw=0.8)
    ax.set_xlabel("log$_{10}$ window (s)")
    ax.set_ylabel("log$_{10}$ F(n)")
    ax.set_title(f"DFA, {band} envelope")
    ax.legend()
    _save(fig, out_png)


def make_all_figures(pkl_path, fig_dir):
    paper_style()
    res = load_results(pkl_path)
    t, colors, details = res["tables"], res["colors"], res["details"]
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    subj, chan = t["subject_metrics"], t["channel_metrics"]
    plot_boxplots(subj, colors, fig_dir / "chaos_boxplots.png")
    plot_distributions(chan, colors, fig_dir / "chaos_distributions.png")
    if subj.group.nunique() >= 1 and chan.channel.nunique() >= 3:
        plot_topomaps(chan, colors, fig_dir)
    plot_mse_curves(t.get("mse_long"), t.get("mse_scale_tests"), colors, fig_dir / "chaos_mse_curves.png")
    for band in sorted({k[4:] for d in details.values() for k in d if k.startswith("dfa_")}):
        plot_dfa_curves(details, subj, band, colors, fig_dir / f"chaos_dfa_{band}.png")
    print(f"Figures saved in {fig_dir}")


if __name__ == "__main__":
    p = Path(sys.argv[1])
    make_all_figures(p, p.parent / "figures")
