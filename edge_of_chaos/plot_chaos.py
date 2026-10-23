# -*- coding: utf-8 -*-
"""
Figures of the edge-of-chaos analysis, from <out>/chaos_results.pkl (no re-analysis).
Columns are <measure>_<band> (MAIN_chaos.analyze_subject); one image per measure:

  per_index/<measure>.png        one panel per band (broadband first): subjects by group,
                                 Mann-Whitney p and FDR across the bands of that measure
  per_index/topo_<measure>.png   group maps + difference map per band (channel measures only)
  chaos_mse_curves.png           MSE vs scale, mean +- SEM per group, one panel per band
  chaos_dfa_<band>.png           DFA fluctuation functions (log-log), mean per group

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

from descriptive_analysis.plots import paper_style, _save, _label, BASE_FONT
from descriptive_analysis.topography import plot_group_topomaps
from descriptive_analysis.index_figures import plot_index_bands, wide_to_long
from edge_of_chaos.io_chaos import load_results
from edge_of_chaos.report_chaos import metric_columns

BAND_ORDER = ["broadband", "delta", "theta", "alpha", "beta", "gamma"]
LABELS = {"lzc": "Lempel-Ziv complexity", "katz_fd": "Katz FD", "renyi_h2": "Rényi H$_2$ (nats)",
          "mse_auc": "MSE (mean over scales)", "aperiodic_exponent": "aperiodic exponent β",
          "lambda_max": "λ$_{max}$ (edgeofpy, band signal)", "cov_width": "covariance width (edgeofpy, band signal)",
          "lambda_max_env": "λ$_{max}$ (Dahmen), amplitude envelope",
          "cov_width_env": "covariance width, amplitude envelope",
          "lambda_max_evt": "λ$_{max}$ (Dahmen), event counts",
          "cov_width_evt": "covariance width, event counts",
          "entropy_production": "entropy production (nats/step)",
          "entropy_production_per_s": "entropy production (nats/s)",
          "k": "0-1 test K", "dfa": "DFA α (amplitude envelope)", "fei": "fE/I",
          "k_lp4": "0-1 test K, low-pass 4 Hz", "k_lp8": "0-1 test K, low-pass 8 Hz",
          "k_lp13": "0-1 test K, low-pass 13 Hz"}
NOTES = {"aperiodic_exponent": "Fit of the whole 2-40 Hz spectrum: one value, not defined per band.",
         "cov_width": "Integrated covariance ~0 on band-passed signals: width not reliable (see README).",
         "lambda_max": "Computed from the covariance width: not reliable on band-passed signals.",
         "cov_width_env": "w = SD(C_ij) / mean(C_ii), C = covariance of 1 s sums of the envelope.",
         "lambda_max_env": "From the envelope covariance width; N = number of channels.",
         "cov_width_evt": "w = SD(C_ij) / mean(C_ii), C = covariance of event counts in 1 s windows.",
         "lambda_max_evt": "From the event-count covariance width; N = number of channels.",
         "k": "K of the band-passed signal; narrow bands are quasi-periodic, so K is lower there.",
         "fei": "fE/I is NaN where DFA α <= 0.6 (no long-range temporal correlations)."}


def split_band(col, bands=BAND_ORDER):
    """'lzc_alpha' -> ('lzc', 'alpha'); columns without a band suffix -> (col, None)."""
    for b in bands:
        if col.endswith(f"_{b}"):
            return col[:-len(b) - 1], b
    return col, None


def label(m):
    base, band = split_band(m)
    return LABELS.get(base, base) + (f", {band}" if band else "")


def measures(df):
    """Base measure names present as <measure>_<band> columns, in first-seen order."""
    out = []
    for c in metric_columns(df):
        base, band = split_band(c)
        if band and base not in out and df[c].replace([np.inf, -np.inf], np.nan).notna().any():
            out.append(base)
    return out


def plot_per_index(subj, chan, colors, fig_dir):
    """per_index/<measure>.png for every measure, per_index/topo_<measure>.png for channel measures."""
    d = Path(fig_dir) / "per_index"
    d.mkdir(parents=True, exist_ok=True)
    for m in measures(subj):
        long = wide_to_long(subj, m, BAND_ORDER)
        plot_index_bands(long, m, LABELS.get(m, m), BAND_ORDER, colors, d / f"{m}.png",
                         title=f"{LABELS.get(m, m)} per band (Mann-Whitney; FDR across bands)",
                         note=NOTES.get(m))
    if chan.channel.nunique() >= 3:
        for m in measures(chan):
            cols = [(f"{m}_{b}", f"{b}\n{LABELS.get(m, m)}") for b in BAND_ORDER if f"{m}_{b}" in chan
                    and chan[f"{m}_{b}"].notna().any()]
            if cols:
                plot_group_topomaps(chan, cols, colors, d / f"topo_{m}.png", f"{LABELS.get(m, m)} per band")


def plot_mse_curves(mse_long, tests, colors, out_png, alpha=0.05):
    """MSE vs scale per band, mean +- SEM per group; * = scales with p_fdr < alpha (FDR within band)."""
    if mse_long is None or mse_long.empty:
        return
    if "band" not in mse_long:
        mse_long = mse_long.assign(band="broadband")
    bands = [b for b in BAND_ORDER if b in set(mse_long.band)]
    fig, axes = plt.subplots(1, len(bands), figsize=(4.2 * len(bands), 4.2), squeeze=False)
    for ax, b in zip(axes[0], bands):
        mb = mse_long[mse_long.band == b]
        for g, col in colors.items():
            d = mb[mb.group == g].groupby("scale")["mse"]
            if not len(d):
                continue
            m, se = d.mean(), d.sem()
            ax.plot(m.index, m.values, color=col, label=_label(g))
            ax.fill_between(m.index, m - se, m + se, color=col, alpha=0.25, lw=0)
        if tests is not None and len(tests) and "band" in tests:
            sig = tests.loc[(tests.band == b) & (tests.p_fdr < alpha), "scale"]
            y = ax.get_ylim()[1]
            ax.scatter(sig, np.full(len(sig), y), marker="*", color="k", s=40, label=f"FDR q < {alpha}")
        ax.set_title(b)
        ax.set_xlabel("scale factor")
    axes[0, 0].set_ylabel("sample entropy")
    axes[0, 0].legend(fontsize=BASE_FONT - 3)
    fig.suptitle("Multiscale entropy per band")
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
    plot_per_index(subj, chan, colors, fig_dir)
    plot_mse_curves(t.get("mse_long"), t.get("mse_scale_tests"), colors, fig_dir / "chaos_mse_curves.png")
    for band in BAND_ORDER:
        if any(f"dfa_{band}" in d for d in details.values()):
            plot_dfa_curves(details, subj, band, colors, fig_dir / f"chaos_dfa_{band}.png")
    print(f"Figures saved in {fig_dir}")


if __name__ == "__main__":
    p = Path(sys.argv[1])
    make_all_figures(p, p.parent / "figures")
