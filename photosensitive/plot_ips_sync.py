# -*- coding: utf-8 -*-
"""
Synchronization figures, IPS database: reads synchrony_results.pkl produced by MAIN_IPS_SYNC.py
(no re-analysis). Figures in config_ips_sync.results_dir (syncron_results, next to criticality_results; the
occipital ROI in syncron_results/occipital).

  figS1_kuramoto_R             R(t) example per group (median-GS subject) and distribution of R per group
  figS2_gs_metastability       GS, M, M pooled, GS and M normalized by surrogates, per band and group
  figS3_pc_plv                 mean PC, PLV and PPC over channel pairs, per band and group
  figS4_matrix_<pc|ppc|wpli|dwpli>   group-mean matrices and difference, pairs significant after FDR marked
  figS5_topo_<pc|plv|ppc|wpli|dwpli|icoh>   node strength (mean coupling with the other channels)
  figS6_stim_freq              GS, M, PC and wPLI vs stimulation frequency, median and IQR per group
  figS7_plv_time               PLV(t) across epochs (mean over pairs) from the stimulation onset
  figS8_summary_heatmap        group medians and group difference (rank-biserial r), metric x band
  figS9_lag_insensitive        mean PLI, wPLI, debiased wPLI, imaginary coherence per band and group

Interpretation: with channels rebuilt from 5 ICA components, Kuramoto GS / M, PC and PLV mostly
reflect the instantaneous mixing (and the polarity of the topographies for R): they are read as
a relative comparison between groups. PLI, wPLI, dwPLI and iCOH keep only the lagged coupling.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from photosensitive import config_ips as base
from photosensitive import config_ips_sync as cfg
from photosensitive.report import FALLBACK_DIR, load_results
from photosensitive.plot_ips_avalan import _band_boxes
from descriptive_analysis import plots, topography
from descriptive_analysis.plots import _save, _label, _stars

MEASURE_LABELS = {"pc": "phase coherence", "plv": "PLV", "ppc": "PPC", "pli": "PLI", "wpli": "wPLI",
                  "dwpli": "debiased wPLI", "icoh": "imaginary coherence"}


def _groups(df):
    return [g for g in cfg.COLORS if g in set(df.group)]


# ------------------------------------------------------------------ FIGURES
def plot_kuramoto_R(details, metrics, bands, colors, out_png):
    """figS1: top, R(t) of the subject with median GS in each group; bottom, mean density of R ± SEM."""
    fig, axes = plt.subplots(2, len(bands), figsize=(3.6 * len(bands), 6.4), squeeze=False)
    for j, b in enumerate(bands):
        d = metrics[metrics.band == b]
        ax, axh = axes[0, j], axes[1, j]
        for g, c in colors.items():
            dg = d[d.group == g].dropna(subset=["gs"])
            if not len(dg):
                continue
            sid = dg.iloc[(dg.gs - dg.gs.median()).abs().argmin()].subject
            det = details[(g, sid, b)]
            t = np.arange(det["R_example"].size) / det["R_example_sfreq"]
            ax.plot(t, det["R_example"], color=c, lw=0.8, alpha=0.85, label=_label(g))
            dens = np.array([details[(g, s, b)]["R_hist"]["density"] for s in dg.subject])
            edges = details[(g, sid, b)]["R_hist"]["edges"]
            ctr = (edges[:-1] + edges[1:]) / 2
            mu, se = dens.mean(0), dens.std(0, ddof=1) / np.sqrt(len(dens)) if len(dens) > 1 else 0
            axh.fill_between(ctr, mu - se, mu + se, color=c, alpha=0.25, lw=0)
            axh.plot(ctr, mu, color=c, label=_label(g))
        rnd = d.gs_random.median()
        axh.axvline(rnd, color="k", ls="--", lw=1, label="random phases")
        ax.set_ylim(0, 1)
        ax.set_title(b)
        ax.set_xlabel("time (s)")
        axh.set_xlabel("R")
        if j == 0:
            ax.set_ylabel("R(t)")
            axh.set_ylabel("density")
            axh.legend(fontsize=plots.BASE_FONT - 3)
    fig.suptitle("Kuramoto order parameter R(t) (eq. 3.15); sensor level: relative comparison between groups")
    _save(fig, out_png)


def plot_matrices(mats, edges, ch_names, bands, colors, measure, out_png):
    """figS4: group-mean matrix of each group and difference; o = p_fdr < ALPHA, · = p < ALPHA."""
    groups = [g for g in colors if (g, bands[0], measure) in mats]
    if len(groups) != 2 or ch_names is None:
        return
    n = len(ch_names)
    fig, axes = plt.subplots(len(bands), 3, figsize=(15, 4.6 * len(bands)), squeeze=False)
    for r, b in enumerate(bands):
        m = [mats[(g, b, measure)].copy() for g in groups]
        for x in m:
            np.fill_diagonal(x, np.nan)
        vals = np.concatenate([x[np.isfinite(x)] for x in m])
        vlim = (np.percentile(vals, 2), np.percentile(vals, 98))
        for c, (g, x) in enumerate(zip(groups, m)):
            im = axes[r, c].imshow(x, cmap="viridis", vmin=vlim[0], vmax=vlim[1])
            axes[r, c].set_title(f"{_label(g)} — {b}", color=colors[g])
            fig.colorbar(im, ax=axes[r, c], fraction=0.046)
        diff = m[0] - m[1]
        lim = np.nanmax(np.abs(diff)) or 1
        ax = axes[r, 2]
        im = ax.imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim)
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f"{_label(groups[0])} − {_label(groups[1])}")
        e = edges[(edges.band == b) & (edges.measure == measure)] if len(edges) else edges
        if len(e):
            for sel, kw in ((e.p_fdr < cfg.ALPHA, dict(marker="o", mfc="none", mec="k", ms=6)),
                            ((e.p < cfg.ALPHA) & (e.p_fdr >= cfg.ALPHA), dict(marker=".", color="k", ms=4))):
                s = e[sel]
                ax.plot(np.r_[s.j, s.i], np.r_[s.i, s.j], ls="none", **kw)
        for ax in axes[r]:
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(ch_names, rotation=90, fontsize=8)
            ax.set_yticklabels(ch_names, fontsize=8)
    fig.suptitle(f"{MEASURE_LABELS[measure]} between channel pairs "
                 f"(o: p_fdr < {cfg.ALPHA}, ·: p < {cfg.ALPHA} uncorrected, Mann-Whitney)")
    _save(fig, out_png)


def plot_stim_freq(epochs, sf_tests, bands, colors, out_png):
    """figS6: per-subject mean at each stim frequency, group median and IQR; * = p_fdr < ALPHA."""
    cols = [("gs", "GS"), ("metastability", "metastability"), ("pc_mean", "mean PC"),
            ("wpli_mean", "mean wPLI")]
    cols = [(c, l) for c, l in cols if c in epochs]
    subj = epochs.dropna(subset=["stim_freq"]).groupby(["group", "subject", "band", "stim_freq"])[
        [c for c, _ in cols]].mean().reset_index()
    fig, axes = plt.subplots(len(cols), len(bands), figsize=(3.8 * len(bands), 3.2 * len(cols)),
                             squeeze=False, sharex=True)
    for i, (col, lab) in enumerate(cols):
        for j, b in enumerate(bands):
            ax = axes[i, j]
            d = subj[subj.band == b]
            for g, c in colors.items():
                q = d[d.group == g].groupby("stim_freq")[col].quantile([.25, .5, .75]).unstack()
                if not len(q):
                    continue
                ax.fill_between(q.index, q[.25], q[.75], color=c, alpha=0.2, lw=0)
                ax.plot(q.index, q[.5], "o-", color=c, ms=3, label=_label(g))
            if len(sf_tests):
                t = sf_tests[(sf_tests.band == b) & (sf_tests.metric == col) & (sf_tests.p_fdr < cfg.ALPHA)]
                top = ax.get_ylim()[1]
                for f in t.stim_freq:
                    ax.text(f, top, "*", ha="center", va="top", fontsize=plots.BASE_FONT + 2)
            if i == 0:
                ax.set_title(b)
            if j == 0:
                ax.set_ylabel(lab)
            if i == len(cols) - 1:
                ax.set_xlabel("stimulation frequency (Hz)")
    axes[0, 0].legend(fontsize=plots.BASE_FONT - 3)
    fig.suptitle(f"Synchrony vs photic stimulation frequency (median, IQR; * p_fdr < {cfg.ALPHA})")
    _save(fig, out_png)


def plot_plv_time(details, metrics, bands, colors, out_png):
    """figS7: PLV(t) across epochs, mean over channel pairs, group mean ± SEM per band."""
    fig, axes = plt.subplots(1, len(bands), figsize=(3.8 * len(bands), 3.4), squeeze=False, sharey=True)
    for j, b in enumerate(bands):
        ax = axes[0, j]
        for g, c in colors.items():
            subs = metrics[(metrics.band == b) & (metrics.group == g)].subject
            dets = [details[(g, s, b)] for s in subs if "plv_t" in details[(g, s, b)]]
            if not dets:
                continue
            n = min(d["plv_t"].size for d in dets)
            x = np.array([d["plv_t"][:n] for d in dets])
            t = dets[0]["plv_t_times"][:n]
            mu = x.mean(0)
            se = x.std(0, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0 * mu
            ax.fill_between(t, mu - se, mu + se, color=c, alpha=0.25, lw=0)
            ax.plot(t, mu, color=c, lw=1, label=_label(g))
        ax.set_title(b)
        ax.set_xlabel("time from epoch onset (s)")
    axes[0, 0].set_ylabel("PLV(t), mean over pairs")
    axes[0, 0].legend(fontsize=plots.BASE_FONT - 3)
    fig.suptitle("Phase locking across epochs (Lachaux et al. 1999); biased by the number of epochs")
    _save(fig, out_png)


def plot_summary_heatmap(metrics, bands, colors, out_png):
    """figS8: group medians (z-scored per row) and group difference (rank-biserial r, Mann-Whitney)."""
    cols = [("gs", "GS"), ("metastability", "metastability"), ("metastability_pooled", "M pooled"),
            ("gs_norm", "GS / surrogate"), ("metastability_norm", "M / surrogate"),
            ("pc_mean", "mean PC"), ("plv_mean", "mean PLV"), ("ppc_mean", "mean PPC"),
            ("pli_mean", "mean PLI"), ("wpli_mean", "mean wPLI"), ("dwpli_mean", "mean dwPLI"),
            ("icoh_mean", "mean iCOH")]
    cols = [(c, l) for c, l in cols if c in metrics and metrics[c].notna().any()]
    groups = _groups(metrics)
    if len(groups) != 2:
        return
    med = {g: np.array([[metrics[(metrics.band == b) & (metrics.group == g)][c].median()
                         for b in bands] for c, _ in cols]) for g in groups}
    eff = np.full((len(cols), len(bands)), np.nan)
    ann = np.full((len(cols), len(bands)), "", object)
    for i, (c, _) in enumerate(cols):
        for j, b in enumerate(bands):
            d = metrics[metrics.band == b]
            a, z = (d.loc[d.group == g, c].dropna() for g in groups)
            if len(a) >= 2 and len(z) >= 2:
                u = stats.mannwhitneyu(a, z)
                eff[i, j] = 2 * u.statistic / (len(a) * len(z)) - 1
                ann[i, j] = _stars(u.pvalue).replace("n.s.", "")
    fig, axes = plt.subplots(1, 3, figsize=(6 + 3.2 * len(bands), 0.6 * len(cols) + 2.5))
    allv = np.hstack([med[g] for g in groups])
    mu, sd = np.nanmean(allv, 1, keepdims=True), np.nanstd(allv, 1, keepdims=True)
    for ax, g in zip(axes[:2], groups):
        ax.imshow((med[g] - mu) / np.where(sd > 0, sd, 1), cmap="viridis", vmin=-1.5, vmax=1.5, aspect="auto")
        for i in range(len(cols)):
            for j in range(len(bands)):
                ax.text(j, i, f"{med[g][i, j]:.2g}", ha="center", va="center", fontsize=9, color="w")
        ax.set_title(f"{_label(g)} (median)", color=colors[g])
    im = axes[2].imshow(eff, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i in range(len(cols)):
        for j in range(len(bands)):
            axes[2].text(j, i, f"{eff[i, j]:.2f}{ann[i, j]}", ha="center", va="center", fontsize=9)
    axes[2].set_title(f"{_label(groups[0])} vs {_label(groups[1])}\n(rank-biserial r, MWU)")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.set_xticks(range(len(bands)))
        ax.set_xticklabels(bands, rotation=45)
        ax.set_yticks(range(len(cols)))
        ax.set_yticklabels([l for _, l in cols] if ax is axes[0] else [])
    fig.suptitle("Synchrony metrics: group medians and group difference")
    _save(fig, out_png)


# ------------------------------------------------------------------ MAIN
def find_results(out_dir, name="synchrony_results.pkl"):
    for d in (out_dir, FALLBACK_DIR / cfg.OUT_SUBDIR):
        if (Path(d) / name).exists():
            return Path(d)
    return None


def main():
    channel_set, _, out_root = base.choose_channel_set()   # same choice as MAIN_IPS_SYNC
    out = find_results(cfg.results_dir(channel_set, out_root))
    if out is None:
        raise SystemExit("synchrony_results.pkl not found: run MAIN_IPS_SYNC.py first")
    print("Results read from:", out)
    res = load_results(out / "synchrony_results.pkl")
    metrics, details = res["metrics"], res["details"]
    bands = list(res["bands"])
    colors = {g: c for g, c in cfg.COLORS.items() if g in set(metrics.group)}
    plots.paper_style()

    plot_kuramoto_R(details, metrics, bands, colors, out / "figS1_kuramoto_R.png")
    _band_boxes(metrics, [("gs", "GS (mean R)"), ("metastability", "metastability (std R)"),
                          ("metastability_pooled", "M pooled over epochs"),
                          ("gs_norm", "GS / surrogate GS"), ("metastability_norm", "M / surrogate M")],
                bands, colors, out / "figS2_gs_metastability.png",
                "Global synchrony and metastability (Kuramoto; Acebrón et al. 2005) — "
                "relative comparison, sensitive to zero-lag mixing")
    _band_boxes(metrics, [("pc_mean", "phase coherence\n(mean over pairs)"),
                          ("plv_mean", "PLV across epochs"), ("ppc_mean", "PPC (unbiased PLV²)")],
                bands, colors, out / "figS3_pc_plv.png",
                "Pair-wise phase synchrony (Mormann et al. 2000; Lachaux et al. 1999; Vinck et al. 2010)")
    _band_boxes(metrics, [("pli_mean", "PLI"), ("wpli_mean", "wPLI"), ("dwpli_mean", "debiased wPLI"),
                          ("icoh_mean", "imaginary coherence")],
                bands, colors, out / "figS9_lag_insensitive.png",
                "Lag-insensitive synchrony (Nolte et al. 2004; Stam et al. 2007; Vinck et al. 2011)")
    for m in ("pc", "ppc", "wpli", "dwpli"):
        plot_matrices(res["matrices"], res["edges"], res["ch_names"], bands, colors, m,
                      out / f"figS4_matrix_{m}.png")
    chan = res["chan"]
    for m in ("pc", "plv", "ppc", "wpli", "dwpli", "icoh"):
        cols = [(f"{m}_strength_{b}", f"{b}\n{MEASURE_LABELS[m]}") for b in bands
                if f"{m}_strength_{b}" in chan]
        if cols:
            topography.plot_group_topomaps(chan, cols, colors, out / f"figS5_topo_{m}.png",
                                           f"Node strength: mean {MEASURE_LABELS[m]} with the other channels",
                                           alpha=cfg.ALPHA)
    plot_stim_freq(res["epochs"], res["stim_freq_tests"], bands, colors, out / "figS6_stim_freq.png")
    plot_plv_time(details, metrics, bands, colors, out / "figS7_plv_time.png")
    plot_summary_heatmap(metrics, bands, colors, out / "figS8_summary_heatmap.png")
    print("\nFigures saved in:", out)


if __name__ == "__main__":
    main()
