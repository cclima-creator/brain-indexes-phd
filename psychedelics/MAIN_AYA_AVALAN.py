# -*- coding: utf-8 -*-
"""
1 - Neuronal avalanches, Ayahuasca database.

Per subject x condition and band (broadband 1-40 Hz, delta ... gamma), settings of config_aya_avalan.py:
  band-pass on every run of contiguous epochs -> z-score per channel -> events at the peak of every
  excursion beyond ±1.9 SD -> avalanches (time bin 4 ms, delta 8 ms) -> rate, size, duration
  -> branching ratio (+ 4-32 ms sweep) -> repertoire and pattern diversity -> Fano factor
Number of avalanches, repertoire and pattern diversity depend on the amount of data: computed on the
same number of epochs in every condition of a subject (io_aya.n_equal; *_alldata = all epochs).
(power-law exponents, goodness of fit and DCC are in power_law_goodness/, set aside)

Per analysis (results/<analysis>/1_avalanches/):
  avalanche_results.xlsx         subject x band values, Friedman + Wilcoxon tests, summary
  figures/                       threshold divergence, mean CCDF, branching ratio vs time bin, Fano
                                 curves, overview grids, summary heatmap
  boxplots_per_index/<index>.png one panel per band (broadband first), paired boxplots

    python psychedelics/MAIN_AYA_AVALAN.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from psychedelics import config_aya as cfg
from psychedelics import config_aya_avalan as acfg
from psychedelics import io_aya, runner
from psychedelics import paired_stats as ps
from psychedelics import paired_plots as pp
from psychedelics import compare_drug
from photosensitive import MAIN_IPS_AVALAN as ipsa
from photosensitive import plot_ips_avalan as pa
from descriptive_analysis.plots import paper_style, _save, BASE_FONT, amplitude_deviation_from_gaussian
from descriptive_analysis.segments import equal_subsets
from avalanche_criticality.binarization import divergence_threshold

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

# the IPS functions read their settings from a module-level cfg: point them to this database
acfg.COLORS = cfg.COLORS
ipsa.cfg = acfg
pa.cfg = acfg

MODULE = "avalanches"
BANDS = list(acfg.BANDS)
# Power-law goodness of fit, exponents (slopes tau, alpha, 1/sigma nu z, Fano slope) and DCC are
# left out of every analysis (decision of 30/10/2026): only rates, sizes, branching, repertoire and
# Fano factor are reported. n_avalanches is not tested (it follows the duration; the rate does not).
INDEX_LABELS = {
    "avalanche_rate_hz": "avalanche rate (1/s)",
    "mean_size": "mean size (events)", "mean_duration_bins": "mean duration (bins)",
    "sigma_bins": "branching ratio σ (Beggs & Plenz)", "sigma_geometric": "σ geometric (Sorrentino)",
    "sigma_ancestors": "σ first generation", "repertoire_rate": "repertoire (unique patterns / s)",
    "repertoire_per_avalanche": "repertoire per avalanche", "diversity_mean": "pattern diversity (Hamming)",
    "ff": "Fano factor"}
OVERVIEW = [("branching_repertoire_fano", [("sigma_bins", "σ"), ("sigma_geometric", "σ geometric"),
                                           ("repertoire_rate", "repertoire /s"), ("diversity_mean", "diversity"),
                                           ("ff", "Fano factor")]),
            ("rate_size_duration", [("avalanche_rate_hz", "rate (1/s)"), ("mean_size", "mean size"),
                                    ("mean_duration_bins", "mean duration")])]


def params():
    return acfg.params()


# ================================================================== ONE SUBJECT x CONDITION
def analyze(cond, sid):
    d = io_aya.load_data(cond, sid)
    subsets = equal_subsets(len(d["data"]), d["n_equal"], cfg.EQUAL_DRAWS, cfg.EQUAL_MODE, cfg.EQUAL_SEED)
    rows, details = [], {}
    for b, band in acfg.BANDS.items():
        row, det = ipsa.analyze_band(d["data"], d["sfreq"], cond, sid, b, band, d["runs"], subsets, cfg.PAD_S)
        det = {k: v for k, v in det.items() if k not in ("unique_patterns", "pattern_counts")}
        rows.append(dict(row, condition=cond, n_runs=len(d["runs"])))
        details[b] = det
    print(f"    {cond} {sid}: " + " | ".join(f"{r['band']} n={r['n_avalanches']} sigma={r['sigma_bins']:.2f}"
                                             for r in rows), flush=True)
    return dict(rows=rows, details=details, ch_names=d["ch_names"], sfreq=d["sfreq"])


# ================================================================== TABLES
def long_table(res):
    df = pd.DataFrame([r for _, _, x in res for r in x["rows"]])
    df["condition"] = [c for c, _, x in res for _ in x["rows"]]
    return df.drop(columns=["group"], errors="ignore")


def index_tables(res):
    return [(long_table(res), INDEX_LABELS, BANDS)]


def tables_for(res, conds):
    df = long_table(res)
    indices = [m for m in INDEX_LABELS if m in df]
    tests = ps.index_band_tests(df, indices, conds, BANDS)
    summ = ps.summary_table(df, indices, conds, BANDS)
    br = pd.DataFrame([dict(condition=c, subject=s, band=b, **r)
                       for c, s, x in res for b, det in x["details"].items() for r in det["br_sweep"]])
    return df, tests, summ, br


def br_bin_tests(br, conds):
    """Friedman per band, estimator and time bin; FDR across the time bins of that band and estimator."""
    out = []
    for b in BANDS:
        for est in ("sigma_bins", "sigma_geometric"):
            d = br[br.band == b]
            t = ps.index_band_tests(d.rename(columns={"bin_ms": "bin"}), [est], conds,
                                    sorted(d.bin_ms.unique()), band_col="bin")
            if len(t):
                t = t.rename(columns={"band": "bin_ms"}).assign(band=b, estimator=est)
                out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ================================================================== FIGURES
def fig_divergence(res, conds, labels, out):
    curves, rows = {}, []
    for c in conds:
        for b in BANDS:
            store = [dict(group=c, amp=dict(hist_z=x["details"][b]["zhist"],
                                            stats=dict(excess_kurtosis=np.nan, ks_D_vs_gauss=np.nan)))
                     for c2, s, x in res if c2 == c]
            r = amplitude_deviation_from_gaussian(store, c)
            if r is None:
                continue
            thr = divergence_threshold(r["centers"], r["heavier"], acfg.DIVERGENCE_MIN_SD)[0]
            curves[(c, b)] = {k: r[k] for k in ("centers", "density", "lo", "hi", "gauss")} | dict(divergence=thr)
            rows.append(dict(condition=c, band=b, divergence_sd=thr, threshold_used_sd=acfg.THRESHOLD_SD))
    pa.plot_threshold_divergence(curves, None, BANDS, {c: cfg.COLORS[c] for c in conds}, out)
    return pd.DataFrame(rows)


def details_by_cond(res):
    return {(c, s, b): det for c, s, x in res for b, det in x["details"].items()}


def fig_branching(br, conds, labels, out, title):
    fig, axes = plt.subplots(2, len(BANDS), figsize=(3.4 * len(BANDS), 7.0), squeeze=False, sharex=True)
    for j, b in enumerate(BANDS):
        for i, est in enumerate(("sigma_bins", "sigma_geometric")):
            ax = axes[i, j]
            d = br[br.band == b]
            curves = {c: d[d.condition == c].pivot_table(index="subject", columns="bin_ms", values=est)
                      for c in conds}
            x = sorted(d.bin_ms.unique())
            pp.plot_curves(ax, x, {c: v.reindex(columns=x) for c, v in curves.items()}, conds, cfg.COLORS, labels)
            ax.axhline(1, color="k", ls="--", lw=1)
            if i == 0:
                ax.set_title(b)
            else:
                ax.set_xlabel("time bin (ms)")
    axes[0, 0].set_ylabel("σ (Beggs & Plenz)")
    axes[1, 0].set_ylabel("σ geometric (Sorrentino)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title + " — branching ratio vs time bin")
    fig.text(0.5, -0.01, pp.CURVE_NOTE, ha="center", fontsize=BASE_FONT - 4)
    _save(fig, out)


def fig_fano(res, conds, labels, out, title):
    det = details_by_cond(res)
    fig, axes = plt.subplots(1, len(BANDS), figsize=(3.4 * len(BANDS), 4.2), squeeze=False)
    for j, b in enumerate(BANDS):
        ax = axes[0, j]
        curves = {}
        for c in conds:
            rows = {s: pd.Series(d["fano_curve"]["ff"], index=np.round(1000 * d["fano_curve"]["windows_s"], 1))
                    for (c2, s, bb), d in det.items() if c2 == c and bb == b and len(d["fano_curve"]["windows_s"])}
            curves[c] = pd.DataFrame(rows).T
        x = curves[conds[0]].columns.values if len(curves[conds[0]]) else []
        if len(x):
            pp.plot_curves(ax, x, {c: v.reindex(columns=x) for c, v in curves.items()}, conds, cfg.COLORS,
                           labels, logx=True, logy=True)
        ax.axhline(1, color="k", ls="--", lw=1)
        ax.set(title=b, xlabel="window W (ms)")
    axes[0, 0].set_ylabel("FF(W)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title + " — Fano factor of the population activity vs window (FF=1: Poisson)")
    fig.text(0.5, -0.02, pp.CURVE_NOTE, ha="center", fontsize=BASE_FONT - 4)
    _save(fig, out)


def fig_ccdf_mean(res, conds, labels, out, title):
    """Mean CCDF per condition on a common log grid (each subject weighs the same)."""
    det = details_by_cond(res)
    fig, axes = plt.subplots(2, len(BANDS), figsize=(3.4 * len(BANDS), 7.0), squeeze=False)
    for j, b in enumerate(BANDS):
        for i, var in enumerate(("sizes", "durations")):
            ax = axes[i, j]
            allx = [d[var] for (c, s, bb), d in det.items() if bb == b and len(d[var])]
            if not allx:
                continue
            grid = np.unique(np.round(np.geomspace(1, max(x.max() for x in allx), 40)))
            curves = {}
            for c in conds:
                rows = {}
                for (c2, s, bb), d in det.items():
                    if c2 == c and bb == b and len(d[var]):
                        xs = np.sort(d[var])
                        rows[s] = 1 - np.searchsorted(xs, grid, "left") / xs.size
                curves[c] = pd.DataFrame(rows, index=grid).T.replace(0, np.nan)
            pp.plot_curves(ax, grid, curves, conds, cfg.COLORS, labels, logx=True, logy=True, test=False)
            ax.set_xlabel("size S (events)" if i == 0 else "duration T (bins)")
            if i == 0:
                ax.set_title(b)
    axes[0, 0].set_ylabel("P(S ≥ s)")
    axes[1, 0].set_ylabel("P(T ≥ t)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title + " — mean CCDF of avalanche size and duration (± SEM)")
    _save(fig, out)


def make_figures(res, conds, df, tests, br, out, analysis):
    labels, title = cfg.labels_for(analysis), cfg.ANALYSIS_TITLES[analysis]
    fig_dir, box_dir = out / "figures", out / "boxplots_per_index"
    fig_dir.mkdir(exist_ok=True)
    box_dir.mkdir(exist_ok=True)
    colors = {c: cfg.COLORS[c] for c in conds}
    det = details_by_cond(res)
    div = fig_divergence(res, conds, labels, fig_dir / "figA01_threshold_divergence.png")
    fig_ccdf_mean(res, conds, labels, fig_dir / "figA02_ccdf_mean.png", title)
    for k, (name, items) in enumerate(OVERVIEW):
        pp.plot_index_grid(df, items, BANDS, conds, cfg.COLORS, tests, fig_dir / f"figA03{'abcd'[k]}_{name}.png",
                           labels, f"{title} — {name.replace('_', ', ')}")
    fig_branching(br, conds, labels, fig_dir / "figA04_branching_ratio_bins.png", title)
    fig_fano(res, conds, labels, fig_dir / "figA05_fano_curves.png", title)
    pp.plot_summary_heatmap(tests, BANDS, conds, fig_dir / "figA06_summary_heatmap.png", labels, INDEX_LABELS,
                            title + " — avalanche indices")
    for m, lab in INDEX_LABELS.items():
        pp.plot_index_bands(df, m, lab, BANDS, conds, cfg.COLORS, tests, box_dir / f"{m}.png", labels)
    return div


def main():
    paper_style()
    results = runner.run_all("psychedelics.MAIN_AYA_AVALAN", MODULE, params())
    for analysis, conds in cfg.ANALYSES.items():
        res, subs = runner.analysis_results(results, analysis)
        out = runner.analysis_dir(analysis, MODULE)
        print(f"\n== {analysis}: {len(subs)} subjects -> {out}")
        df, tests, summ, br = tables_for(res, conds)
        brt = br_bin_tests(br, conds)
        div = make_figures(res, conds, df, tests, br, out, analysis)
        runner.save_tables_excel(dict(tests=tests, summary=summ, subject_band_metrics=df, branching_ratio_bins=br,
                                      branching_ratio_bin_tests=brt, threshold_divergence=div,
                                      parameters=pd.DataFrame([dict(parameter=k, value=str(v))
                                                               for k, v in params().items()])),
                                 out / "avalanche_results.xlsx")
        sig = tests[tests.p_fdr_bands < cfg.ALPHA]
        print(f"   Friedman FDR q < {cfg.ALPHA}: " + (", ".join(f"{r.index}_{r.band}" for r in sig.itertuples()) or "none"))

    for comp in cfg.COMPARISONS:
        res, subs = runner.comparison_results(results, comp)
        print(f"\n== {comp}: {len(subs)} subjects with both ceremonies")
        compare_drug.run(index_tables(res), comp, MODULE, "avalanche_aya_vs_placebo.xlsx")


if __name__ == "__main__":
    main()
