# -*- coding: utf-8 -*-
"""
One image per avalanche index, database IPS: reads avalanche_results.pkl (MAIN_IPS_AVALAN.py, no
re-analysis) and writes <results>/per_index/<index>.png, one panel per band (broadband first),
subjects by group, Mann-Whitney p and FDR across the bands of that index
(descriptive_analysis/index_figures.py). The thematic figures A1-A15 stay in plot_ips_avalan.py.

The branching ratio is shown differently, as in figA7: per_index/branching_ratio.png has one row
per estimator (bins, geometric, ancestors), one column per band, sigma vs time bin (sweep 4-35 ms,
plus the band's own avalanche bin as an open diamond), mean +- SEM per group, stars = Mann-Whitney
with FDR over the time bins of that band and estimator.

The Fano factor slope (ff_beta) is shown with its fits:
  per_index/fano_slope.png        per band: FF(W) of every subject with its OLS fit in log-log,
                                  group mean curve with the fit of the mean; below, beta per group
                                  (Mann-Whitney p, FDR across bands)
  per_index/fano_fit_<band>.png   one panel per subject: points, fit line, beta +- SE and R^2
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from photosensitive import config_ips as base
from photosensitive import config_ips_avalan as cfg
from photosensitive.report import load_results
from photosensitive.plot_ips_avalan import find_results
from descriptive_analysis import plots
from descriptive_analysis.index_figures import plot_index_bands, band_tests
from descriptive_analysis.plots import _box_by_group
from descriptive_analysis.plots import _save, _label, _stars
from descriptive_analysis.summary_stats import fdr_bh

INDEX_LABELS = {
    "n_avalanches": "number of avalanches", "avalanche_rate_hz": "avalanche rate (Hz)",
    "mean_size": "mean size S", "mean_duration_bins": "mean duration T (bins)",
    "tau": "τ (size exponent)", "alpha": "α (duration exponent)", "third": "1/σνz (fitted)",
    "dcc": "DCC", "dcc_clauset": "DCC (Clauset xmin)", "tau_range": "τ (power-law range)",
    "alpha_range": "α (power-law range)", "dcc_range": "DCC (power-law range)",
    "tau_rigorous": "τ (rigorous)", "alpha_rigorous": "α (rigorous)", "third_rigorous": "1/σνz (rigorous)",
    "dcc_rigorous": "DCC (rigorous)",
    "size_p_gof": "power-law p (S)", "duration_p_gof": "power-law p (T)",
    "size_decades": "power-law range, decades (S)", "duration_decades": "power-law range, decades (T)",
    "size_xmax_cutoff": "upper cutoff (S)", "duration_xmax_cutoff": "upper cutoff (T)",
    "size_pct_in_range": "% avalanches in power-law range (S)",
    "duration_pct_in_range": "% avalanches in power-law range (T)",
    "size_mse": "log-log MSE (S)", "duration_mse": "log-log MSE (T)",
    "repertoire_rate": "repertoire (unique patterns / s)", "repertoire_per_avalanche": "unique patterns / avalanches",
    "diversity_mean": "pattern diversity (mean Hamming)", "ff": "Fano factor",
}

ESTIMATORS = [("sigma_bins", "σ bins\n(Beggs & Plenz; Maschke)"),
              ("sigma_geometric", "σ geometric\n(Sorrentino 2021)"),
              ("sigma_ancestors", "σ ancestors\n(Beggs & Plenz, 2nd/1st bin)")]


def branching_long(br, metrics):
    """Sweep table (bin_ms 4-35) + the band's own avalanche bin from the metrics table (own_bin = True)."""
    cols = ["group", "subject", "band"] + [c for c, _ in ESTIMATORS]
    sweep = br[cols + ["bin_ms"]].assign(own_bin=False)
    own = metrics[cols + ["time_bin_ms"]].rename(columns={"time_bin_ms": "bin_ms"}).assign(own_bin=True)
    return pd.concat([sweep, own], ignore_index=True)


def bin_tests(long, groups):
    """Mann-Whitney per band, estimator and time bin; p_fdr = FDR over the bins of that band and estimator."""
    rows = []
    for b, db in long.groupby("band", sort=False):
        for est, _ in ESTIMATORS:
            part = []
            for (ms, own), d in db.groupby(["bin_ms", "own_bin"]):
                x = [d.loc[d.group == g, est].dropna().values for g in groups]
                if min(len(v) for v in x) >= 2:
                    part.append(dict(band=b, estimator=est, bin_ms=ms, own_bin=own,
                                     p=stats.mannwhitneyu(*x).pvalue))
            for r, q in zip(part, fdr_bh(np.array([r["p"] for r in part]))):
                r["p_fdr"] = q
            rows += part
    return pd.DataFrame(rows)


def plot_branching(br, metrics, bands, colors, out_png):
    """All three sigma estimators vs time bin, one panel per estimator x band (as figA7)."""
    groups = [g for g in colors if g in set(metrics.group)]
    long = branching_long(br, metrics)
    tests = bin_tests(long, groups) if len(groups) == 2 else pd.DataFrame()
    fig, axes = plt.subplots(len(ESTIMATORS), len(bands), figsize=(3.4 * len(bands), 3.3 * len(ESTIMATORS)),
                             squeeze=False, sharex=True)
    for i, (est, ylab) in enumerate(ESTIMATORS):
        for j, b in enumerate(bands):
            ax = axes[i, j]
            db = long[long.band == b]
            for k, g in enumerate(groups):
                dg = db[db.group == g]
                off = 1 + 0.03 * (k - 0.5)                 # small horizontal offset between groups
                sw = dg[~dg.own_bin].groupby("bin_ms")[est]
                m, se = sw.mean(), sw.std() / np.sqrt(sw.count())
                ax.errorbar(m.index * off, m.values, yerr=se.values, color=colors[g], marker="o", ms=3.5,
                            lw=1.4, capsize=2, label=_label(g))
                ob = dg[dg.own_bin].groupby("bin_ms")[est]
                mo, so = ob.mean(), ob.std() / np.sqrt(ob.count())
                ax.errorbar(mo.index * off, mo.values, yerr=so.values, color=colors[g], marker="D", ms=6,
                            mfc="none", mew=1.5, ls="none", capsize=2,
                            label="band's avalanche bin" if k == 0 else None)
            ax.axhline(1, color="k", ls="--", lw=1)
            own_ms = db.loc[db.own_bin, "bin_ms"].iloc[0] if db.own_bin.any() else None
            if own_ms is not None:
                ax.axvline(own_ms, color="0.6", ls=":", lw=1)
            if len(tests):
                t = tests[(tests.band == b) & (tests.estimator == est)]
                d = db.groupby(["bin_ms", "group"])[est]
                top = (d.mean() + d.std() / np.sqrt(d.count())).groupby("bin_ms").max()
                for ms, q in zip(t.bin_ms, t.p_fdr):
                    star = _stars(q).replace("n.s.", "")
                    if star and ms in top.index:
                        ax.text(ms, top[ms], star, ha="center", va="bottom", fontsize=plots.BASE_FONT - 4)
            ax.set_xscale("log")
            ax.set_xticks([1, 2, 4, 8, 16, 32])
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
            if i == 0:
                ax.set_title(f"{b}\n(avalanche bin {own_ms:.2f} ms)" if own_ms else b)
            if i == len(ESTIMATORS) - 1:
                ax.set_xlabel("time bin (ms)")
            if j == 0:
                ax.set_ylabel(ylab)
    axes[0, 0].legend(fontsize=plots.BASE_FONT - 4)
    fig.suptitle("Branching ratio vs time bin, three estimators (mean ± SEM; open diamond = band's avalanche bin; "
                 "* p_fdr < 0.05, Mann-Whitney, FDR over time bins)")
    _save(fig, out_png)
    return tests


def fano_fit(curve):
    """OLS of log10 FF on log10 W (FF > 0), as fano_factor.fano_curve: dict(w, ff, ok, slope, intercept, se, r2)."""
    w, ff = np.asarray(curve["windows_s"]), np.asarray(curve["ff"])
    ok = np.isfinite(ff) & (ff > 0)
    if ok.sum() < 3:
        return None
    lr = stats.linregress(np.log10(w[ok]), np.log10(ff[ok]))
    return dict(w=w, ff=ff, ok=ok, slope=lr.slope, intercept=lr.intercept, se=lr.stderr, r2=lr.rvalue ** 2)


def _fit_line(ax, f, color, **kw):
    x = np.array([f["w"][f["ok"]].min(), f["w"][f["ok"]].max()])
    ax.plot(1000 * x, 10 ** (f["intercept"] + f["slope"] * np.log10(x)), color=color, **kw)


def plot_fano_slope(details, metrics, bands, colors, out_png):
    """Per band: FF(W) curves + fits (top) and beta per group (bottom)."""
    groups = [g for g in colors if g in set(metrics.group)]
    tests = band_tests(metrics, "ff_beta", bands)
    fig, axes = plt.subplots(2, len(bands), figsize=(3.5 * len(bands), 7.4), squeeze=False)
    for j, b in enumerate(bands):
        ax = axes[0, j]
        for g in groups:
            fits = [fano_fit(det["fano_curve"]) for (gg, sid, bb), det in details.items() if gg == g and bb == b]
            fits = [f for f in fits if f is not None]
            for f in fits:
                ax.plot(1000 * f["w"], f["ff"], "o", color=colors[g], ms=2, alpha=0.35)
                _fit_line(ax, f, colors[g], lw=0.8, alpha=0.45)
            if fits:
                n = min(f["w"].size for f in fits)
                logff = np.nanmean([np.log10(np.where(f["ff"][:n] > 0, f["ff"][:n], np.nan)) for f in fits], 0)
                mean = fano_fit(dict(windows_s=fits[0]["w"][:n], ff=10 ** logff))
                ax.plot(1000 * mean["w"], mean["ff"], "o", color=colors[g], ms=5, mec="k", mew=0.5)
                _fit_line(ax, mean, colors[g], lw=2.6,
                          label=f"{_label(g)}: β = {np.median([f['slope'] for f in fits]):.2f} (median)")
        ax.axhline(1, color="k", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(b)
        ax.set_xlabel("window W (ms)")
        ax.legend(fontsize=plots.BASE_FONT - 5, loc="upper left")
        bx = axes[1, j]
        _box_by_group(bx, metrics[metrics.band == b], "ff_beta", colors, "Fano slope β" if j == 0 else "")
        q = tests.get(b, (np.nan, np.nan))[1]
        bx.set_title(f"{bx.get_title()}\nFDR q = {q:.3f}" if np.isfinite(q) else bx.get_title(),
                     fontsize=plots.BASE_FONT - 1)
    axes[0, 0].set_ylabel("FF(W)")
    fig.suptitle("Fano factor curve FF(W) ~ W^β: thin = each subject (points + OLS fit), thick = fit of "
                 "the group mean (log10 average); β > 0 = correlations at long time scales")
    _save(fig, out_png)


def plot_fano_subjects(details, metrics, band, colors, out_png, ncols=6):
    """One panel per subject in one band: FF(W) points, OLS fit, beta +- SE, R^2."""
    keys = [(g, sid) for g in colors for (gg, sid, bb) in details if gg == g and bb == band]
    if not keys:
        return
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.7 * nrows), squeeze=False,
                             sharex=True, sharey=True)
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    for ax, (g, sid) in zip(axes.ravel(), keys):
        f = fano_fit(details[(g, sid, band)]["fano_curve"])
        if f is None:
            ax.set_title(f"{sid.split('_')[0]}: no fit", fontsize=plots.BASE_FONT - 4)
            continue
        ax.plot(1000 * f["w"][f["ok"]], f["ff"][f["ok"]], "o", color=colors[g], ms=4)
        _fit_line(ax, f, "k", lw=1.2)
        ax.axhline(1, color="0.5", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f"{sid.split('_')[0]}\nβ = {f['slope']:.2f} ± {f['se']:.2f}, R² = {f['r2']:.2f}",
                     fontsize=plots.BASE_FONT - 4, color=colors[g])
    for ax in axes[-1]:
        ax.set_xlabel("W (ms)")
    for ax in axes[:, 0]:
        ax.set_ylabel("FF(W)")
    fig.suptitle(f"Fano factor curves and log-log OLS fits, {band} (title color = group)")
    _save(fig, out_png)


def main():
    channel_set, _, out_root = base.choose_channel_set()      # same choice as MAIN_IPS_AVALAN
    out = find_results(cfg.results_dir(channel_set, out_root))
    if out is None:
        raise SystemExit("avalanche_results.pkl not found: run MAIN_IPS_AVALAN.py first")
    res = load_results(out / "avalanche_results.pkl")
    metrics, bands = res["metrics"], list(res["bands"])
    colors = {g: c for g, c in cfg.COLORS.items() if g in set(metrics.group)}
    plots.paper_style()
    d = out / "per_index"
    d.mkdir(exist_ok=True)
    for col, lab in INDEX_LABELS.items():
        if col in metrics:
            plot_index_bands(metrics, col, lab, bands, colors, d / f"{col}.png",
                             title=f"{lab} per band (Mann-Whitney; FDR across bands)")
    if res.get("br") is not None and len(res["br"]):
        tests = plot_branching(res["br"], metrics, bands, colors, d / "branching_ratio.png")
        if len(tests):
            sig = tests[tests.p_fdr < 0.05]
            print(f"branching ratio: {len(sig)} band x estimator x bin with p_fdr < 0.05")
            if len(sig):
                print(sig.round(4).to_string(index=False))
    if "ff_beta" in metrics:
        plot_fano_slope(res["details"], metrics, bands, colors, d / "fano_slope.png")
        for b in bands:
            plot_fano_subjects(res["details"], metrics, b, colors, d / f"fano_fit_{b}.png")
    # replaced by branching_ratio.png and fano_slope.png
    for old in ("sigma_bins", "sigma_geometric", "sigma_ancestors", "ff_beta"):
        (d / f"{old}.png").unlink(missing_ok=True)
    print("Figures saved in:", d)


if __name__ == "__main__":
    main()
