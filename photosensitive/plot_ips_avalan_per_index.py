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

The Fano curve FF(W) bends in log-log (flat at short windows, steeper at long ones), so it is
fitted with two slopes and a break point (avalanche_criticality.fano_factor.fano_two_slopes):
  per_index/fano_slope.png        per band: FF(W) of every subject with its two-slope fit, group
                                  mean curve with its fit and break point; below, beta short,
                                  beta long and break point per group (Mann-Whitney, FDR across bands)
  per_index/fano_fit_<band>.png   one panel per subject: points, single-slope fit (grey dashed),
                                  two-slope fit (black), break point, delta BIC
  fano_two_slopes.xlsx            the fits of every subject x band, group tests and medians
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
from descriptive_analysis.summary_stats import fdr_bh, compare_groups
from photosensitive.report import save_tables_excel
from avalanche_criticality.fano_factor import fano_two_slopes, two_slope_curve

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


def fano_table(details):
    """One row per subject x band: single-slope fit and two-slope fit with break point (fano_two_slopes)."""
    rows = []
    for (g, sid, b), det in details.items():
        c = det["fano_curve"]
        rows.append(dict(group=g, subject=sid, band=b, **fano_two_slopes(c["windows_s"], c["ff"])))
    t = pd.DataFrame(rows)
    t["break_ms"] = 1000 * t.pop("break_s")
    return t


def _two_slope_line(ax, w, fit, color, **kw):
    ws = np.logspace(np.log10(w.min()), np.log10(w.max()), 100)
    ax.plot(1000 * ws, two_slope_curve(fit, ws), color=color, **kw)


def _log_axes(ax):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())


FANO_ROWS = [("beta_short", "β short windows"), ("beta_long", "β long windows"), ("break_ms", "break point (ms)")]


def plot_fano_slope(details, table, bands, colors, out_png):
    """
    Per band: FF(W) of every subject with its two-slope fit (thin) and the group-mean curve with its
    two-slope fit (thick, dotted line = break point); below, beta short, beta long and break point
    per group (Mann-Whitney p, FDR across bands).
    """
    groups = [g for g in colors if g in set(table.group)]
    tests = {c: band_tests(table, c, bands) for c, _ in FANO_ROWS}
    fig, axes = plt.subplots(1 + len(FANO_ROWS), len(bands), figsize=(3.5 * len(bands), 3.4 * (1 + len(FANO_ROWS))),
                             squeeze=False)
    for j, b in enumerate(bands):
        ax = axes[0, j]
        for g in groups:
            curves = [det["fano_curve"] for (gg, sid, bb), det in details.items() if gg == g and bb == b]
            for c in curves:
                f = fano_two_slopes(c["windows_s"], c["ff"])
                ax.plot(1000 * c["windows_s"], c["ff"], "o", color=colors[g], ms=2, alpha=0.35)
                if np.isfinite(f["break_s"]):
                    _two_slope_line(ax, c["windows_s"], f, colors[g], lw=0.7, alpha=0.4)
            if curves:
                n = min(len(c["ff"]) for c in curves)
                w = np.asarray(curves[0]["windows_s"][:n])
                logff = np.nanmean([np.log10(np.where(np.asarray(c["ff"][:n]) > 0, c["ff"][:n], np.nan))
                                    for c in curves], 0)
                fm = fano_two_slopes(w, 10 ** logff)
                ax.plot(1000 * w, 10 ** logff, "o", color=colors[g], ms=5, mec="k", mew=0.5)
                _two_slope_line(ax, w, fm, colors[g], lw=2.6,
                                label=f"{_label(g)}: {fm['beta_short']:.2f} → {fm['beta_long']:.2f}, "
                                      f"break {1000 * fm['break_s']:.0f} ms")
                ax.axvline(1000 * fm["break_s"], color=colors[g], ls=":", lw=1.2)
        ax.axhline(1, color="k", ls="--", lw=1)
        _log_axes(ax)
        ax.set_title(b)
        ax.set_xlabel("window W (ms)")
        ax.legend(fontsize=plots.BASE_FONT - 6, loc="upper left")
        for i, (col, lab) in enumerate(FANO_ROWS, start=1):
            bx = axes[i, j]
            _box_by_group(bx, table[table.band == b], col, colors, lab if j == 0 else "")
            q = tests[col].get(b, (np.nan, np.nan))[1]
            bx.set_title(f"{bx.get_title()}\nFDR q = {q:.3f}" if np.isfinite(q) else bx.get_title(),
                         fontsize=plots.BASE_FONT - 2)
            if col == "break_ms":
                bx.set_yscale("log")
    axes[0, 0].set_ylabel("FF(W)")
    fig.suptitle("Fano factor curve with two regimes: FF(W) ~ W^β_short below the break point, W^β_long above "
                 "(continuous broken line in log-log); thick = fit of the group mean (log10 average)")
    _save(fig, out_png)


def plot_fano_subjects(details, table, band, colors, out_png, ncols=6):
    """One panel per subject: FF(W), single-slope fit (grey dashed), two-slope fit (black), break point."""
    keys = [(g, sid) for g in colors for (gg, sid, bb) in details if gg == g and bb == band]
    if not keys:
        return
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.9 * nrows), squeeze=False,
                             sharex=True, sharey=True)
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    for ax, (g, sid) in zip(axes.ravel(), keys):
        c = details[(g, sid, band)]["fano_curve"]
        one = fano_fit(c)
        two = fano_two_slopes(c["windows_s"], c["ff"])
        if one is None or not np.isfinite(two["break_s"]):
            ax.set_title(f"{sid.split('_')[0]}: no fit", fontsize=plots.BASE_FONT - 4)
            continue
        ax.plot(1000 * one["w"][one["ok"]], one["ff"][one["ok"]], "o", color=colors[g], ms=4)
        _fit_line(ax, one, "0.55", lw=1, ls="--")
        _two_slope_line(ax, one["w"][one["ok"]], two, "k", lw=1.3)
        ax.axvline(1000 * two["break_s"], color="k", ls=":", lw=1)
        ax.axhline(1, color="0.5", ls="--", lw=0.8)
        _log_axes(ax)
        ax.set_title(f"{sid.split('_')[0]}: β {two['beta_short']:.2f} → {two['beta_long']:.2f}\n"
                     f"break {1000 * two['break_s']:.0f} ms, ΔBIC {two['delta_bic']:.0f}",
                     fontsize=plots.BASE_FONT - 4, color=colors[g])
    for ax in axes[-1]:
        ax.set_xlabel("W (ms)")
    for ax in axes[:, 0]:
        ax.set_ylabel("FF(W)")
    fig.suptitle(f"Fano factor curves, {band}: one slope (grey dashed) vs two slopes with break point (black); "
                 "ΔBIC > 6 strongly favours two regimes")
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
        table = fano_table(res["details"])
        plot_fano_slope(res["details"], table, bands, colors, d / "fano_slope.png")
        for b in bands:
            plot_fano_subjects(res["details"], table, b, colors, d / f"fano_fit_{b}.png")
        cols = ["beta_one", "beta_short", "beta_long", "break_ms", "delta_bic"]
        tests = pd.concat([compare_groups(table[table.band == b], cols).assign(band=b) for b in bands],
                          ignore_index=True)
        summary = table.groupby(["band", "group"], sort=False)[cols + ["r2_one", "r2_two"]].median().reset_index()
        save_tables_excel({"fano_two_slopes": table, "group_tests": tests, "group_medians": summary},
                          out / "fano_two_slopes.xlsx")
        print("\n=== Fano two slopes: group medians ===")
        print(summary.round(3).to_string(index=False))
        print("\n=== Fano two slopes: Mann-Whitney (p_fdr across the 5 columns of each band) ===")
        print(tests[["band", "metric", "p", "p_fdr"]].round(4).to_string(index=False))
    # replaced by branching_ratio.png and fano_slope.png
    for old in ("sigma_bins", "sigma_geometric", "sigma_ancestors", "ff_beta"):
        (d / f"{old}.png").unlink(missing_ok=True)
    print("Figures saved in:", d)


if __name__ == "__main__":
    main()
