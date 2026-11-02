# -*- coding: utf-8 -*-
"""
3 - Edge of chaos, Ayahuasca database.

Per subject x condition: edge_of_chaos.MAIN_chaos.analyze_subject on broadband 1-45 Hz and every
band (columns <measure>_<band>): 0-1 chaos test K (+ broadband low-passed at 4, 8, 13 Hz),
lambda_max and covariance width (Dahmen 2019) on amplitude envelopes and event counts,
Lempel-Ziv complexity, Katz FD, Renyi H2, multiscale entropy, entropy production, DFA alpha and
fE/I of the amplitude envelope, aperiodic exponent (broadband only).
Settings: defaults of edge_of_chaos/config_chaos.py, overridden below for 5 s epochs at 250 Hz.

Per analysis (results/<analysis>/3_edge_of_chaos/):
  chaos_results.xlsx             subject values, tests, channel tests, MSE per scale
  figures/                       MSE curves, DFA fluctuation functions, overview grids, topographies,
                                 summary heatmap
  boxplots_per_index/<index>.png one panel per band (broadband first), paired boxplots

    python psychedelics/MAIN_AYA_CHAOS.py
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
from psychedelics import io_aya, runner
from psychedelics import paired_stats as ps
from psychedelics import paired_plots as pp
from psychedelics import compare_drug
from edge_of_chaos import config_chaos as ccfg
from edge_of_chaos import MAIN_chaos
from edge_of_chaos.report_chaos import metric_columns
from edge_of_chaos.plot_chaos import split_band, LABELS, NOTES
from descriptive_analysis.segments import subset_runs
from descriptive_analysis.plots import paper_style, _save, BASE_FONT

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

MODULE = "chaos"

# ---- overrides of config_chaos for this database
ccfg.BANDS = dict(cfg.BANDS)
ccfg.SIGNAL_BANDS = ["broadband", *cfg.BANDS]
# filters / Hilbert on the runs of contiguous epochs (io_aya.contiguous_runs), whole 5 s epochs kept
# (before 2026-11-02: every 5 s epoch filtered alone, trimmed to 3 s and the pieces concatenated)
ccfg.EDGE_TRIM_S = 0.0
ccfg.PAD_S = cfg.PAD_S
ccfg.K_CHUNK_S = 30.0                # K on 30 s chunks of continuous signal (same length everywhere)
ccfg.EQUAL_MODE = cfg.EQUAL_MODE     # covariance width / lambda_max, entropy production on
ccfg.EQUAL_DRAWS = cfg.EQUAL_DRAWS   # io_aya.n_equal epochs (blocks of consecutive epochs)
ccfg.EQUAL_SEED = cfg.EQUAL_SEED
ccfg.K_N_C = 100                     # 100 random c (Gottwald & Melbourne 2009); 1000 took ~7 min/subject
ccfg.MSE_SCALES = list(range(1, 11))  # 5 s epochs at 250 Hz = 1249 samples: scale 10 leaves 124 points
ccfg.OVERWRITE = False
# MSE (~6 s per epoch, the slowest measure) on at most 60 epochs evenly spaced over the recording
# (pre / post / during 27-33 have 40-72 epochs, the 20 min slots 110-240); every other measure
# uses all the epochs
MSE_MAX_EPOCHS = 60

BANDS = cfg.BAND_ORDER
OVERVIEW = [("chaoticity", ["k", "k_lp4", "k_lp8", "k_lp13", "lambda_max_env", "lambda_max_evt"]),
            ("complexity_entropy", ["lzc", "katz_fd", "renyi_h2", "mse_auc", "entropy_production"]),
            ("scale_free", ["dfa", "fei", "aperiodic_exponent", "cov_width_env", "cov_width_evt"])]


def params():
    return dict(ccfg.params(), CONDITIONS=repr(cfg.CONDITIONS), MSE_MAX_EPOCHS=MSE_MAX_EPOCHS)


def label(m):
    return LABELS.get(m, m)


# ================================================================== ONE SUBJECT x CONDITION
def analyze(cond, sid):
    sub = io_aya.load_data(cond, sid)
    measures = list(ccfg.MEASURES)
    n_ep = len(sub["data"])
    if "multiscale_entropy" not in measures or n_ep <= MSE_MAX_EPOCHS:
        return MAIN_chaos.analyze_subject(sub, ccfg, cond, sid)
    try:                                   # everything except MSE on all epochs
        ccfg.MEASURES = [m for m in measures if m != "multiscale_entropy"]
        res = MAIN_chaos.analyze_subject(sub, ccfg, cond, sid)
        keep = np.unique(np.linspace(0, n_ep - 1, MSE_MAX_EPOCHS).round().astype(int))
        sub_mse = dict(sub, data=sub["data"][keep], labels=[sub["labels"][i] for i in keep],
                       runs=subset_runs(sub["runs"], keep))
        ccfg.MEASURES = ["multiscale_entropy"]
        mse = MAIN_chaos.analyze_subject(sub_mse, ccfg, cond, sid)
    finally:
        ccfg.MEASURES = measures
    cols = [k for k in mse["row"] if k.startswith("mse_")]
    res["row"].update({k: mse["row"][k] for k in cols})
    by_ch = {c["channel"]: c for c in mse["chans"]}
    for c in res["chans"]:
        c.update({k: by_ch[c["channel"]][k] for k in cols if k in by_ch[c["channel"]]})
    res["detail"].update({k: v for k, v in mse["detail"].items() if k.startswith("mse_")})
    res["row"]["mse_n_epochs"] = len(keep)
    return res


# ================================================================== TABLES
def long_tables(res):
    subj = pd.DataFrame([dict(x["row"], condition=c) for c, s, x in res]).drop(columns=["group"], errors="ignore")
    chan = pd.DataFrame([dict(r, condition=c) for c, s, x in res for r in x["chans"]]).drop(columns=["group"], errors="ignore")
    bases = []
    for col in metric_columns(subj.drop(columns=["condition"])):
        base, band = split_band(col, BANDS)
        if band and base not in bases and subj[col].replace([np.inf, -np.inf], np.nan).notna().any():
            bases.append(base)
    long = pd.concat([ps.wide_to_long(subj, m, BANDS).set_index(["condition", "subject", "band"]) for m in bases],
                     axis=1).reset_index()
    return subj, chan, long, bases


def index_tables(res):
    _, _, long, bases = long_tables(res)
    return [(long, {m: label(m) for m in bases}, BANDS)]


def tables_for(res, conds):
    subj, chan, long, bases = long_tables(res)
    tests = ps.index_band_tests(long, bases, conds, BANDS)
    summ = ps.summary_table(long, bases, conds, BANDS)
    mse = []
    for c, s, x in res:
        d = x["detail"]
        for key, curve in d.items():
            if key.startswith("mse_curve_"):
                mse += [dict(condition=c, subject=s, band=key[len("mse_curve_"):], scale=int(k), mse=float(v))
                        for k, v in zip(d["mse_scales"], np.nanmean(curve, 0))]
    mse = pd.DataFrame(mse)
    mse_tests = ps.index_band_tests(mse.rename(columns={"scale": "sc"}).assign(
        key=lambda t: t.band + "_" + t.sc.astype(str)), ["mse"], conds,
        sorted(set(mse.band + "_" + mse.scale.astype(str))), band_col="key") if len(mse) else pd.DataFrame()
    return subj, chan, long, bases, tests, summ, mse, mse_tests


# ================================================================== FIGURES
def fig_mse(mse, conds, labels, out, title):
    bands = [b for b in BANDS if b in set(mse.band)]
    fig, axes = plt.subplots(1, len(bands), figsize=(3.6 * len(bands), 4.4), squeeze=False)
    for ax, b in zip(axes[0], bands):
        d = mse[mse.band == b]
        curves = {c: d[d.condition == c].pivot_table(index="subject", columns="scale", values="mse") for c in conds}
        x = sorted(d.scale.unique())
        pp.plot_curves(ax, x, {c: v.reindex(columns=x) for c, v in curves.items()}, conds, cfg.COLORS, labels)
        ax.set(title=b, xlabel="scale factor")
    axes[0, 0].set_ylabel("sample entropy")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title + " — multiscale entropy")
    fig.text(0.5, -0.02, pp.CURVE_NOTE, ha="center", fontsize=BASE_FONT - 4)
    _save(fig, out)


def fig_dfa(res, conds, labels, out, title):
    bands = [b for b in BANDS if any(f"dfa_{b}" in x["detail"] for _, _, x in res)]
    fig, axes = plt.subplots(1, len(bands), figsize=(3.6 * len(bands), 4.4), squeeze=False)
    for ax, b in zip(axes[0], bands):
        curves, w = {c: {} for c in conds}, None
        for c, s, x in res:
            d = x["detail"].get(f"dfa_{b}")
            if d is None or d["fluct"].size == 0:
                continue
            w = d["windows_s"] if w is None else w
            f = np.log10(d["fluct"]).mean(0) if d["fluct"].ndim == 2 else np.log10(d["fluct"])
            if f.size == w.size:
                curves[c][s] = f
        if w is None:
            continue
        lw = np.round(np.log10(w), 3)
        pp.plot_curves(ax, lw, {c: pd.DataFrame(v, index=lw).T for c, v in curves.items()}, conds, cfg.COLORS,
                       labels, test=False)
        ax.set(title=b, xlabel="log$_{10}$ window (s)")
    axes[0, 0].set_ylabel("log$_{10}$ F(n)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title + " — DFA fluctuation function of the amplitude envelope (slope = α)")
    _save(fig, out)


def make_figures(res, conds, subj, chan, long, bases, tests, mse, out, analysis):
    labels, title = cfg.labels_for(analysis), cfg.ANALYSIS_TITLES[analysis]
    fig_dir, box_dir = out / "figures", out / "boxplots_per_index"
    fig_dir.mkdir(exist_ok=True)
    box_dir.mkdir(exist_ok=True)
    for k, (name, items) in enumerate(OVERVIEW):
        pp.plot_index_grid(long, [(m, label(m)) for m in items if m in bases], BANDS, conds, cfg.COLORS, tests,
                           fig_dir / f"figC01{'abc'[k]}_{name}.png", labels, f"{title} — {name.replace('_', ', ')}")
    fig_mse(mse, conds, labels, fig_dir / "figC02_mse_curves.png", title)
    fig_dfa(res, conds, labels, fig_dir / "figC03_dfa_curves.png", title)
    chan_bases = [m for m in bases if any(f"{m}_{b}" in chan for b in BANDS)]
    cols = [c for m in chan_bases for b in BANDS for c in [f"{m}_{b}"] if c in chan and chan[c].notna().any()]
    ctests = ps.channel_tests(chan, cols, conds)
    for m in chan_bases:
        items = [(f"{m}_{b}", b) for b in BANDS if f"{m}_{b}" in chan and chan[f"{m}_{b}"].notna().any()]
        pp.plot_topomaps(chan, items, conds, cfg.COLORS, ctests, fig_dir / f"figC04_topo_{m}.png", labels,
                         f"{title} — topography of {label(m)}")
    pp.plot_summary_heatmap(tests, BANDS, conds, fig_dir / "figC05_summary_heatmap.png", labels,
                            {m: label(m) for m in bases}, title + " — edge-of-chaos indices")
    for m in bases:
        pp.plot_index_bands(long, m, label(m), BANDS, conds, cfg.COLORS, tests, box_dir / f"{m}.png", labels,
                            note=NOTES.get(m))
    return ctests


def main():
    paper_style()
    results = runner.run_all("psychedelics.MAIN_AYA_CHAOS", MODULE, params())
    for analysis, conds in cfg.ANALYSES.items():
        res, subs = runner.analysis_results(results, analysis)
        out = runner.analysis_dir(analysis, MODULE)
        print(f"\n== {analysis}: {len(subs)} subjects -> {out}")
        subj, chan, long, bases, tests, summ, mse, mse_tests = tables_for(res, conds)
        ctests = make_figures(res, conds, subj, chan, long, bases, tests, mse, out, analysis)
        runner.save_tables_excel(dict(tests=tests, summary=summ, subject_metrics=subj, channel_tests=ctests,
                                      channel_metrics=chan, mse_long=mse, mse_scale_tests=mse_tests,
                                      parameters=pd.DataFrame([dict(parameter=k, value=str(v))
                                                               for k, v in params().items()])),
                                 out / "chaos_results.xlsx")
        sig = tests[tests.p_fdr_bands < cfg.ALPHA]
        print(f"   Friedman FDR q < {cfg.ALPHA}: " + (", ".join(f"{r.index}_{r.band}" for r in sig.itertuples()) or "none"))

    for comp in cfg.COMPARISONS:
        res, subs = runner.comparison_results(results, comp)
        print(f"\n== {comp}: {len(subs)} subjects with both ceremonies")
        compare_drug.run(index_tables(res), comp, MODULE, "chaos_aya_vs_placebo.xlsx")


if __name__ == "__main__":
    main()
