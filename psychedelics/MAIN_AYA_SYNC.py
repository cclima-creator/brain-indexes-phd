# -*- coding: utf-8 -*-
"""
2 - Synchronization, Ayahuasca database.

Per subject x condition and band (broadband 1-45 Hz, delta ... gamma) the IPS pipeline
photosensitive.MAIN_IPS_SYNC.analyze_band with config_aya_sync.py, "new method" (2026-11-02):
  Butterworth band-pass + Hilbert on every run of contiguous epochs (whole 5 s epochs kept)
  -> Kuramoto R(t): global synchrony GS, metastability M = SD of R(t) over the analysed recording
  -> phase coherence PC, PLV / PPC across epochs (consecutive resting epochs, not stimulus-locked:
     an estimate of stationary locking, like PC)
  -> lag-insensitive coupling PLI, wPLI, debiased wPLI, iCOH from multitaper cross-spectra across epochs
M, PLV / PPC and the lag-insensitive indices on io_aya.n_equal epochs (blocks of consecutive epochs).
Results in <analysis>/2_synchrony/new_method/ (NOTE_new_method.txt explains what changed).
Sensor-level GS / M / PC / PLV are inflated by volume conduction (zero-lag mixing): they are read
as relative changes between conditions of the same subjects; PLI / wPLI / dwPLI / iCOH keep only
the lagged coupling.

Per analysis (results/<analysis>/2_synchrony/):
  synchrony_results.xlsx         subject x band values, tests, node strength, edge-wise tests
  figures/                       R(t) and R distribution, overview grids, connectivity matrices with
                                 pairwise differences, node-strength topographies, summary heatmap
  boxplots_per_index/<index>.png one panel per band (broadband first), paired boxplots

    python psychedelics/MAIN_AYA_SYNC.py
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
from psychedelics import config_aya_sync as scfg
from psychedelics import io_aya, runner
from psychedelics import paired_stats as ps
from psychedelics import paired_plots as pp
from psychedelics import compare_drug
from photosensitive import MAIN_IPS_SYNC as ipss
from photosensitive import plot_ips_sync as psync
from descriptive_analysis.plots import paper_style, _save, BASE_FONT
from descriptive_analysis.summary_stats import fdr_bh
from descriptive_analysis.segments import equal_subsets
from synchrony_criticality.phase import drop_flat_channels
from synchrony_criticality.phase_coherence import upper

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

scfg.COLORS = cfg.COLORS
ipss.cfg = scfg
psync.cfg = scfg

MODULE = "synchrony"
BANDS = list(scfg.BANDS)
INDEX_LABELS = {"gs": "global synchrony (mean R)", "metastability": "metastability (SD of R)",
                "metastability_epoch": "SD of R within 5 s epochs", "metastability_cv": "metastability CV",
                "gs_norm": "GS / surrogate GS", "metastability_norm": "M / surrogate M",
                "R_median": "median R", "R_q95": "95th percentile of R", "pc_mean": "phase coherence (mean)",
                "pc_median": "phase coherence (median)", "plv_mean": "PLV (across epochs)",
                "ppc_mean": "PPC (unbiased PLV²)", "pli_mean": "PLI", "wpli_mean": "wPLI",
                "dwpli_mean": "debiased wPLI", "icoh_mean": "imaginary coherence |iCOH|"}
OVERVIEW = [("kuramoto", [("gs", "GS"), ("metastability", "M"), ("gs_norm", "GS/surr"), ("metastability_norm", "M/surr")]),
            ("zero_lag_coupling", [("pc_mean", "PC"), ("plv_mean", "PLV"), ("ppc_mean", "PPC")]),
            ("lag_insensitive", [("pli_mean", "PLI"), ("wpli_mean", "wPLI"), ("dwpli_mean", "dwPLI"),
                                 ("icoh_mean", "|iCOH|")])]
MATRIX_MEASURES = ["pc", "ppc", "wpli", "dwpli"]
STRENGTH = ["pc", "ppc", "wpli", "dwpli", "icoh"]


def params():
    return scfg.params()


# ================================================================== ONE SUBJECT x CONDITION
def analyze(cond, sid):
    d = io_aya.load_data(cond, sid)
    data = d["data"]
    keep = drop_flat_channels(data)
    data, ch = data[:, keep], [d["ch_names"][i] for i in keep]
    subsets = equal_subsets(len(data), d["n_equal"], cfg.EQUAL_DRAWS, cfg.EQUAL_MODE, cfg.EQUAL_SEED)
    lag = ipss.lag_insensitive(data, d["sfreq"], scfg.BANDS, subsets, scfg.EDGE_TRIM_S, scfg.MT_NW)
    rows, chans, details = [], [], {}
    nanf = np.full(len(data), np.nan)
    for b, band in scfg.BANDS.items():
        row, _, c, det = ipss.analyze_band(data, d["sfreq"], ch, d["labels"], nanf, cond, sid, b, band,
                                           d["runs"], subsets, lag[b], cfg.PAD_S, scfg.EDGE_TRIM_S)
        rows.append(dict(row, condition=cond))
        chans += [dict(x, condition=cond) for x in c]
        details[b] = det
    return dict(rows=rows, chans=chans, details=details, ch_names=ch, sfreq=d["sfreq"])


# ================================================================== TABLES
def long_table(res):
    # condition from the analysis, not from the cache (a virtual condition reuses a real one)
    return pd.DataFrame([dict(r, condition=c) for c, _, x in res for r in x["rows"]]).drop(columns=["group"],
                                                                                         errors="ignore")


def index_tables(res):
    return [(long_table(res), INDEX_LABELS, BANDS)]


def tables_for(res, conds):
    df = long_table(res)
    indices = [m for m in INDEX_LABELS if m in df]
    tests = ps.index_band_tests(df, indices, conds, BANDS)
    summ = ps.summary_table(df, indices, conds, BANDS)
    chan = pd.DataFrame([dict(r, condition=c) for c, _, x in res for r in x["chans"]]).drop(columns=["group"],
                                                                                          errors="ignore")
    cols = [f"{m}_strength" for m in STRENGTH if f"{m}_strength" in chan]
    wide = chan.pivot_table(index=["condition", "subject", "channel"], columns="band", values=cols, sort=False)
    wide.columns = [f"{m.replace('_strength', '')}_strength_{b}" for m, b in wide.columns]
    wide = wide.reset_index()
    return df, tests, summ, wide


def edge_tests(res, conds, ch_names):
    """Wilcoxon per channel pair for every pair of conditions; FDR across channel pairs per band x measure x pair."""
    subs = sorted({s for _, s, _ in res})
    by = {(c, s): x for c, s, x in res}
    iu, ju = np.triu_indices(len(ch_names), 1)
    rows, mats = [], {}
    for b in BANDS:
        for m in MATRIX_MEASURES:
            X = {c: np.array([upper(by[(c, s)]["details"][b][m]) for s in subs]) for c in conds}
            for c in conds:
                full = np.zeros((len(ch_names), len(ch_names)))
                full[iu, ju] = X[c].mean(0)
                mats[(c, b, m)] = full + full.T
            fr = np.array([ps.friedman_row(pd.DataFrame(np.stack([X[c][:, k] for c in conds], 1)))["p"]
                           for k in range(len(iu))])
            t = pd.DataFrame(dict(band=b, measure=m, ch_i=np.array(ch_names)[iu], ch_j=np.array(ch_names)[ju],
                                  i=iu, j=ju, p_friedman=fr, q_friedman=fdr_bh(fr)))
            for a, c in ps.pairs_of(conds):
                from scipy import stats
                p = stats.wilcoxon(X[c] - X[a], axis=0).pvalue
                t[f"diff_{ps.pair_name(a, c)}"] = np.median(X[c] - X[a], 0)
                t[f"p_{ps.pair_name(a, c)}"] = p
                t[f"q_{ps.pair_name(a, c)}"] = fdr_bh(p)
            rows.append(t)
    return pd.concat(rows, ignore_index=True), mats


def align(res):
    """Same channel order in every subject x condition (they are the same montage here)."""
    names = res[0][2]["ch_names"]
    for _, _, x in res:
        if x["ch_names"] != names:
            idx = [x["ch_names"].index(n) for n in names]
            for det in x["details"].values():
                for m in ipss.MATRIX_MEASURES:
                    if m in det:
                        det[m] = det[m][np.ix_(idx, idx)]
            x["ch_names"] = names
    return names


# ================================================================== FIGURES
def fig_matrices(mats, edges, ch, conds, labels, measure, out, title, alpha=0.05):
    pairs = ps.pairs_of(conds)
    ncol = len(conds) + len(pairs)
    fig, axes = plt.subplots(len(BANDS), ncol, figsize=(3.4 * ncol, 3.3 * len(BANDS)), squeeze=False)
    n = len(ch)
    for r, b in enumerate(BANDS):
        M = {c: mats[(c, b, measure)].copy() for c in conds}
        for x in M.values():
            np.fill_diagonal(x, np.nan)
        vals = np.concatenate([x[np.isfinite(x)] for x in M.values()])
        vlim = np.percentile(vals, [2, 98])
        for k, c in enumerate(conds):
            im = axes[r, k].imshow(M[c], cmap="viridis", vmin=vlim[0], vmax=vlim[1])
            axes[r, k].set_title(f"{labels[c].replace(chr(10), ' ')} — {b}", color=cfg.COLORS[c], fontsize=9)
            if k == len(conds) - 1:
                fig.colorbar(im, ax=axes[r, k], fraction=0.046)
        e = edges[(edges.band == b) & (edges.measure == measure)]
        for k, (a, c) in enumerate(pairs):
            ax = axes[r, len(conds) + k]
            diff = M[c] - M[a]
            lim = np.nanmax(np.abs(diff)) or 1
            im = ax.imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim)
            fig.colorbar(im, ax=ax, fraction=0.046)
            q, p = e[f"q_{ps.pair_name(a, c)}"], e[f"p_{ps.pair_name(a, c)}"]
            for sel, kw in ((q < alpha, dict(marker="o", mfc="none", mec="k", ms=6)),
                            ((p < alpha) & (q >= alpha), dict(marker=".", color="k", ms=4))):
                s = e[sel]
                ax.plot(np.r_[s.j, s.i], np.r_[s.i, s.j], ls="none", **kw)
            ax.set_title(f"{labels[c]} − {labels[a]}".replace("\n", " "), fontsize=9)
        for ax in axes[r]:
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(ch, rotation=90, fontsize=6)
            ax.set_yticklabels(ch, fontsize=6)
    fig.suptitle(f"{title} — {psync.MEASURE_LABELS[measure]} between channel pairs")
    fig.text(0.5, -0.005, f"Difference panels: o = q < {alpha} (Wilcoxon signed-rank per pair of channels, FDR across "
             f"channel pairs); · = p < {alpha} uncorrected.", ha="center", fontsize=BASE_FONT - 4)
    _save(fig, out)


def make_figures(res, conds, df, tests, wide, edges, mats, ch, out, analysis):
    labels, title = cfg.labels_for(analysis), cfg.ANALYSIS_TITLES[analysis]
    fig_dir, box_dir = out / "figures", out / "boxplots_per_index"
    fig_dir.mkdir(exist_ok=True)
    box_dir.mkdir(exist_ok=True)
    details = {(c, s, b): det for c, s, x in res for b, det in x["details"].items()}
    psync.plot_kuramoto_R(details, df.assign(group=df.condition), BANDS, {c: cfg.COLORS[c] for c in conds},
                          fig_dir / "figS01_kuramoto_R.png")
    for k, (name, items) in enumerate(OVERVIEW):
        pp.plot_index_grid(df, items, BANDS, conds, cfg.COLORS, tests, fig_dir / f"figS02{'abc'[k]}_{name}.png",
                           labels, f"{title} — {name.replace('_', ' ')}")
    for k, m in enumerate(MATRIX_MEASURES):
        fig_matrices(mats, edges, ch, conds, labels, m, fig_dir / f"figS03{'abcd'[k]}_matrix_{m}.png", title)
    ctests = ps.channel_tests(wide, [c for c in wide.columns if "_strength_" in c], conds)
    for m in STRENGTH:
        cols = [(f"{m}_strength_{b}", f"{b}") for b in BANDS if f"{m}_strength_{b}" in wide]
        pp.plot_topomaps(wide, cols, conds, cfg.COLORS, ctests, fig_dir / f"figS04_topo_strength_{m}.png", labels,
                         f"{title} — node strength of {psync.MEASURE_LABELS[m]}")
    pp.plot_summary_heatmap(tests, BANDS, conds, fig_dir / "figS05_summary_heatmap.png", labels, INDEX_LABELS,
                            title + " — synchrony indices")
    for m, lab in INDEX_LABELS.items():
        pp.plot_index_bands(df, m, lab, BANDS, conds, cfg.COLORS, tests, box_dir / f"{m}.png", labels)
    return ctests


def main():
    paper_style()
    results = runner.run_all("psychedelics.MAIN_AYA_SYNC", MODULE, params())
    for analysis, conds in cfg.ANALYSES.items():
        res, subs = runner.analysis_results(results, analysis)
        out = runner.analysis_dir(analysis, MODULE) / scfg.NEW_METHOD_SUBDIR
        out.mkdir(exist_ok=True)
        print(f"\n== {analysis}: {len(subs)} subjects -> {out}")
        ch = align(res)
        df, tests, summ, wide = tables_for(res, conds)
        edges, mats = edge_tests(res, conds, ch)
        ctests = make_figures(res, conds, df, tests, wide, edges, mats, ch, out, analysis)
        runner.save_tables_excel(dict(tests=tests, summary=summ, subject_band_metrics=df, channel_strength=wide,
                                      channel_tests=ctests, edge_tests=edges,
                                      parameters=pd.DataFrame([dict(parameter=k, value=str(v))
                                                               for k, v in params().items()])),
                                 out / "synchrony_results.xlsx")
        sig = tests[tests.p_fdr_bands < cfg.ALPHA]
        print(f"   Friedman FDR q < {cfg.ALPHA}: " + (", ".join(f"{r.index}_{r.band}" for r in sig.itertuples()) or "none"))

    for comp in cfg.COMPARISONS:
        res, subs = runner.comparison_results(results, comp)
        print(f"\n== {comp}: {len(subs)} subjects with both ceremonies")
        compare_drug.run(index_tables(res), comp, MODULE, "synchrony_aya_vs_placebo.xlsx", scfg.NEW_METHOD_SUBDIR)


if __name__ == "__main__":
    main()
