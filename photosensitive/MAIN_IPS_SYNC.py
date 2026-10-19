# -*- coding: utf-8 -*-
"""
MAIN synchronization, IPS database: Kuramoto order parameter R(t), global synchrony (GS),
metastability (M), phase coherence (PC) and phase locking value (PLV / PPC) per band.
Independent from MAIN_IPS.py and MAIN_IPS_AVALAN.py: it reads the same .set files.
Figures are made afterwards with plot_ips_sync.py (no re-analysis).

For each subject the results are saved in <results>/subjects/<file>.pkl: when re-running, the
subjects already analysed with the same parameters (config_ips_sync.params) are only re-read.
To redo everything: OVERWRITE = True in config_ips_sync.py.

Pipeline per subject and band (modules in synchrony_criticality/):
  Butterworth band-pass (zero phase) -> Hilbert -> phase phi_j(t) -> 1 s edge trimming
  -> R(t) per epoch -> GS, M (+ circular-shift surrogates)
  -> PC_ij per epoch (time average) -> PLV_ij / PPC_ij across epochs (Lachaux 1999; Vinck 2010)
  -> PLI, wPLI, debiased wPLI, imaginary coherence per epoch (only the lagged part of the coupling:
     not inflated by zero-lag mixing of the 5 back-projected components / volume conduction)
Kuramoto GS and M at sensor level depend on the mixing and on the polarity of the topographies
(channels in anti-phase cancel in the mean): read them as a relative comparison between groups,
not as absolute levels of synchrony.
Group level: Mann-Whitney per band (FDR within band), per stimulation frequency (FDR across
frequencies), per channel (node strength, FDR across channels) and per channel pair (FDR across pairs).
"""
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# repository root (brain-indexes-phd/) on the path, as in MAIN_IPS.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from photosensitive import config_ips as base
from photosensitive import config_ips_sync as cfg
from photosensitive.io_eeg import discover_files, load_epochs, get_data_uv, epoch_labels
from photosensitive.report import check_output_dir, save_tables_excel, save_results, load_results, FALLBACK_DIR
from photosensitive.MAIN_IPS import pick_channels
from descriptive_analysis.summary_stats import compare_groups, fdr_bh, channel_group_tests
from descriptive_analysis.topography import ALIASES
from synchrony_criticality.phase import band_analytic, drop_flat_channels
from synchrony_criticality.kuramoto import kuramoto_epochs
from synchrony_criticality.phase_coherence import phase_coherence_epochs, upper, node_strength
from synchrony_criticality.plv import plv_summary
from synchrony_criticality.lag_insensitive import lag_insensitive_epochs

warnings.filterwarnings("ignore")

OUTPUT_NAMES = ["synchrony_results.xlsx", "synchrony_results.pkl"]
SHEET_ORDER = ["subject_band_metrics", "group_tests", "group_summary", "stim_freq_tests",
               "epoch_metrics", "channel_tests", "channel_strength", "edge_tests", "parameters"]

# columns of the subject x band table compared between groups
TEST_METRICS = ["gs", "metastability", "metastability_pooled", "metastability_cv", "gs_norm",
                "metastability_norm", "R_median", "R_q95", "pc_mean", "pc_median", "plv_mean",
                "ppc_mean", "pli_mean", "wpli_mean", "dwpli_mean", "icoh_mean"]
EPOCH_METRICS = ["gs", "metastability", "pc_mean", "wpli_mean"]   # per epoch, tested per stim frequency
LAG_MEASURES = ["pli", "wpli", "dwpli", "icoh"]               # lag-insensitive indices
MATRIX_MEASURES = ["pc", "plv", "ppc"] + LAG_MEASURES         # pair-wise matrices kept per subject
EDGE_MEASURES = ["pc", "ppc", "wpli", "dwpli"]                # pair-wise matrices tested per edge
STRENGTH_MEASURES = ["pc", "plv", "ppc", "wpli", "dwpli", "icoh"]   # node strength per channel


# ================================================================== ONE SUBJECT
def analyze_band(data, sfreq, ch_names, labels, fstim, group, sid, band_name, band):
    """All synchrony measures for one subject in one band. Returns (row, epoch rows, channel rows, detail)."""
    z = band_analytic(data, sfreq, band, cfg.FILTER_ORDER, cfg.EDGE_TRIM_S)
    phases = np.angle(z)
    n_ep, n_ch, n_t = phases.shape

    kur, kur_ep, R, hist = kuramoto_epochs(phases, cfg.N_SURROGATES, cfg.SEED, cfg.R_HIST_BINS)
    pc_ep = phase_coherence_epochs(phases)                    # (n_ep, n_ch, n_ch)
    pc = pc_ep.mean(0)
    pl = plv_summary(phases) if n_ep >= 2 else None
    lag, lag_ep = lag_insensitive_epochs(z)
    del z

    row = dict(group=group, subject=sid, band=band_name, n_channels=n_ch, n_epochs=n_ep,
               duration_s=n_ep * n_t / sfreq, **kur,
               pc_mean=float(upper(pc).mean()), pc_median=float(np.median(upper(pc))),
               plv_mean=float(upper(pl["plv"]).mean()) if pl else np.nan,
               ppc_mean=float(upper(pl["ppc"]).mean()) if pl else np.nan,
               **{f"{m}_mean": float(upper(lag[m]).mean()) for m in LAG_MEASURES})

    pc_ep_mean = upper(pc_ep).mean(-1)
    epochs = [dict(group=group, subject=sid, band=band_name, epoch=k, label=labels[k],
                   stim_freq=fstim[k], gs=kur_ep["gs"][k], metastability=kur_ep["metastability"][k],
                   pc_mean=pc_ep_mean[k], wpli_mean=lag_ep["wpli"][k]) for k in range(n_ep)]

    strength = dict(pc=node_strength(pc), **{m: node_strength(lag[m]) for m in ("wpli", "dwpli", "icoh")})
    if pl:
        strength.update(plv=node_strength(pl["plv"]), ppc=node_strength(pl["ppc"]))
    chans = [dict(group=group, subject=sid, band=band_name, channel=c,
                  **{f"{m}_strength": float(v[i]) for m, v in strength.items()})
             for i, c in enumerate(ch_names)]

    dec = cfg.DECIMATE_PLOT
    n_ex = int(cfg.R_EXAMPLE_S * sfreq)
    detail = dict(pc=pc.astype(np.float32), R_hist=hist, R_example=R[0, :n_ex:dec].astype(np.float32),
                  R_example_sfreq=sfreq / dec, R_epoch_gs=kur_ep["gs"], R_epoch_ms=kur_ep["metastability"],
                  **{m: lag[m].astype(np.float32) for m in LAG_MEASURES})
    if pl:
        detail.update(plv=pl["plv"].astype(np.float32), ppc=pl["ppc"].astype(np.float32),
                      plv_t=pl["plv_t"][::dec].astype(np.float32),
                      plv_t_times=(cfg.EDGE_TRIM_S + np.arange(n_t)[::dec] / sfreq).astype(np.float32))
    return row, epochs, chans, detail


def analyze_subject(fpath, group, channels):
    """Loads one .set and analyses all bands. Returns dict(rows, epochs, chans, details, ch_names, params)."""
    ep = load_epochs(fpath, base.FIXED_LENGTH_S, base.ANALYSIS_WINDOW)
    if channels is not None:
        ep = pick_channels(ep, channels, base.OCCIPITAL_CHANNELS)
    sfreq = ep.info["sfreq"]
    data = get_data_uv(ep)
    keep = drop_flat_channels(data)
    if keep.size < data.shape[1]:
        print(f"    flat channels removed: {[c for i, c in enumerate(ep.ch_names) if i not in keep]}")
    data, ch_names = data[:, keep], [ep.ch_names[i] for i in keep]
    labels = epoch_labels(ep)
    fstim = np.array([base.STIM_FREQ_MAP.get(l, base.parse_stim_freq(l)) for l in labels], float)
    print(f"    {data.shape[0]} epochs x {data.shape[1]} channels x {data.shape[2] / sfreq:.1f} s, "
          f"sfreq {sfreq:g} Hz")
    rows, epochs, chans, details = [], [], [], {}
    for band_name, band in cfg.BANDS.items():
        t0 = time.time()
        row, e, c, det = analyze_band(data, sfreq, ch_names, labels, fstim, group, fpath.stem,
                                      band_name, band)
        rows.append(row)
        epochs += e
        chans += c
        details[band_name] = det
        print(f"    {band_name:>6}: GS {row['gs']:.3f} (surr {row.get('gs_surrogate', np.nan):.3f}) | "
              f"M {row['metastability']:.3f} | PC {row['pc_mean']:.3f} | PLV {row['plv_mean']:.3f} "
              f"PPC {row['ppc_mean']:.3f} | wPLI {row['wpli_mean']:.3f} dwPLI {row['dwpli_mean']:.3f}  "
              f"({time.time() - t0:.1f} s)")
    return dict(rows=rows, epochs=epochs, chans=chans, details=details, ch_names=ch_names,
                params=cfg.params(), sfreq=sfreq)


def load_or_run(fpath, group, channels, cache_dir):
    """Re-reads the subject cache if it has the same parameters, otherwise analyses and saves."""
    cache = cache_dir / f"{fpath.stem}.pkl"
    if cache.exists() and not cfg.OVERWRITE:
        res = load_results(cache)
        if res.get("params") == cfg.params():
            print("    (already analysed: read from cache)")
            return res
        print("    parameters changed: re-analysing")
    res = analyze_subject(fpath, group, channels)
    with open(cache, "wb") as f:
        pickle.dump(res, f)
    return res


# ================================================================== GROUP TABLES
def _groups(df):
    return [g for g in cfg.GROUPS if g in set(df.group)]


def group_tables(df):
    """Between-group tests band by band (Mann-Whitney, FDR within each band) and group statistics."""
    tests, summ = [], []
    metrics = [m for m in TEST_METRICS if m in df.columns]
    for b in cfg.BANDS:
        d = df[df.band == b]
        t = compare_groups(d, metrics)
        if len(t):
            t.insert(0, "band", b)
            tests.append(t)
        s = d.groupby("group")[metrics].agg(["median", "mean", "std", "count"])
        s.columns = [f"{m}_{a}" for m, a in s.columns]
        summ.append(s.reset_index().assign(band=b))
    tests = pd.concat(tests, ignore_index=True) if tests else pd.DataFrame()
    return tests, pd.concat(summ, ignore_index=True) if summ else pd.DataFrame()


def stim_freq_tests(ep):
    """
    GS, M and PC per stimulation frequency: each subject's epochs at the same frequency are averaged,
    then Mann-Whitney between groups; p_fdr = FDR across frequencies within band and metric.
    """
    groups = _groups(ep)
    if len(groups) != 2:
        return pd.DataFrame()
    subj = ep.dropna(subset=["stim_freq"]).groupby(["group", "subject", "band", "stim_freq"])[EPOCH_METRICS] \
        .mean().reset_index()
    rows = []
    for b in cfg.BANDS:
        for m in EPOCH_METRICS:
            part = []
            for f, d in subj[subj.band == b].groupby("stim_freq"):
                x = [d.loc[d.group == g, m].dropna().values for g in groups]
                if min(len(v) for v in x) < cfg.MIN_SUBJECTS_PER_FREQ:
                    continue
                part.append(dict(band=b, metric=m, stim_freq=f, p=stats.mannwhitneyu(*x).pvalue,
                                 **{f"n_{g}": len(v) for g, v in zip(groups, x)},
                                 **{f"median_{g}": float(np.median(v)) for g, v in zip(groups, x)}))
            if part:
                for r, q in zip(part, fdr_bh(np.array([r["p"] for r in part]))):
                    r["p_fdr"] = q
                rows += part
    return pd.DataFrame(rows)


def channel_tables(chan):
    """Node strength in wide format (one column per measure x band) and tests channel by channel."""
    cols = [f"{m}_strength" for m in STRENGTH_MEASURES if f"{m}_strength" in chan]
    wide = chan.pivot_table(index=["group", "subject", "channel"], columns="band", values=cols,
                            sort=False)
    wide.columns = [f"{m}_{b}" for m, b in wide.columns]
    wide = wide.reset_index()
    metrics = [c for c in wide.columns if c not in ("group", "subject", "channel")]
    tests = channel_group_tests(wide, metrics) if len(_groups(wide)) == 2 else pd.DataFrame()
    return wide, tests


def _canonical(name):
    """10-20 name -> 10-10 name for the same electrode (T3 = T7, T4 = T8, T5 = P7, T6 = P8)."""
    up = {k.upper(): v for k, v in ALIASES.items()}
    return up.get(name.upper(), name)


def align_channels(subjects):
    """
    Puts every subject's matrices in the same channel order, matching channels by name with the old
    and new 10-20 names unified. Returns the common channel names, or None if the sets differ
    (then group matrices and edge tests are skipped).
    """
    canon = [[_canonical(c) for c in s["ch_names"]] for s in subjects]
    if len({frozenset(c) for c in canon}) != 1:
        return None
    names = canon[0]
    for s, c in zip(subjects, canon):
        if c == names:
            continue
        idx = [c.index(n) for n in names]
        for det in s["details"].values():
            for m in MATRIX_MEASURES:
                if m in det:
                    det[m] = det[m][np.ix_(idx, idx)]
    return names


def group_matrices(subjects):
    """Mean matrix of every pair-wise measure per group and band (only if the channels coincide)."""
    out = {}
    for b in cfg.BANDS:
        for g in cfg.GROUPS:
            ss = [s for s in subjects if s["group"] == g]
            for m in MATRIX_MEASURES:
                mats = [s["details"][b][m] for s in ss if m in s["details"][b]]
                if mats:
                    out[(g, b, m)] = np.mean(mats, axis=0)
    return out


def edge_tests(subjects, names):
    """
    Mann-Whitney between groups on every channel pair, for EDGE_MEASURES in every band.
    p_fdr = FDR across the pairs of that band and measure. Long table, one row per pair.
    names: common channel order from align_channels.
    """
    groups = [g for g in cfg.GROUPS if any(s["group"] == g for s in subjects)]
    if len(groups) != 2:
        return pd.DataFrame()
    iu, ju = np.triu_indices(len(names), k=1)
    rows = []
    for b in cfg.BANDS:
        for m in EDGE_MEASURES:
            x = [np.array([upper(s["details"][b][m]) for s in subjects
                           if s["group"] == g and m in s["details"][b]]) for g in groups]
            if min(len(v) for v in x) < 2:
                continue
            p = stats.mannwhitneyu(x[0], x[1], axis=0).pvalue
            q = fdr_bh(p)
            med = [np.median(v, axis=0) for v in x]
            rows.append(pd.DataFrame(dict(band=b, measure=m, ch_i=np.array(names)[iu],
                                          ch_j=np.array(names)[ju], i=iu, j=ju,
                                          **{f"median_{g}": md for g, md in zip(groups, med)},
                                          diff=med[0] - med[1], p=p, p_fdr=q)))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ================================================================== MAIN
def main():
    channel_set, channels, out_root = base.choose_channel_set()
    print(f"Channels: {'all EEG channels' if channels is None else base.OCCIPITAL_CHANNELS}")
    fallback = FALLBACK_DIR if channels is None else FALLBACK_DIR / base.OCCIPITAL_SUBDIR
    out_dir = check_output_dir(cfg.results_dir(channel_set, out_root), OUTPUT_NAMES, fallback / cfg.OUT_SUBDIR)
    cache_dir = out_dir / "subjects"
    cache_dir.mkdir(exist_ok=True)
    print("Output folder:", out_dir)
    print(f"Bands: {cfg.BANDS} | Butterworth order {cfg.FILTER_ORDER} | edge trim {cfg.EDGE_TRIM_S} s | "
          f"surrogates {cfg.N_SURROGATES}")

    subjects, t_start = [], time.time()
    for group, folder in cfg.GROUPS.items():
        files = base.subject_files(group, discover_files(folder))
        print(f"{group}: {len(files)} files")
        for fpath in files:
            print("  ", fpath.name)
            try:
                res = load_or_run(fpath, group, channels, cache_dir)
                subjects.append(dict(group=group, subject=fpath.stem, **res))
            except Exception as e:
                print("    ERROR ->", e)
    if not subjects:
        raise SystemExit("No file analysed: check the paths in config_ips.py")
    print(f"\nAnalysis: {(time.time() - t_start) / 60:.1f} min")
    save_sync_results(subjects, out_dir, channel_set)


def save_sync_results(subjects, out_dir, channel_set):
    """Group tables, Excel and synchrony_results.pkl."""
    df = pd.DataFrame([r for s in subjects for r in s["rows"]])
    ep = pd.DataFrame([r for s in subjects for r in s["epochs"]])
    chan = pd.DataFrame([r for s in subjects for r in s["chans"]])
    tests, summary = group_tables(df)
    sf_tests = stim_freq_tests(ep)
    chan_wide, chan_tests = channel_tables(chan)
    ch_names = align_channels(subjects)
    if ch_names is not None:
        mats, edges = group_matrices(subjects), edge_tests(subjects, ch_names)
    else:
        print("Subjects do not all have the same channels: group matrices and edge tests skipped")
        mats, edges, ch_names = {}, pd.DataFrame(), None

    if len(tests):
        print("\n=== Between-group tests (Mann-Whitney, p_fdr within each band), only p < 0.05 ===")
        sig = tests[tests.p < 0.05]
        print(sig.round(4).to_string(index=False) if len(sig) else "none")
    if len(edges):
        n_sig = edges[edges.p_fdr < cfg.ALPHA].groupby(["band", "measure"]).size()
        print("\n=== Channel pairs with p_fdr < %.2f ===" % cfg.ALPHA)
        print(n_sig.to_string() if len(n_sig) else "none")

    params = pd.DataFrame([dict(parameter=k, value=str(v)) for k, v in cfg.params().items()])
    tables = {"subject_band_metrics": df, "group_tests": tests, "group_summary": summary,
              "stim_freq_tests": sf_tests, "epoch_metrics": ep, "channel_strength": chan_wide,
              "channel_tests": chan_tests, "edge_tests": edges, "parameters": params}
    save_tables_excel(tables, out_dir / "synchrony_results.xlsx", SHEET_ORDER)
    # for the figures: per-subject details without the full matrices (they stay in the caches)
    details = {(s["group"], s["subject"], b): {k: v for k, v in det.items() if k not in MATRIX_MEASURES}
               for s in subjects for b, det in s["details"].items()}
    save_results(dict(metrics=df, tests=tests, summary=summary, epochs=ep, stim_freq_tests=sf_tests,
                      chan=chan_wide, chan_tests=chan_tests, edges=edges, matrices=mats,
                      ch_names=ch_names, details=details, params=cfg.params(),
                      channel_set=channel_set, bands=dict(cfg.BANDS)),
                 out_dir / "synchrony_results.pkl")
    print("\nDone. Excel and synchrony_results.pkl in:", out_dir)


if __name__ == "__main__":
    main()
