# -*- coding: utf-8 -*-
"""
MAIN synchronization, IPS database: Kuramoto order parameter R(t), global synchrony (GS),
metastability (M), phase coherence (PC) and phase locking value (PLV / PPC) per band.
Independent from MAIN_IPS.py and MAIN_IPS_AVALAN.py: it reads the same .set files.
Figures are made afterwards with plot_ips_sync.py (no re-analysis).

For each subject the results are saved in <results>/subjects/<file>.pkl: when re-running, the
subjects already analysed with the same parameters (config_ips_sync.params) are only re-read.
To redo everything: OVERWRITE = True in config_ips_sync.py.

Pipeline per subject and band (modules in synchrony_criticality/), "new method" (2026-11-02):
  Butterworth band-pass (zero phase) + Hilbert on every run of contiguous epochs (an IPS epoch is
  one trial, padded by reflection) -> phase phi_j(t) -> edge trimming only where the data end
  -> R(t) -> GS (mean R); M = SD of R(t) over the whole analysed signal (not within each epoch)
  -> PC_ij per epoch (time average) -> PLV_ij / PPC_ij across epochs (Lachaux 1999; Vinck 2010)
  -> PLI, wPLI, debiased wPLI, imaginary coherence from multitaper cross-spectra averaged ACROSS
     EPOCHS (Nolte 2004; Stam 2007; Vinck 2011; cross_spectral.py): only the lagged coupling
M, PLV / PPC and the lag-insensitive indices depend on the number of epochs: they are computed on
the same number of epochs in every subject (mean over subsets, descriptive_analysis/segments.py).
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
from synchrony_criticality.phase import bandpass_filter, analytic_signal, trim_edges, drop_flat_channels
from synchrony_criticality.kuramoto import kuramoto_epochs
from synchrony_criticality.phase_coherence import phase_coherence_epochs, upper, node_strength
from synchrony_criticality.plv import plv_summary
from synchrony_criticality.cross_spectral import epoch_spectra, lag_insensitive_across_epochs
from descriptive_analysis.segments import apply_on_runs, runs_of_single_epochs, equal_subsets, mean_over_subsets
from photosensitive.io_eeg import source_signature, n_equal_epochs

warnings.filterwarnings("ignore")

OUTPUT_NAMES = ["synchrony_results.xlsx", "synchrony_results.pkl"]
SHEET_ORDER = ["subject_band_metrics", "group_tests", "group_summary", "stim_freq_tests",
               "epoch_metrics", "channel_tests", "channel_strength", "edge_tests", "parameters"]

# columns of the subject x band table compared between groups
TEST_METRICS = ["gs", "metastability", "metastability_cv", "gs_norm", "metastability_norm", "R_median",
                "R_q95", "pc_mean", "pc_median", "plv_mean", "ppc_mean", "pli_mean", "wpli_mean",
                "dwpli_mean", "icoh_mean"]
EPOCH_METRICS = ["gs", "metastability_epoch", "pc_mean"]      # per epoch, tested per stim frequency
LAG_MEASURES = ["pli", "wpli", "dwpli", "icoh"]               # lag-insensitive indices
MATRIX_MEASURES = ["pc", "plv", "ppc"] + LAG_MEASURES         # pair-wise matrices kept per subject
EDGE_MEASURES = ["pc", "ppc", "wpli", "dwpli"]                # pair-wise matrices tested per edge
STRENGTH_MEASURES = ["pc", "plv", "ppc", "wpli", "dwpli", "icoh"]   # node strength per channel


# ================================================================== ONE SUBJECT
def band_signal(data, sfreq, band, runs=None, pad_s=2.0, trim_s=0.0):
    """
    Analytic signal of one band: Butterworth band-pass + Hilbert on every run of contiguous epochs
    (padded by pad_s s of reflection), cut back into the epochs, then trim_s s removed at both ends
    of every epoch (IPS: the 1 s before and after the 10 s of stimulation). data: (n_ep, n_ch, n_t).
    """
    runs = runs_of_single_epochs(len(data)) if runs is None else runs
    z = apply_on_runs(data, runs, lambda x: analytic_signal(bandpass_filter(x, sfreq, band, cfg.FILTER_ORDER)),
                      sfreq, pad_s)
    return trim_edges(z, sfreq, trim_s) if trim_s else z


def lag_insensitive(data, sfreq, bands, subsets, trim_s=0.0, nw=2.0):
    """
    PLI, wPLI, dwPLI, iCOH of every band from multitaper cross-spectra across epochs (on the same
    samples as the phases: trim_s s cut at both ends), mean over the epoch subsets.
    Returns {band: {measure: (n_ch, n_ch)}}.
    """
    x = trim_edges(data, sfreq, trim_s) if trim_s else data
    X, freqs = epoch_spectra(x, sfreq, nw, max(b[1] for b in bands.values()) + 1)
    out = {}
    for b, band in bands.items():
        out[b] = mean_over_subsets(subsets, lambda idx: lag_insensitive_across_epochs(X, freqs, band, idx))
    return out


def kuramoto_equal(R, Rs, subsets):
    """
    Metastability on the same amount of data: M = SD of R(t) over all the samples of a subset of
    epochs (contiguous blocks in a continuous recording: the slow fluctuations of R are included),
    mean over subsets; Rs = R(t) of the surrogates, for M / surrogate M.
    """
    def one(idx):
        m = float(np.std(R[idx]))
        out = dict(metastability=m, metastability_cv=m / R[idx].mean() if R[idx].mean() > 0 else np.nan)
        if Rs:
            ms = float(np.mean([np.std(r[idx]) for r in Rs]))
            out.update(metastability_surrogate=ms, metastability_norm=m / ms)
        return out
    return mean_over_subsets(subsets, one)


def surrogate_R(u, n_surrogates, seed):
    """R(t) of n_surrogates circular-shift surrogates (every channel of every epoch shifted independently)."""
    from synchrony_criticality.kuramoto import circular_shift_surrogate
    rng = np.random.default_rng(seed)
    return [np.abs(circular_shift_surrogate(u, rng).mean(axis=-2)).astype(np.float32) for _ in range(n_surrogates)]


def plv_equal(phases, subsets):
    """PLV and PPC across epochs on the same number of epochs (mean over subsets)."""
    return mean_over_subsets(subsets, lambda idx: {k: v for k, v in plv_summary(phases[idx]).items()
                                                   if k in ("plv", "ppc")})


def analyze_band(data, sfreq, ch_names, labels, fstim, group, sid, band_name, band, runs=None, subsets=None,
                 lag=None, pad_s=2.0, trim_s=None):
    """
    All synchrony measures for one subject in one band. Returns (row, epoch rows, channel rows, detail).
    runs: contiguous epochs (None = separate trials); subsets: epoch subsets of equal size for the
    indices that depend on the number of epochs; lag: this band of lag_insensitive() (computed for
    all bands at once).
    """
    trim_s = cfg.EDGE_TRIM_S if trim_s is None else trim_s
    z = band_signal(data, sfreq, band, runs, pad_s, trim_s)
    phases = np.angle(z)
    n_ep, n_ch, n_t = phases.shape
    subsets = subsets or [np.arange(n_ep)]
    del z

    kur, kur_ep, R, hist = kuramoto_epochs(phases, 0, cfg.SEED, cfg.R_HIST_BINS)
    Rs = surrogate_R(np.exp(1j * phases).astype(np.complex64), cfg.N_SURROGATES, cfg.SEED) if cfg.N_SURROGATES else []
    if Rs:
        gs_s = float(np.mean([r.mean() for r in Rs]))
        kur.update(gs_surrogate=gs_s, gs_norm=kur["gs"] / gs_s)
    kur.pop("metastability_pooled", None)
    kur.update(metastability_epoch=kur.pop("metastability"), **kuramoto_equal(R, Rs, subsets))
    pc_ep = phase_coherence_epochs(phases)                    # (n_ep, n_ch, n_ch)
    pc = pc_ep.mean(0)
    pl = plv_equal(phases, subsets) if len(subsets[0]) >= 2 else None
    pl_t = plv_summary(phases)["plv_t"] if n_ep >= 2 else None

    row = dict(group=group, subject=sid, band=band_name, n_channels=n_ch, n_epochs=n_ep,
               n_epochs_equal=len(subsets[0]), n_equal_draws=len(subsets),
               duration_s=n_ep * n_t / sfreq, **kur,
               pc_mean=float(upper(pc).mean()), pc_median=float(np.median(upper(pc))),
               plv_mean=float(upper(pl["plv"]).mean()) if pl else np.nan,
               ppc_mean=float(upper(pl["ppc"]).mean()) if pl else np.nan,
               **{f"{m}_mean": float(upper(lag[m]).mean()) for m in LAG_MEASURES})

    pc_ep_mean = upper(pc_ep).mean(-1)
    epochs = [dict(group=group, subject=sid, band=band_name, epoch=k, label=labels[k],
                   stim_freq=fstim[k], gs=kur_ep["gs"][k], metastability_epoch=kur_ep["metastability"][k],
                   pc_mean=pc_ep_mean[k]) for k in range(n_ep)]

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
        detail.update(plv=pl["plv"].astype(np.float32), ppc=pl["ppc"].astype(np.float32))
    if pl_t is not None:
        detail.update(plv_t=pl_t[::dec].astype(np.float32),
                      plv_t_times=(trim_s + np.arange(n_t)[::dec] / sfreq).astype(np.float32))
    return row, epochs, chans, detail


def analyze_subject(fpath, group, channels, n_equal=None):
    """
    Loads one .set and analyses all bands. n_equal = epochs for the indices that depend on their
    number (the same in every subject). Returns dict(rows, epochs, chans, details, ch_names, params).
    """
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
    subsets = equal_subsets(len(data), n_equal, base.EQUAL_DRAWS, base.EQUAL_MODE, base.EQUAL_SEED)
    lag = lag_insensitive(data, sfreq, cfg.BANDS, subsets, cfg.EDGE_TRIM_S, cfg.MT_NW)
    rows, epochs, chans, details = [], [], [], {}
    for band_name, band in cfg.BANDS.items():
        t0 = time.time()
        row, e, c, det = analyze_band(data, sfreq, ch_names, labels, fstim, group, fpath.stem,
                                      band_name, band, None, subsets, lag[band_name], base.PAD_S)
        rows.append(row)
        epochs += e
        chans += c
        details[band_name] = det
        print(f"    {band_name:>6}: GS {row['gs']:.3f} (surr {row.get('gs_surrogate', np.nan):.3f}) | "
              f"M {row['metastability']:.3f} | PC {row['pc_mean']:.3f} | PLV {row['plv_mean']:.3f} "
              f"PPC {row['ppc_mean']:.3f} | wPLI {row['wpli_mean']:.3f} dwPLI {row['dwpli_mean']:.3f}  "
              f"({time.time() - t0:.1f} s)")
    return dict(rows=rows, epochs=epochs, chans=chans, details=details, ch_names=ch_names,
                params=cfg.params(), source=source_signature(fpath), n_equal=n_equal, sfreq=sfreq)


def load_or_run(fpath, group, channels, cache_dir, n_equal=None):
    """Re-reads the subject cache if it has the same parameters, file and n_equal, otherwise analyses and saves."""
    cache = cache_dir / f"{fpath.stem}.pkl"
    if cache.exists() and not cfg.OVERWRITE:
        res = load_results(cache)
        if (res.get("params") == cfg.params() and res.get("source") == source_signature(fpath)
                and res.get("n_equal") == n_equal):
            print("    (already analysed: read from cache)")
            return res
        print("    parameters, file or number of epochs changed: re-analysing")
    res = analyze_subject(fpath, group, channels, n_equal)
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
    out_dir = check_output_dir(cfg.results_dir(channel_set, out_root) / cfg.NEW_METHOD_SUBDIR, OUTPUT_NAMES,
                               fallback / cfg.OUT_SUBDIR / cfg.NEW_METHOD_SUBDIR)
    cache_dir = out_dir / "subjects"
    cache_dir.mkdir(exist_ok=True)
    print("Output folder:", out_dir)
    print(f"Bands: {cfg.BANDS} | Butterworth order {cfg.FILTER_ORDER} | edge trim {cfg.EDGE_TRIM_S} s | "
          f"surrogates {cfg.N_SURROGATES}")

    n_equal = n_equal_epochs(base.all_subject_files(cfg.GROUPS))
    print(f"M, PLV / PPC, PLI / wPLI / dwPLI / iCOH on {n_equal} epochs per subject ({base.EQUAL_DRAWS} random subsets)")
    subjects, t_start = [], time.time()
    for group, folder in cfg.GROUPS.items():
        files = base.subject_files(group, discover_files(folder))
        print(f"{group}: {len(files)} files")
        for fpath in files:
            print("  ", fpath.name)
            try:
                res = load_or_run(fpath, group, channels, cache_dir, n_equal)
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
