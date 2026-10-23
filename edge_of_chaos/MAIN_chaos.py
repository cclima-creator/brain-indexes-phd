# -*- coding: utf-8 -*-
"""
MAIN edge of chaos: loads the EEG of every subject, computes the indexes of chaos_indexes.py,
saves tables (Excel + pickle) and figures. Generic: a database MAIN calls run(); for a quick use
from the command line:

    python edge_of_chaos/MAIN_chaos.py --group asd=D:/eeg/asd --group td=D:/eeg/td --out D:/chaos_results

Database MAIN (e.g. autism/MAIN_AUT_CHAOS.py):

    from edge_of_chaos import config_chaos as cfg
    from edge_of_chaos.MAIN_chaos import run
    cfg.SIGNAL_BANDS = ["broadband", "alpha"]        # override what is needed
    run({"asd": FOLDER_ASD, "td": FOLDER_TD}, OUT_DIR, cfg, colors={"asd": "tab:red", "td": "tab:blue"},
        fixed_length_s=10.0)

Pipeline per subject (data in uV, (n_epochs, n_channels, n_times)), for every signal of
SIGNAL_BANDS (broadband 1-45 Hz, delta ... gamma), columns <measure>_<band>:
  band-pass, 1 s edge trimming -> per epoch and channel: LZC, Katz FD, Renyi H2, MSE
                               -> whole montage: covariance width and lambda_max on the amplitude envelope
                                  and on event counts (Dahmen 2019), entropy production
                               -> 0-1 chaos test K per channel (epochs concatenated); broadband
                                  also low-passed at each K_LOWPASS_HZ (k_lp<cutoff>_broadband)
  amplitude envelope           -> DFA alpha, fE/I per channel (epochs concatenated)
  Welch PSD averaged over epochs -> aperiodic exponent per channel (broadband only)
Every subject is cached in <out>/subjects/<file>.pkl and not recomputed with the same parameters;
run(..., n_jobs=k) analyses the missing subjects in k parallel processes.
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from edge_of_chaos import config_chaos as default_cfg
from edge_of_chaos import chaos_indexes as ci
from edge_of_chaos.io_chaos import (discover_files, load_subject, check_output_dir, save_results,
                                    load_cached_subject, save_tables_excel, load_results)

warnings.filterwarnings("ignore")

SHEET_ORDER = ["subject_metrics", "group_tests", "group_summary", "channel_tests",
               "channel_metrics", "epoch_metrics", "mse_scale_tests", "parameters"]


def _long_signal(x, concatenate):
    """(n_epochs, n_channels, n_times) -> list of (n_channels, n) signals: one, or one per epoch."""
    return [ci.concat_epochs(x)] if concatenate else list(x)


def _mean_over(signals, f):
    """f applied to every long signal (n_channels, n) channel by channel, mean over signals."""
    vals = np.array([[f(s[c]) for c in range(s.shape[0])] for s in signals], float)
    return np.nanmean(vals, axis=0)


# ================================================================== ONE SUBJECT
LAMBDA_TAG = {"envelope": "env", "events": "evt"}
LAMBDA_PARAMS = {"LAMBDA_N_UNITS", "LAMBDA_SIGNALS", "LAMBDA_WINDOW_S", "LAMBDA_EVENT_SD",
                 "LAMBDA_DEMEAN_EPOCHS"}


def lambda_band(x, env, sf, bname, cfg):
    """
    Covariance width and lambda_max of one signal band for every cfg.LAMBDA_SIGNALS:
    x = band-passed signal, env = its amplitude envelope, both (n_epochs, n_channels, n_times).
    Returns (whole-montage values, details): cov_width_<env|evt>_<band>, lambda_max_<env|evt>_<band>
    (+ mean_diag, sd_offdiag, mean_offdiag, event rate), and cov_width_<band> / lambda_max_<band>
    for the old edgeofpy estimate if "edgeofpy" is listed.
    """
    glob, detail = {}, {}
    for sig in cfg.LAMBDA_SIGNALS:
        if sig == "edgeofpy":
            cw = ci.compute_covariance_width(x)
            glob[f"cov_width_{bname}"] = cw["width"]
            glob[f"lambda_max_{bname}"] = ci.compute_lyapunov_exponent(x, cfg.LAMBDA_N_UNITS, cw["width"])["lambda_max"]
            continue
        r = ci.compute_covariance_width_dahmen(env if sig == "envelope" else x, sf, sig, cfg.LAMBDA_WINDOW_S,
                                               cfg.LAMBDA_EVENT_SD, cfg.LAMBDA_N_UNITS, cfg.LAMBDA_DEMEAN_EPOCHS)
        t = LAMBDA_TAG[sig]
        glob.update({f"cov_width_{t}_{bname}": r["width"], f"lambda_max_{t}_{bname}": r["lambda_max"],
                     f"cov_mean_diag_{t}_{bname}": r["mean_diag"], f"cov_sd_offdiag_{t}_{bname}": r["sd_offdiag"],
                     f"cov_mean_offdiag_{t}_{bname}": r["mean_offdiag"]})
        if sig == "events":
            glob[f"event_rate_{t}_{bname}"] = r["event_rate"]
        detail[f"cov_{t}_{bname}"] = r["cov"].astype(np.float32)
    return glob, detail


def analyze_lambda(sub, cfg):
    """Only the covariance-width / lambda_max measures of one subject (to update a cached subject)."""
    data, sf = sub["data"], sub["sfreq"]
    glob, detail = {}, {}
    for bname, band in signal_bands(cfg).items():
        x = ci.broadband(data, sf, band, cfg.FILTER_ORDER, cfg.EDGE_TRIM_S)
        env = ci.band_envelope(data, sf, band, cfg.FILTER_ORDER, cfg.EDGE_TRIM_S)
        g, d = lambda_band(x, env, sf, bname, cfg)
        glob.update(g)
        detail.update(d)
    return glob, detail


def signal_bands(cfg):
    """Signals every measure is computed on: {"broadband": BROADBAND, delta: ..., ...} (SIGNAL_BANDS)."""
    return {name: (cfg.BROADBAND if name == "broadband" else cfg.BANDS[name]) for name in cfg.SIGNAL_BANDS}


def analyze_subject(sub, cfg, group, sid):
    """
    All measures for one subject, on every signal of cfg.SIGNAL_BANDS (broadband and bands).
    Column names: <measure>_<band> (e.g. lzc_alpha, dfa_broadband, lambda_max_gamma); the
    aperiodic exponent is a property of the whole spectrum and is computed once (broadband).
    sub = io_chaos.load_subject output. Returns dict(subject row, channel rows, epoch rows, detail).
    """
    data, sf, ch = sub["data"], sub["sfreq"], sub["ch_names"]
    n_ep, n_ch, n_t = data.shape
    want = set(cfg.MEASURES)
    o, tr = cfg.FILTER_ORDER, cfg.EDGE_TRIM_S
    chan = {c: dict(group=group, subject=sid, channel=c) for c in ch}
    glob, detail, timing, ep_cols = {}, {}, {}, {}

    def put(col, values):
        for c, v in zip(ch, values):
            chan[c][col] = float(v)

    def tick(key, t0):
        timing[key] = timing.get(key, 0.0) + time.time() - t0

    per_epoch = [("lempel_ziv_complexity", "lzc", lambda x: ci.compute_lempel_ziv_complexity(x, cfg.LZC_SYMBOLIZE)),
                 ("katz_fd", "katz_fd", ci.compute_katz_fd),
                 ("renyi_entropy", f"renyi_h{cfg.RENYI_ALPHA}",
                  lambda x: ci.compute_renyi_entropy(x, cfg.RENYI_ALPHA, cfg.RENYI_BINS))]
    k01 = lambda x: ci.compute_01_chaos_test(x, sigma=cfg.K_SIGMA, n_c=cfg.K_N_C, denoise=cfg.K_DENOISE,
                                             downsample=cfg.K_DOWNSAMPLE, impl=cfg.K_IMPL, seed=cfg.SEED)

    for bname, band in signal_bands(cfg).items():
        x = ci.broadband(data, sf, band, o, tr)            # band-passed, edge-trimmed signal

        # ---- per epoch and channel
        for name, col, f in per_epoch:
            if name in want:
                t0 = time.time()
                v = np.array([[f(x[e, c]) for c in range(n_ch)] for e in range(n_ep)])
                put(f"{col}_{bname}", v.mean(0))
                ep_cols[f"{col}_{bname}"] = v.mean(1)
                tick(col, t0)
        if "multiscale_entropy" in want:
            t0 = time.time()
            auc = np.empty((n_ep, n_ch))
            curves = np.empty((n_ep, n_ch, len(cfg.MSE_SCALES)))
            for e in range(n_ep):
                for c in range(n_ch):
                    m = ci.compute_multiscale_entropy(x[e, c], cfg.MSE_SCALES, cfg.MSE_DIMENSION,
                                                      cfg.MSE_TOLERANCE, cfg.MSE_METHOD)
                    auc[e, c], curves[e, c] = m["auc"], m["curve"]
            put(f"mse_auc_{bname}", auc.mean(0))
            ep_cols[f"mse_auc_{bname}"] = auc.mean(1)
            detail[f"mse_curve_{bname}"] = np.nanmean(curves, axis=0)      # (n_channels, n_scales)
            detail["mse_scales"] = np.asarray(cfg.MSE_SCALES)
            tick("mse", t0)

        # ---- 0-1 chaos test on the band signal (+ the low-passed variants for broadband)
        if "01_chaos_test" in want:
            t0 = time.time()
            put(f"k_{bname}", _mean_over(_long_signal(x, cfg.CONCATENATE_EPOCHS), k01))
            if bname == "broadband":
                for fc in cfg.K_LOWPASS_HZ:
                    lp = ci.trim_edges(ci.lowpass_filter(ci.bandpass_filter(data, sf, band, o), sf, fc, o), sf, tr)
                    put(f"k_lp{fc:g}_broadband", _mean_over(_long_signal(lp, cfg.CONCATENATE_EPOCHS), k01))
            tick("01_test", t0)

        # ---- DFA and fE/I on the amplitude envelope of the signal
        if {"dfa", "fei_ratio"} & want:
            t0 = time.time()
            env = _long_signal(ci.band_envelope(data, sf, band, o, tr), cfg.CONCATENATE_EPOCHS)
            dfa = [[ci.compute_dfa(s[c], sf, cfg.DFA_FIT_S, cfg.DFA_N_SCALES, cfg.DFA_OVERLAP)
                    for c in range(n_ch)] for s in env]
            alpha = np.array([[d["alpha"] for d in row] for row in dfa])          # (n_signals, n_ch)
            if "dfa" in want:
                put(f"dfa_{bname}", np.nanmean(alpha, 0))
                detail[f"dfa_{bname}"] = dict(windows_s=dfa[0][0]["windows"] / sf,
                                              fluct=np.array([d["fluctuations"] for d in dfa[0]]))
            if "fei_ratio" in want:
                fei = np.array([[ci.compute_fei_ratio(s[c], sf, cfg.FEI_WINDOW_S, cfg.FEI_OVERLAP,
                                                      alpha[i, c], cfg.FEI_DFA_THRESHOLD)["fei"]
                                 for c in range(n_ch)] for i, s in enumerate(env)])
                put(f"fei_{bname}", np.nanmean(fei, 0))
            tick("dfa_fei", t0)

        # ---- whole montage
        if {"covariance_width", "lyapunov_exponent"} & want:
            t0 = time.time()
            g, d = lambda_band(x, ci.band_envelope(data, sf, band, o, tr), sf, bname, cfg)
            glob.update(g)
            detail.update(d)
            tick("cov_lambda", t0)
        if "entropy_production" in want:
            t0 = time.time()
            epr = ci.compute_entropy_production(x, sf, cfg.EP_N_STATES, cfg.EP_LAG_S, cfg.EP_PSEUDOCOUNT, cfg.SEED)
            glob[f"entropy_production_{bname}"] = epr["ep"]
            glob[f"entropy_production_per_s_{bname}"] = epr["ep_per_s"]
            detail[f"ep_P_{bname}"] = epr["P"]
            tick("entropy_production", t0)

    # ---- aperiodic exponent: one fit of the whole spectrum (not defined per band)
    if "spectral_slope" in want:
        t0 = time.time()
        sp = ci.compute_spectral_slope(data, sf, cfg.APERIODIC_FRANGE, cfg.APERIODIC_MODE,
                                       cfg.PSD_SEGMENT_S, cfg.BROADBAND[1])
        put("aperiodic_exponent_broadband", sp["exponent"])
        put("aperiodic_offset", sp["offset"])
        put("aperiodic_r2", sp["r2"])
        detail.update(psd=sp["psd"].astype(np.float32), psd_freqs=sp["freqs"], aperiodic_method=sp["method"])
        tick("aperiodic", t0)

    chan_rows = list(chan.values())
    num = pd.DataFrame(chan_rows).drop(columns=["group", "subject", "channel"])
    row = dict(group=group, subject=sid, n_channels=n_ch, n_epochs=n_ep, duration_s=n_ep * n_t / sf,
               sfreq=sf, **num.mean(numeric_only=True).to_dict(), **glob)
    epochs = [dict(group=group, subject=sid, epoch=e, label=sub["labels"][e],
                   **{k: float(v[e]) for k, v in ep_cols.items()}) for e in range(n_ep)]
    print(f"    {sid}: " + ", ".join(f"{k} {v:.0f}s" for k, v in timing.items()), flush=True)
    return dict(row=row, chans=chan_rows, epochs=epochs, detail=detail, ch_names=ch)


def _only_lambda_changed(cache, params):
    """True if the cache exists and differs from params only in the lambda settings."""
    if not Path(cache).exists():
        return False
    try:
        old = load_results(cache).get("params", {})
    except Exception:
        return False
    diff = {k for k in set(old) | set(params) if old.get(k) != params.get(k)}
    return bool(diff) and diff <= LAMBDA_PARAMS


def _cfg_state(cfg):
    """Upper-case settings of a config module (to rebuild it in a worker process)."""
    return {k: getattr(cfg, k) for k in dir(cfg) if k.isupper()}


def _analyze_file(job):
    """
    Worker (one process per subject when n_jobs > 1): rebuilds the config from its settings,
    loads, analyses and caches one file. job = (path, group, cache, params, cfg_state, load_kw,
    patch); patch = only the lambda settings changed: just those measures are recomputed into the cache.
    """
    path, group, cache, params, state, load_kw, patch = job
    from edge_of_chaos import config_chaos as cfg
    for k, v in state.items():
        setattr(cfg, k, v)
    t0 = time.time()
    sub = load_subject(path, sfreq=cfg.SFREQ, max_epochs=cfg.MAX_EPOCHS, **load_kw)
    if patch:                       # cached with other lambda settings only: redo just those
        res = load_results(cache)
        glob, detail = analyze_lambda(sub, cfg)
        lam = ("cov_", "lambda_max_", "event_rate_")
        res["row"] = {k: v for k, v in res["row"].items() if not k.startswith(lam)}
        res["row"].update(glob)
        res["detail"] = {k: v for k, v in res["detail"].items() if not k.startswith("cov")}
        res["detail"].update(detail)
    else:
        res = analyze_subject(sub, cfg, group, Path(path).stem)
    res["params"] = params
    save_results(res, cache)
    print(f"  {Path(path).stem} done in {time.time() - t0:.0f} s", flush=True)
    return str(cache)


# ================================================================== ALL SUBJECTS
def run(groups, out_dir, cfg=default_cfg, colors=None, pattern="*.set", fixed_length_s=None,
        window=None, channels=None, file_filter=None, plots=True, n_jobs=1):
    """
    groups: {group name: folder}; out_dir: results folder. fixed_length_s / window / channels are
    passed to io_chaos.load_subject; file_filter(group, files) -> files (e.g. excluded subjects).
    n_jobs > 1 analyses the subjects not yet cached in parallel processes (one subject each).
    Returns dict of tables (also saved in out_dir/chaos_results.xlsx and .pkl).
    """
    from edge_of_chaos import report_chaos
    out = check_output_dir(out_dir)
    (out / "subjects").mkdir(exist_ok=True)
    colors = colors or {g: f"C{i}" for i, g in enumerate(groups)}
    params = dict(cfg.params(), fixed_length_s=fixed_length_s, window=window, channels=channels)
    print(f"Edge of chaos -> {out}\nmeasures: {cfg.MEASURES}")

    jobs, caches = [], []
    load_kw = dict(fixed_length_s=fixed_length_s, window=window, channels=channels)
    for group, folder in groups.items():
        files = discover_files(folder, pattern)
        if file_filter is not None:
            files = file_filter(group, files)
        print(f"\n== {group}: {len(files)} files in {folder}")
        for f in files:
            cache = out / "subjects" / f"{f.stem}.pkl"
            caches.append(cache)
            if not cfg.OVERWRITE and load_cached_subject(cache, params) is not None:
                print(f"  {f.stem} (cache)")
            else:
                patch = not cfg.OVERWRITE and _only_lambda_changed(cache, params)
                if patch:
                    print(f"  {f.stem} (cache, lambda measures to update)")
                jobs.append((str(f), group, cache, params, _cfg_state(cfg), load_kw, patch))
    print(f"\n{len(jobs)} subjects to analyse, {n_jobs} parallel process(es)", flush=True)
    if n_jobs > 1 and len(jobs) > 1:
        import os
        from concurrent.futures import ProcessPoolExecutor
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[v] = "1"                       # one BLAS thread per process (inherited)
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            list(ex.map(_analyze_file, jobs))
    else:
        for job in jobs:
            _analyze_file(job)
    results = [load_cached_subject(c, params) for c in caches]
    results = [r for r in results if r is not None]
    if not results:
        raise SystemExit("No subjects found: check folders and pattern.")

    tables = report_chaos.build_tables(results, cfg)
    tables["parameters"] = pd.DataFrame([dict(parameter=k, value=str(v)) for k, v in params.items()])
    save_tables_excel(tables, out / "chaos_results.xlsx", SHEET_ORDER)
    save_results(dict(tables=tables, details={r["row"]["subject"]: r["detail"] for r in results},
                      colors=colors, params=params), out / "chaos_results.pkl")
    print(f"\nTables saved in {out / 'chaos_results.xlsx'}")
    if plots:
        from edge_of_chaos import plot_chaos
        plot_chaos.make_all_figures(out / "chaos_results.pkl", out / "figures")
    return tables


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Edge-of-chaos indexes for groups of EEG files")
    p.add_argument("--group", action="append", required=True, metavar="NAME=FOLDER",
                   help="group name and folder (repeat for every group)")
    p.add_argument("--out", required=True, help="results folder")
    p.add_argument("--pattern", default="*.set")
    p.add_argument("--fixed-length", type=float, default=None,
                   help="epoch length (s) for continuous recordings")
    p.add_argument("--max-epochs", type=int, default=None, help="only the first N epochs (quick run)")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    a = _parse_args()
    groups = dict(g.split("=", 1) for g in a.group)
    default_cfg.MAX_EPOCHS = a.max_epochs
    run(groups, a.out, default_cfg, pattern=a.pattern, fixed_length_s=a.fixed_length,
        plots=not a.no_plots)
