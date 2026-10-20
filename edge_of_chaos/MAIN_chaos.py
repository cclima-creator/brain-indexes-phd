# -*- coding: utf-8 -*-
"""
MAIN edge of chaos: loads the EEG of every subject, computes the indexes of chaos_indexes.py,
saves tables (Excel + pickle) and figures. Generic: a database MAIN calls run(); for a quick use
from the command line:

    python edge_of_chaos/MAIN_chaos.py --group asd=D:/eeg/asd --group td=D:/eeg/td --out D:/chaos_results

Database MAIN (e.g. autism/MAIN_AUT_CHAOS.py):

    from edge_of_chaos import config_chaos as cfg
    from edge_of_chaos.MAIN_chaos import run
    cfg.LRTC_BANDS = ["alpha"]                       # override what is needed
    run({"asd": FOLDER_ASD, "td": FOLDER_TD}, OUT_DIR, cfg, colors={"asd": "tab:red", "td": "tab:blue"},
        fixed_length_s=10.0)

Pipeline per subject (data in uV, (n_epochs, n_channels, n_times)):
  broadband 1-45 Hz, 1 s edge trimming -> per epoch and channel: LZC, Katz FD, Renyi H2, MSE
                                       -> whole montage: covariance width, lambda_max, entropy production
  low-pass at each K_LOWPASS_HZ        -> 0-1 chaos test K per channel (epochs concatenated)
  band envelope (LRTC_BANDS)           -> DFA alpha, fE/I per channel (epochs concatenated)
  Welch PSD averaged over epochs       -> aperiodic exponent per channel
Every subject is cached in <out>/subjects/<file>.pkl and not recomputed with the same parameters.
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
                                    load_cached_subject, save_tables_excel)

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
def analyze_subject(sub, cfg, group, sid):
    """
    All measures for one subject. sub = io_chaos.load_subject output.
    Returns dict(subject row, channel rows, epoch rows, detail for the figures).
    """
    data, sf, ch = sub["data"], sub["sfreq"], sub["ch_names"]
    n_ep, n_ch, n_t = data.shape
    want = set(cfg.MEASURES)
    o, tr = cfg.FILTER_ORDER, cfg.EDGE_TRIM_S
    chan = {c: dict(group=group, subject=sid, channel=c) for c in ch}
    glob, detail, timing = {}, {}, {}

    def put(col, values):
        for c, v in zip(ch, values):
            chan[c][col] = float(v)

    bb = ci.broadband(data, sf, cfg.BROADBAND, o, tr)

    # ---- per epoch and channel, broadband
    ep_cols = {}
    per_epoch = [("lempel_ziv_complexity", "lzc", lambda x: ci.compute_lempel_ziv_complexity(x, cfg.LZC_SYMBOLIZE)),
                 ("katz_fd", "katz_fd", ci.compute_katz_fd),
                 ("renyi_entropy", f"renyi_h{cfg.RENYI_ALPHA}",
                  lambda x: ci.compute_renyi_entropy(x, cfg.RENYI_ALPHA, cfg.RENYI_BINS))]
    for name, col, f in per_epoch:
        if name in want:
            t0 = time.time()
            v = np.array([[f(bb[e, c]) for c in range(n_ch)] for e in range(n_ep)])
            put(col, v.mean(0))
            ep_cols[col] = v.mean(1)
            timing[col] = time.time() - t0
    if "multiscale_entropy" in want:
        t0 = time.time()
        auc = np.empty((n_ep, n_ch))
        curves = np.empty((n_ep, n_ch, len(cfg.MSE_SCALES)))
        for e in range(n_ep):
            for c in range(n_ch):
                m = ci.compute_multiscale_entropy(bb[e, c], cfg.MSE_SCALES, cfg.MSE_DIMENSION,
                                                  cfg.MSE_TOLERANCE, cfg.MSE_METHOD)
                auc[e, c], curves[e, c] = m["auc"], m["curve"]
        put("mse_auc", auc.mean(0))
        ep_cols["mse_auc"] = auc.mean(1)
        detail["mse_curve"] = np.nanmean(curves, axis=0)          # (n_channels, n_scales)
        detail["mse_scales"] = np.asarray(cfg.MSE_SCALES)
        timing["mse"] = time.time() - t0

    # ---- 0-1 chaos test on the low-passed signal
    if "01_chaos_test" in want:
        t0 = time.time()
        for fc in cfg.K_LOWPASS_HZ:
            lp = ci.trim_edges(ci.lowpass_filter(ci.bandpass_filter(data, sf, cfg.BROADBAND, o), sf, fc, o), sf, tr)
            k = _mean_over(_long_signal(lp, cfg.CONCATENATE_EPOCHS),
                           lambda x: ci.compute_01_chaos_test(x, sigma=cfg.K_SIGMA, n_c=cfg.K_N_C,
                                                              denoise=cfg.K_DENOISE,
                                                              downsample=cfg.K_DOWNSAMPLE,
                                                              impl=cfg.K_IMPL, seed=cfg.SEED))
            put(f"k_lp{fc:g}", k)
        timing["01_test"] = time.time() - t0

    # ---- DFA and fE/I on band envelopes
    if {"dfa", "fei_ratio"} & want:
        t0 = time.time()
        for band in cfg.LRTC_BANDS:
            env = _long_signal(ci.band_envelope(data, sf, cfg.BANDS[band], o, tr), cfg.CONCATENATE_EPOCHS)
            dfa = [[ci.compute_dfa(s[c], sf, cfg.DFA_FIT_S, cfg.DFA_N_SCALES, cfg.DFA_OVERLAP)
                    for c in range(n_ch)] for s in env]
            alpha = np.array([[d["alpha"] for d in row] for row in dfa])          # (n_signals, n_ch)
            if "dfa" in want:
                put(f"dfa_{band}", np.nanmean(alpha, 0))
                detail[f"dfa_{band}"] = dict(windows_s=dfa[0][0]["windows"] / sf,
                                             fluct=np.array([d["fluctuations"] for d in dfa[0]]))
            if "fei_ratio" in want:
                fei = np.array([[ci.compute_fei_ratio(s[c], sf, cfg.FEI_WINDOW_S, cfg.FEI_OVERLAP,
                                                      alpha[i, c], cfg.FEI_DFA_THRESHOLD)["fei"]
                                 for c in range(n_ch)] for i, s in enumerate(env)])
                put(f"fei_{band}", np.nanmean(fei, 0))
        timing["dfa_fei"] = time.time() - t0

    # ---- aperiodic exponent
    if "spectral_slope" in want:
        t0 = time.time()
        sp = ci.compute_spectral_slope(data, sf, cfg.APERIODIC_FRANGE, cfg.APERIODIC_MODE,
                                       cfg.PSD_SEGMENT_S, cfg.BROADBAND[1])
        put("aperiodic_exponent", sp["exponent"])
        put("aperiodic_offset", sp["offset"])
        put("aperiodic_r2", sp["r2"])
        detail.update(psd=sp["psd"].astype(np.float32), psd_freqs=sp["freqs"], aperiodic_method=sp["method"])
        timing["aperiodic"] = time.time() - t0

    # ---- whole montage
    if {"covariance_width", "lyapunov_exponent"} & want:
        t0 = time.time()
        cw = ci.compute_covariance_width(bb)
        glob["cov_width"] = cw["width"]
        if "lyapunov_exponent" in want:
            glob["lambda_max"] = ci.compute_lyapunov_exponent(bb, cfg.LAMBDA_N_UNITS, cw["width"])["lambda_max"]
        detail["cov"] = cw["cov"].astype(np.float32)
        timing["cov_lambda"] = time.time() - t0
    if "entropy_production" in want:
        t0 = time.time()
        epr = ci.compute_entropy_production(bb, sf, cfg.EP_N_STATES, cfg.EP_LAG_S, cfg.EP_PSEUDOCOUNT, cfg.SEED)
        glob["entropy_production"], glob["entropy_production_per_s"] = epr["ep"], epr["ep_per_s"]
        detail["ep_P"] = epr["P"]
        timing["entropy_production"] = time.time() - t0

    chan_rows = list(chan.values())
    num = pd.DataFrame(chan_rows).drop(columns=["group", "subject", "channel"])
    row = dict(group=group, subject=sid, n_channels=n_ch, n_epochs=n_ep, duration_s=n_ep * n_t / sf,
               sfreq=sf, **num.mean(numeric_only=True).to_dict(), **glob)
    epochs = [dict(group=group, subject=sid, epoch=e, label=sub["labels"][e],
                   **{k: float(v[e]) for k, v in ep_cols.items()}) for e in range(n_ep)]
    print("    " + ", ".join(f"{k} {v:.1f}s" for k, v in timing.items()))
    return dict(row=row, chans=chan_rows, epochs=epochs, detail=detail, ch_names=ch)


# ================================================================== ALL SUBJECTS
def run(groups, out_dir, cfg=default_cfg, colors=None, pattern="*.set", fixed_length_s=None,
        window=None, channels=None, file_filter=None, plots=True):
    """
    groups: {group name: folder}; out_dir: results folder. fixed_length_s / window / channels are
    passed to io_chaos.load_subject; file_filter(group, files) -> files (e.g. excluded subjects).
    Returns dict of tables (also saved in out_dir/chaos_results.xlsx and .pkl).
    """
    from edge_of_chaos import report_chaos
    out = check_output_dir(out_dir)
    (out / "subjects").mkdir(exist_ok=True)
    colors = colors or {g: f"C{i}" for i, g in enumerate(groups)}
    params = dict(cfg.params(), fixed_length_s=fixed_length_s, window=window, channels=channels)
    print(f"Edge of chaos -> {out}\nmeasures: {cfg.MEASURES}")

    results = []
    for group, folder in groups.items():
        files = discover_files(folder, pattern)
        if file_filter is not None:
            files = file_filter(group, files)
        print(f"\n== {group}: {len(files)} files in {folder}")
        for i, f in enumerate(files, 1):
            cache = out / "subjects" / f"{f.stem}.pkl"
            res = None if cfg.OVERWRITE else load_cached_subject(cache, params)
            if res is not None:
                print(f"  [{i}/{len(files)}] {f.stem} (cache)")
            else:
                print(f"  [{i}/{len(files)}] {f.stem}")
                t0 = time.time()
                sub = load_subject(f, fixed_length_s, window, channels, cfg.SFREQ, cfg.MAX_EPOCHS)
                print(f"    {sub['data'].shape[0]} epochs x {sub['data'].shape[1]} channels x "
                      f"{sub['data'].shape[2] / sub['sfreq']:.1f} s, sfreq {sub['sfreq']:g} Hz")
                res = analyze_subject(sub, cfg, group, f.stem)
                res["params"] = params
                save_results(res, cache)
                print(f"    done in {time.time() - t0:.0f} s")
            results.append(res)
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
