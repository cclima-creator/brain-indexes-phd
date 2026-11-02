# -*- coding: utf-8 -*-
"""
0 - Statistical (descriptive) analysis, Ayahuasca database.

Per subject x condition (cache in results/cache/stats/):
  PSD (Welch, one 5 s window per epoch, 0.2 Hz) -> absolute / relative / log10 power per band and
  channel, total 1-45 Hz power ("broadband"), broadband SNR (1-45 Hz vs 55-70 Hz)
  per signal (broadband 1-45 Hz and every band, zero-phase FIR):
      log10 variance, ACF width, skewness, excess kurtosis, fraction of samples beyond 3 SD,
      fraction of non-stationary epochs x channels (Reverse Arrangement Test, p < 0.05)
  raw signal: amplitude distribution (z-scored histogram, KS distance from the Gaussian)
  FOOOF (knee mode, 2-40 Hz): aperiodic exponent, offset, knee frequency, individual alpha peak

Per analysis (results/<analysis>/0_statistical_analysis/):
  stats_results.xlsx             subject values, Friedman + Wilcoxon tests, descriptive summary
  figures/                       explanatory figures (PSD, power, amplitude, stationarity, aperiodic,
                                 topographies, summary heatmap)
  boxplots_per_index/<index>.png one panel per band (broadband first), paired boxplots

    python psychedelics/MAIN_AYA_STATS.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mne
from photosensitive.MAIN_IPS_AVALAN import band_filter

from psychedelics import config_aya as cfg
from psychedelics import io_aya, runner
from psychedelics import paired_stats as ps
from psychedelics import paired_plots as pp
from psychedelics import compare_drug
from descriptive_analysis.spectral import compute_psd, band_power_table, aperiodic_fit
from descriptive_analysis.snr import snr_broadband
from descriptive_analysis.amplitude import amplitude_distribution
from descriptive_analysis.summary_stats import acf_width, stationarity_test
from descriptive_analysis.plots import paper_style, _save, BASE_FONT

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

MODULE = "stats"
BROADBAND = (1.0, 45.0)
SIG_BANDS = {"broadband": BROADBAND, **cfg.BANDS}
FMIN, FMAX_PSD, FMAX_PLOT = 1.0, 70.0, 45.0
SNR_SIGNAL, SNR_NOISE = (1.0, 45.0), (55.0, 70.0)
STAT_N_SEGMENTS, STAT_ALPHA, ACF_MAX_LAG_S = 20, 0.05, 1.0
APERIODIC_RANGE, APERIODIC_MODE = (2.0, 40.0), "knee"
FOOOF_SETTINGS = dict(peak_width_limits=(1.0, 12.0), max_n_peaks=8, min_peak_height=0.05, peak_threshold=2.0)
ALPHA_PEAK_BAND = (8.0, 13.0)
COMMON_F = np.round(np.arange(FMIN, FMAX_PLOT + 1e-9, 0.2), 1)

# indices with one value per band (columns <index>_<band>) and labels for the figures
BAND_INDICES = {"log10_abs": "log10 absolute power (µV²)", "rel": "relative power (of 1-45 Hz)",
                "log10_var": "log10 variance (µV²)", "acfw_ms": "ACF width (ms)", "skew": "skewness",
                "excess_kurtosis": "excess kurtosis", "frac_beyond_3sd": "fraction of samples beyond ±3 SD",
                "frac_nonstationary": "fraction non-stationary (RAT p<0.05)"}
SCALAR_INDICES = {"aperiodic_exponent": "aperiodic exponent (FOOOF knee)",
                  "aperiodic_offset": "aperiodic offset (log10 µV²/Hz)",
                  "aperiodic_knee_hz": "knee frequency (Hz)", "alpha_peak_hz": "individual alpha peak (Hz)",
                  "alpha_peak_power": "alpha peak power above 1/f (log10)",
                  "snr_broadband_dB": "broadband SNR (dB, 1-45 vs 55-70 Hz)",
                  "rms_uV": "RMS amplitude raw signal (µV)", "ks_D_vs_gauss": "KS distance from Gaussian (raw)"}
# n_epochs (and duration_s) stay in the subject table as a quality check, not tested: they follow the
# condition length (6 vs 20 min) and the epochs removed, they are not a property of the brain signal


def params():
    return dict(SIG_BANDS=SIG_BANDS, BANDS=cfg.BANDS, FMIN=FMIN, FMAX_PSD=FMAX_PSD, SNR=(SNR_SIGNAL, SNR_NOISE),
                STAT=(STAT_N_SEGMENTS, STAT_ALPHA, ACF_MAX_LAG_S), AP=(APERIODIC_RANGE, APERIODIC_MODE),
                FOOOF=FOOOF_SETTINGS, CONDITIONS=repr(cfg.CONDITIONS), FILTER="contiguous runs, reflection padding",
                PAD_S=cfg.PAD_S)


# ================================================================== ONE SUBJECT x CONDITION
def analyze(cond, sid):
    d = io_aya.load_data(cond, sid)
    x, sf, ch = d["data"], d["sfreq"], d["ch_names"]
    n_ep, n_ch, n_t = x.shape
    freqs, psd = compute_psd(x, sf, FMIN, FMAX_PSD, None)
    psd_sub = psd.mean(0)                                         # (n_ch, n_f)
    bands_df = band_power_table(psd_sub, freqs, ch, cfg.BANDS, BROADBAND)
    tot = band_power_table(psd_sub, freqs, ch, {"broadband": BROADBAND}, BROADBAND)
    bands_df = pd.concat([tot.assign(rel=np.nan), bands_df], ignore_index=True)

    chan = {c: dict(channel=c) for c in ch}
    for r in bands_df.itertuples():
        chan[r.channel][f"log10_abs_{r.band}"] = r.log10_abs
        chan[r.channel][f"rel_{r.band}"] = r.rel
    row = dict(n_epochs=n_ep, n_channels=n_ch, duration_s=n_ep * n_t / sf)
    for b, band in SIG_BANDS.items():
        xb = band_filter(x, sf, band, d["runs"], cfg.PAD_S)          # filtered on the continuous runs
        v = xb.var(-1)                                            # (n_ep, n_ch)
        acfw = acf_width(xb, sf, ACF_MAX_LAG_S)
        pst = stationarity_test(xb, STAT_N_SEGMENTS)
        flat = xb.transpose(1, 0, 2).reshape(n_ch, -1)
        z = (flat - flat.mean(1, keepdims=True)) / flat.std(1, keepdims=True)
        from scipy import stats
        vals = dict(log10_var=np.log10(v.mean(0)), acfw_ms=np.nanmean(acfw, 0),
                    skew=stats.skew(flat, axis=1), excess_kurtosis=stats.kurtosis(flat, axis=1),
                    frac_beyond_3sd=(np.abs(z) > 3).mean(1), frac_nonstationary=(pst < STAT_ALPHA).mean(0))
        for k, arr in vals.items():
            for c, val in zip(ch, arr):
                chan[c][f"{k}_{b}"] = float(val)
    amp = amplitude_distribution(x)
    ap = aperiodic_fit(psd_sub, freqs, APERIODIC_RANGE, APERIODIC_MODE, FOOOF_SETTINGS, ALPHA_PEAK_BAND)
    for c_i, c in enumerate(ch):
        chan[c].update(aperiodic_exponent=ap["exponent"][c_i], aperiodic_offset=ap["offset"][c_i],
                       aperiodic_knee_hz=ap["knee_hz"][c_i], alpha_peak_hz=ap["alpha_cf"][c_i],
                       alpha_peak_power=ap["alpha_pw"][c_i])
    chan_df = pd.DataFrame(chan.values())
    row.update(chan_df.drop(columns="channel").mean(numeric_only=True).to_dict())
    row["snr_broadband_dB"] = float(np.nanmean(snr_broadband(psd_sub, freqs, SNR_SIGNAL, SNR_NOISE)))
    row.update(rms_uV=amp["stats"]["rms_uV"], ks_D_vs_gauss=amp["stats"]["ks_D_vs_gauss"])
    # per-epoch stationarity p (broadband) is too big to keep: only the histogram of p
    pst_bb = stationarity_test(band_filter(x, sf, BROADBAND, d["runs"], cfg.PAD_S), STAT_N_SEGMENTS)
    detail = dict(freqs=freqs, psd_mean=psd_sub.mean(0), psd_ch=psd_sub.astype(np.float32),
                  hist_z=amp["hist_z"], stat_p_hist=np.histogram(pst_bb.ravel(), bins=50, range=(0, 1))[0],
                  ap_fit_freqs=ap["fit_freqs"], ap_fit=np.nanmean(ap["ap_fit"], 0), ap_method=ap["method"],
                  ch_names=ch)
    return dict(row=row, chans=chan_df.to_dict("records"), detail=detail)


# ================================================================== TABLES
def long_tables(res):
    subj = pd.DataFrame([dict(condition=c, subject=s, **r["row"]) for c, s, r in res])
    chan = pd.DataFrame([dict(condition=c, subject=s, **ch) for c, s, r in res for ch in r["chans"]])
    long = pd.concat([ps.wide_to_long(subj, m, cfg.BAND_ORDER).set_index(["condition", "subject", "band"])
                      for m in BAND_INDICES], axis=1).reset_index()
    return subj, chan, long


def index_tables(res):
    """(long table, {index: label}, bands) for the drug x time comparisons."""
    subj, _, long = long_tables(res)
    return [(long, BAND_INDICES, cfg.BAND_ORDER), (subj, SCALAR_INDICES, None)]


def tables_for(res, conds):
    subj, chan, long = long_tables(res)
    tests_b = ps.index_band_tests(long, list(BAND_INDICES), conds, cfg.BAND_ORDER)
    tests_s = ps.index_band_tests(subj, list(SCALAR_INDICES), conds)
    if len(tests_s):                                  # FDR across the scalar indices for their figure
        from descriptive_analysis.summary_stats import fdr_bh
        ok = tests_s.p.notna()
        tests_s.loc[ok, "p_fdr_bands"] = fdr_bh(tests_s.loc[ok, "p"].values)
    tests = pd.concat([tests_b, tests_s], ignore_index=True)
    ok = tests.p.notna()
    from descriptive_analysis.summary_stats import fdr_bh
    tests["p_fdr_all"] = np.nan
    tests.loc[ok, "p_fdr_all"] = fdr_bh(tests.loc[ok, "p"].values)
    summ = pd.concat([ps.summary_table(long, list(BAND_INDICES), conds, cfg.BAND_ORDER),
                      ps.summary_table(subj, list(SCALAR_INDICES), conds)], ignore_index=True)
    return subj, chan, long, tests, tests_b, tests_s, summ


# ================================================================== FIGURES
def curves_df(res, conds, key, x_new=None, x_key="freqs", transform=None):
    """{cond: DataFrame subjects x points} of detail[key] (interpolated on x_new if given)."""
    out = {c: {} for c in conds}
    for c, s, r in res:
        y = np.asarray(r["detail"][key], float)
        if x_new is not None:
            y = np.interp(x_new, r["detail"][x_key], y)
        out[c][s] = transform(y) if transform else y
    return {c: pd.DataFrame(v).T for c, v in out.items()}


def fig_psd(res, conds, labels, out, title):
    P = curves_df(res, conds, "psd_mean", COMMON_F, transform=lambda y: 10 * np.log10(y))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    pp.plot_curves(axes[0], COMMON_F, P, conds, cfg.COLORS, labels)
    axes[0].set(xlabel="Frequency (Hz)", ylabel="PSD (dB re 1 µV²/Hz)", title="Mean PSD (all channels)")
    axes[0].legend(fontsize=BASE_FONT - 4)
    for b, (lo, hi) in cfg.BANDS.items():
        axes[0].axvline(hi, color="0.85", lw=0.8, zorder=0)
    # difference from the first condition, per subject
    ref = P[conds[0]]
    D = {c: (P[c] - ref).dropna() for c in conds[1:]}
    pp.plot_curves(axes[1], COMMON_F, D, conds[1:], cfg.COLORS, labels, test=False)
    axes[1].axhline(0, color="k", lw=1)
    axes[1].set(xlabel="Frequency (Hz)", ylabel=f"PSD change vs {labels[conds[0]].replace(chr(10), ' ')} (dB)",
                title="Within-subject change")
    axes[1].legend(fontsize=BASE_FONT - 4)
    fig.suptitle(title + " — power spectrum")
    fig.text(0.5, -0.02, pp.CURVE_NOTE, ha="center", fontsize=BASE_FONT - 4)
    _save(fig, out)


def fig_psd_subjects(res, conds, labels, out):
    subs = list(dict.fromkeys(s for _, s, _ in res))
    ncol = 6
    nrow = int(np.ceil(len(subs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.8 * nrow), squeeze=False, sharex=True)
    for ax, s in zip(axes.ravel(), subs):
        for c, s2, r in res:
            if s2 != s:
                continue
            f = r["detail"]["freqs"]
            m = f <= FMAX_PLOT
            ax.plot(f[m], 10 * np.log10(r["detail"]["psd_mean"][m]), color=cfg.COLORS[c], lw=1.4,
                    label=labels[c].replace("\n", " "))
        ax.set_title(s, fontsize=BASE_FONT - 2)
    for ax in axes.ravel()[len(subs):]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=8)
    fig.supxlabel("Frequency (Hz)")
    fig.supylabel("PSD (dB)")
    fig.suptitle("PSD of every subject (mean over channels)")
    _save(fig, out)


def fig_amplitude(res, conds, labels, subj, tests, out, title):
    x = res[0][2]["detail"]["hist_z"]["centers"]
    H = {c: pd.DataFrame({s: r["detail"]["hist_z"]["density"] for c2, s, r in res if c2 == c}).T for c in conds}
    fig = plt.figure(figsize=(17, 5.4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.7, 1, 1, 1])
    ax = fig.add_subplot(gs[0])
    pp.plot_curves(ax, x, H, conds, cfg.COLORS, labels, logy=True, test=False)
    from scipy import stats
    ax.plot(x, stats.norm.pdf(x), "k--", lw=1.2, label="N(0,1)")
    ax.set(xlabel="z-scored amplitude (SD)", ylabel="Density", ylim=(1e-6, 1), xlim=(-6, 6),
           title="Raw signal amplitude distribution")
    ax.legend(fontsize=BASE_FONT - 5)
    for k, m in enumerate(["excess_kurtosis_broadband", "skew_broadband", "ks_D_vs_gauss"]):
        a = fig.add_subplot(gs[k + 1])
        idx, band = (m.rsplit("_", 1) if m.endswith("broadband") else (m, None))
        t = pp._test_lookup(tests, idx, band)
        pp.paired_box(a, subj, m, conds, cfg.COLORS, labels, t,
                      {"excess_kurtosis_broadband": "excess kurtosis (1-45 Hz)", "skew_broadband": "skewness (1-45 Hz)",
                       "ks_D_vs_gauss": "KS D vs Gaussian (raw)"}[m])
    fig.suptitle(title + " — amplitude distribution (deviation from Gaussianity)")
    fig.text(0.5, -0.03, pp.STATS_NOTE, ha="center", fontsize=BASE_FONT - 5)
    _save(fig, out)


def fig_stationarity(res, conds, labels, subj, tests, out, title):
    fig, axes = plt.subplots(1, 4, figsize=(19, 5.2), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.4]))
    for ax, (m, lab) in zip(axes[:3], [("n_epochs", "Epochs per subject"),
                                       ("log10_var_broadband", "log10 variance 1-45 Hz (µV²)"),
                                       ("acfw_ms_broadband", "ACF width 1-45 Hz (ms)")]):
        idx, band = (m.rsplit("_", 1) if m.endswith("broadband") else (m, None))
        pp.paired_box(ax, subj, m, conds, cfg.COLORS, labels, pp._test_lookup(tests, idx, band), lab)
    ax = axes[3]
    for c in conds:
        h = np.sum([r["detail"]["stat_p_hist"] for c2, s, r in res if c2 == c], 0)
        cdf = np.concatenate([[0], np.cumsum(h) / h.sum()])
        ax.step(np.linspace(0, 1, 51), cdf, where="post", color=cfg.COLORS[c], lw=2.2,
                label=labels[c].replace("\n", " "))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="stationary (uniform p)")
    ax.axvline(STAT_ALPHA, color="0.6", ls=":")
    ax.set(xlabel="Reverse Arrangement Test p (epoch × channel)", ylabel="Cumulative probability",
           title="Stationarity (1-45 Hz)")
    ax.legend(fontsize=BASE_FONT - 5)
    fig.suptitle(title + " — data amount, variance, autocorrelation and stationarity")
    fig.text(0.5, -0.03, pp.STATS_NOTE, ha="center", fontsize=BASE_FONT - 5)
    _save(fig, out)


def fig_aperiodic(res, conds, labels, subj, tests, out, title):
    panels = [("aperiodic_exponent", "aperiodic exponent"), ("aperiodic_knee_hz", "knee frequency (Hz)"),
              ("alpha_peak_hz", "individual alpha peak (Hz)")]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.4), gridspec_kw=dict(width_ratios=[1.6, 1, 1, 1]))
    ax = axes[0]
    for c in conds:
        items = [r["detail"] for c2, s, r in res if c2 == c]
        f = items[0]["freqs"]
        m = (f >= 1) & (f <= 45)
        ax.loglog(f[m], np.mean([d["psd_mean"][m] for d in items], 0), color=cfg.COLORS[c], lw=2,
                  label=labels[c].replace("\n", " "))
        ff = items[0]["ap_fit_freqs"]
        ax.loglog(ff, 10 ** np.nanmean([d["ap_fit"] for d in items], 0), color=cfg.COLORS[c], ls="--", lw=2)
    ax.axvspan(*APERIODIC_RANGE, color="0.93", zorder=0)
    ax.set(xlabel="Frequency (Hz)", ylabel="PSD (µV²/Hz)", title=f"Mean PSD (solid) and aperiodic fit (dashed)\n"
           f"FOOOF {APERIODIC_MODE}, {APERIODIC_RANGE[0]:g}-{APERIODIC_RANGE[1]:g} Hz")
    ax.legend(fontsize=BASE_FONT - 4)
    for a, (m, lab) in zip(axes[1:], panels):
        pp.paired_box(a, subj, m, conds, cfg.COLORS, labels, pp._test_lookup(tests, m, None), lab)
    fig.suptitle(title + " — aperiodic (1/f) component and alpha peak")
    fig.text(0.5, -0.03, pp.STATS_NOTE.replace("across the bands of the index", "across the scalar indices"),
             ha="center", fontsize=BASE_FONT - 5)
    _save(fig, out)


def make_figures(res, conds, subj, chan, long, tests, tests_b, tests_s, out, analysis):
    labels, title = cfg.labels_for(analysis), cfg.ANALYSIS_TITLES[analysis]
    fig_dir, box_dir = out / "figures", out / "boxplots_per_index"
    fig_dir.mkdir(exist_ok=True)
    box_dir.mkdir(exist_ok=True)
    fig_psd(res, conds, labels, fig_dir / "fig01_psd.png", title)
    fig_psd_subjects(res, conds, labels, fig_dir / "fig02_psd_each_subject.png")
    pp.plot_index_grid(long, [("log10_abs", "log10 abs power"), ("rel", "relative power")], cfg.BAND_ORDER,
                       conds, cfg.COLORS, tests_b, fig_dir / "fig03_band_power.png", labels,
                       title + " — band power")
    fig_amplitude(res, conds, labels, subj, tests, fig_dir / "fig04_amplitude_distribution.png", title)
    fig_stationarity(res, conds, labels, subj, tests, fig_dir / "fig05_epochs_variance_acfw_stationarity.png", title)
    fig_aperiodic(res, conds, labels, subj, tests_s, fig_dir / "fig06_aperiodic_alpha_peak.png", title)
    pp.plot_index_grid(long, [(m, l) for m, l in BAND_INDICES.items() if m not in ("log10_abs", "rel")],
                       cfg.BAND_ORDER, conds, cfg.COLORS, tests_b, fig_dir / "fig07_signal_measures_per_band.png",
                       labels, title + " — signal measures per band")
    topo = [(f"log10_abs_{b}", f"log10 power {b}") for b in cfg.BAND_ORDER] + \
           [(f"rel_{b}", f"rel. power {b}") for b in cfg.BANDS] + \
           [("aperiodic_exponent", "aperiodic exponent"), ("alpha_peak_hz", "alpha peak (Hz)")]
    ctests = ps.channel_tests(chan, [m for m, _ in topo], conds)
    pp.plot_topomaps(chan, topo[:6], conds, cfg.COLORS, ctests, fig_dir / "fig08_topo_power.png", labels,
                     title + " — topography of log10 band power")
    pp.plot_topomaps(chan, topo[6:], conds, cfg.COLORS, ctests, fig_dir / "fig09_topo_relpower_aperiodic.png",
                     labels, title + " — topography of relative power, aperiodic exponent, alpha peak")
    pp.plot_summary_heatmap(tests_b, cfg.BAND_ORDER, conds, fig_dir / "fig10_summary_heatmap_bands.png", labels,
                            BAND_INDICES, title + " — summary of the band indices")
    pp.plot_summary_heatmap(tests_s, [""], conds, fig_dir / "fig11_summary_heatmap_scalar.png", labels,
                            SCALAR_INDICES, title + " — summary of the whole-spectrum indices")
    for m, lab in BAND_INDICES.items():
        pp.plot_index_bands(long, m, lab, cfg.BAND_ORDER, conds, cfg.COLORS, tests_b, box_dir / f"{m}.png", labels)
    for m, lab in SCALAR_INDICES.items():
        pp.plot_index_single(subj, m, lab, conds, cfg.COLORS, tests_s, box_dir / f"{m}.png", labels)
    return ctests


def main():
    paper_style()
    results = runner.run_all("psychedelics.MAIN_AYA_STATS", MODULE, params())
    for analysis, conds in cfg.ANALYSES.items():
        res, subs = runner.analysis_results(results, analysis)
        out = runner.analysis_dir(analysis, MODULE)
        print(f"\n== {analysis}: {len(subs)} subjects -> {out}")
        subj, chan, long, tests, tests_b, tests_s, summ = tables_for(res, conds)
        ctests = make_figures(res, conds, subj, chan, long, tests, tests_b, tests_s, out, analysis)
        runner.save_tables_excel(dict(tests=tests, summary=summ, subject_metrics=subj, channel_tests=ctests,
                                      channel_metrics=chan,
                                      parameters=pd.DataFrame([dict(parameter=k, value=str(v))
                                                               for k, v in params().items()])),
                                 out / "stats_results.xlsx")
        sig = tests[tests.p_fdr_bands < cfg.ALPHA]
        print(f"   Friedman FDR q < {cfg.ALPHA}: " + (", ".join(f"{r.index}{'_' + r.band if r.band else ''}"
                                                       for r in sig.itertuples()) or "none"))

    for comp in cfg.COMPARISONS:
        res, subs = runner.comparison_results(results, comp)
        print(f"\n== {comp}: {len(subs)} subjects with both ceremonies")
        compare_drug.run(index_tables(res), comp, MODULE, "stats_aya_vs_placebo.xlsx")


if __name__ == "__main__":
    main()
