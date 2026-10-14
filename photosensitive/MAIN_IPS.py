# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 16:13:54 2026

@author: Chiara Clima
"""

"""
MAIN database IPS: importa i dati, calcola le feature con i moduli, salva Excel + risultati.
I grafici si fanno dopo con plot_ips.py (senza rifare l'analisi).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# cartella principale del repository (brain-indexes-phd/) nel path, cosi' si possono
# importare sia photosensitive/ che descriptive_analysis/ lanciando questo file direttamente
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from photosensitive import config_ips as cfg
from photosensitive.io_eeg import discover_files, load_epochs, epoch_labels, get_data_uv   ##io_eeg to load data
from photosensitive.report import check_output_dir, save_tables_excel, save_results
from descriptive_analysis.spectral import compute_psd, band_power_table, aperiodic_fit
from descriptive_analysis.snr import snr_broadband, snr_narrowband
from descriptive_analysis.amplitude import amplitude_distribution
from descriptive_analysis.summary_stats import (band_group_summary, compare_groups, epoch_counts,
                                                signal_tables, ks_permutation_test,
                                                channel_group_tests)

warnings.filterwarnings("ignore")


def analyze_subject(epochs, group, sid):
    """Tutte le feature di un soggetto. Usa solo get_data/info/ch_names/events/event_id."""
    sf = epochs.info["sfreq"]
    names = epochs.ch_names
    data = get_data_uv(epochs)                        # (n_ep, n_ch, n_t) uV
    n_ep, n_ch, n_t = data.shape

    freqs, psd = compute_psd(data, sf, cfg.FMIN, cfg.FMAX_PSD, cfg.SEGMENT_S)
    psd_sub = psd.mean(0)                             # (n_ch, n_f) media sulle epoche

    labels = epoch_labels(epochs)
    fstim = np.array([cfg.STIM_FREQ_MAP.get(l, cfg.parse_stim_freq(l)) for l in labels],
                     dtype=float)
    print(f"    etichette: {sorted(set(labels))[:6]}{' ...' if len(set(labels)) > 6 else ''}")

    roi_up = {c.upper() for c in cfg.ROI_CHANNELS}
    roi = [i for i, c in enumerate(names) if c.upper() in roi_up]
    if not roi:
        print("    ATTENZIONE: nessun canale ROI trovato, uso tutti i canali")
        roi = list(range(n_ch))

    # SNR a banda stretta per epoca
    epoch_rows = []
    for k in range(n_ep):
        row = dict(group=group, subject=sid, epoch=k, label=labels[k], stim_freq=fstim[k])
        for h, s in snr_narrowband(psd[k], freqs, fstim[k], cfg.N_HARMONICS,
                                   cfg.NB_GAP, cfg.NB_WIDTH, cfg.FMAX_ANALYSIS).items():
            row[f"snr_h{h}_all_dB"] = np.nanmean(s)
            row[f"snr_h{h}_roi_dB"] = np.nanmean(s[roi])
        epoch_rows.append(row)

    # potenza per banda
    band_df = band_power_table(psd_sub, freqs, names, cfg.BANDS, cfg.BAND_TOTAL)
    band_df.insert(0, "subject", sid)
    band_df.insert(0, "group", group)

    # ampiezze nel tempo
    amp = amplitude_distribution(data)

    # varianza, ACFW e stazionarieta' per epoca e canale
    ec, ch_sig, sig_subj = signal_tables(data, sf, names, labels, fstim, cfg.STAT_N_SEGMENTS,
                                         cfg.ACF_MAX_LAG_S, cfg.STAT_ALPHA)
    # componente aperiodica 1/f per canale (sulla PSD media sulle epoche)
    ap = aperiodic_fit(psd_sub, freqs, cfg.APERIODIC_RANGE)
    ch_sig["aperiodic_exponent"] = ap["exponent"]
    ch_sig["aperiodic_offset"] = ap["offset"]
    ch_sig["aperiodic_r2"] = ap["r2"]
    for df in (ec, ch_sig):
        df.insert(0, "subject", sid)
        df.insert(0, "group", group)

    summary = dict(group=group, subject=sid, sfreq=sf, n_channels=n_ch, n_epochs=n_ep,
                   n_roi_channels=len(roi), epoch_len_s=n_t / sf,
                   snr_broadband_dB=np.nanmean(
                       snr_broadband(psd_sub, freqs, cfg.SNR_SIGNAL, cfg.SNR_NOISE)),
                   **amp["stats"], **sig_subj,
                   aperiodic_exponent=np.nanmean(ap["exponent"]),
                   aperiodic_offset=np.nanmean(ap["offset"]),
                   aperiodic_r2=np.nanmean(ap["r2"]))
    store = dict(group=group, subject=sid, f=freqs, psd=psd_sub.mean(0),
                 heat=np.log10(psd[:, roi, :].mean(1)), fstim=fstim, amp=amp, aperiodic=ap)
    return band_df, epoch_rows, summary, store, ec, ch_sig


def print_snr_summary(subj_meta):
    """Console: SNR per soggetto, media ± SD per gruppo e test tra gruppi (FDR sulle misure di SNR)."""
    cols = ["snr_broadband_dB"] + [c for c in subj_meta.columns
                                    if c.startswith("snr_h") and c.endswith("_roi_dB")]
    cols = [c for c in cols if subj_meta[c].notna().any()]
    print("\n=== SNR per soggetto (dB) ===")
    print("snr_broadband_dB: potenza media 1-45 Hz / potenza media 55-70 Hz (rumore), media sui canali")
    print("snr_hK_roi_dB: SNR a banda stretta alla K-esima armonica dello stimolo (<= 45 Hz), canali ROI")
    print(subj_meta[["group", "subject"] + cols].round(2).to_string(index=False))
    print("\nMedia ± SD per gruppo:")
    for g, d in subj_meta.groupby("group"):
        print(f"  {g} (n={len(d)}): " + ", ".join(
            f"{c} = {d[c].mean():.2f} ± {d[c].std():.2f}" for c in cols))
    t = compare_groups(subj_meta, cols)
    if len(t):
        print("\nTest tra gruppi (p_fdr = FDR sulle misure di SNR qui sopra):")
        print(t.round(4).to_string(index=False))
    else:
        print("\nTest tra gruppi non possibile (servono >= 2 soggetti per gruppo)")


def print_epoch_and_cdf_summary(counts_group, ks_df):
    """Console: epoche disponibili per gruppo e test KS a permutazione sulle CDF."""
    print("\n=== Epoche disponibili per gruppo ===")
    print(counts_group.round(1).to_string(index=False))
    if len(ks_df):
        print("\n=== CDF tra gruppi: Kolmogorov-Smirnov con permutazione dei soggetti ===")
        print(ks_df.round(4).to_string(index=False))


def main():
    out_dir = check_output_dir(cfg.OUT_DIR, cfg.OUTPUT_NAMES)
    print("Cartella di output:", out_dir)

    band_dfs, epoch_rows, subj_rows, store, ec_list, ch_list = [], [], [], [], [], []
    for group, folder in cfg.GROUPS.items():
        files = discover_files(folder)
        print(f"{group}: {len(files)} file")
        for fpath in files:
            print("  ", fpath.name)
            try:
                ep = load_epochs(fpath, cfg.FIXED_LENGTH_S, cfg.ANALYSIS_WINDOW)
                bd, er, sm, st, ec, ch = analyze_subject(ep, group, fpath.stem)
                band_dfs.append(bd); epoch_rows += er; subj_rows.append(sm); store.append(st)
                ec_list.append(ec); ch_list.append(ch)
            except Exception as e:
                print("    ERRORE ->", e)
    if not store:
        raise SystemExit("Nessun file analizzato: controlla i percorsi in config_ips.py")

    long = pd.concat(band_dfs, ignore_index=True)
    epoch_df = pd.DataFrame(epoch_rows)
    subj_meta = pd.DataFrame(subj_rows)
    snr_cols = [c for c in epoch_df.columns if c.startswith("snr_")]

    subj_meta = subj_meta.merge(
        epoch_df.groupby(["group", "subject"])[snr_cols].mean().reset_index(),
        on=["group", "subject"], how="left")
    snr_freq = epoch_df.groupby(["group", "subject", "stim_freq"])[snr_cols].mean().reset_index()
    snr_freq_group = snr_freq.groupby(["group", "stim_freq"])[snr_cols].agg(
        ["mean", "median", "count"])
    snr_freq_group.columns = ["_".join(c) for c in snr_freq_group.columns]

    subj = (long.groupby(["group", "subject", "band"])[["abs_uV2", "rel", "log10_abs"]]
            .mean().reset_index())
    wide = subj.pivot(index=["group", "subject"], columns="band", values=["log10_abs", "rel"])
    wide.columns = [f"{m}_{b}" for m, b in wide.columns]
    wide = wide.reset_index().merge(subj_meta, on=["group", "subject"])
    metrics = [c for c in wide.columns if c.startswith(("log10_abs_", "rel_"))] + \
              ["snr_broadband_dB", "rms_uV", "std_uV", "skew", "excess_kurtosis",
               "frac_beyond_3sd", "ks_D_vs_gauss", "n_epochs", "log10_var", "var_cv_epochs",
               "acfw_ms", "frac_nonstationary", "aperiodic_exponent", "aperiodic_offset"] + snr_cols

    # epoche disponibili
    counts, counts_group = epoch_counts(epoch_df)
    # varianza / ACFW: tutte le epoche x canali, CDF confrontate con KS a permutazione
    ec_all = pd.concat(ec_list, ignore_index=True)
    ks = {m: ks_permutation_test(ec_all, m, n_perm=cfg.KS_N_PERM) for m in ("log10_var", "acfw_ms")}
    ks_df = pd.DataFrame([v for v in ks.values() if v])
    # una riga per soggetto x canale: potenza per banda + misure del segnale (per le topografie)
    chan_bands = long.pivot_table(index=["group", "subject", "channel"], columns="band",
                                  values="log10_abs")
    chan_bands.columns = [f"log10_abs_{b}" for b in chan_bands.columns]
    chan = chan_bands.reset_index().merge(pd.concat(ch_list, ignore_index=True),
                                          on=["group", "subject", "channel"], how="outer")
    topo_metrics = [f"log10_abs_{b}" for b in cfg.BANDS] + ["log10_var", "acfw_ms",
                                                         "aperiodic_exponent"]
    chan_tests = channel_group_tests(chan, topo_metrics)
    print_epoch_and_cdf_summary(counts_group, ks_df)

    tables = {
        "group_summary_bands": band_group_summary(long, subj),
        "group_tests": compare_groups(wide, metrics),
        "subject_level_metrics": subj_meta,
        "snr_by_stim_freq_group": snr_freq_group.reset_index(),
        "epoch_table_snr": epoch_df,
        "band_power_per_subject": subj,
        "band_power_per_channel": long,
        "epoch_counts_group": counts_group,
        "epoch_counts": counts,
        "ks_permutation_tests": ks_df,
        "channel_tests": chan_tests,
        "signal_per_channel": chan,
    }
    if len(ec_all) < 1_000_000:                       # limite di righe di Excel
        tables["signal_per_epoch_channel"] = ec_all
    else:
        print("signal_per_epoch_channel troppo grande per Excel: resta solo in results.pkl")
    print_snr_summary(subj_meta)
    save_tables_excel(tables, out_dir / "risultati_eeg.xlsx", cfg.SHEET_ORDER)
    save_results(dict(store=store, long=long, subj=subj, snr_freq=snr_freq, subj_meta=subj_meta,
                      counts=counts, ec=ec_all, chan=chan, ks=ks),
                 out_dir / "results.pkl")
    print("\nFatto. Excel e results.pkl in:", out_dir)


if __name__ == "__main__":
    main()