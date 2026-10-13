# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 16:14:26 2026

@author: Chiara Clima
"""

"""Grafici database IPS: legge results.pkl prodotto da MAIN_IPS.py (nessun ricalcolo)."""
import sys
from pathlib import Path

# cartella principale del repository (brain-indexes-phd/) nel path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from photosensitive import config_ips as cfg
from photosensitive.report import FALLBACK_DIR, load_results
from descriptive_analysis import plots


def find_results():
    for d in (cfg.OUT_DIR, FALLBACK_DIR):
        if (d / "results.pkl").exists():
            return d
    raise SystemExit("results.pkl non trovato: lancia prima MAIN_IPS.py")


def main():
    out = find_results()
    res = load_results(out / "results.pkl")
    store, long, subj, snr_freq = res["store"], res["long"], res["subj"], res["snr_freq"]
    colors = cfg.COLORS

    plots.plot_psd_all_subjects(store, colors, out / "fig1_psd_all_subjects.png",
                                out / "fig1_psd_all_subjects_interactive.html", cfg.COMMON_F)
    plots.plot_psd_each_subject(store, colors, out / "fig2_psd_each_subject.png")
    plots.plot_epoch_maps(store, colors, out / "fig3_epoch_maps.png")
    plots.plot_snr_vs_stim_freq(snr_freq, colors, out / "fig4_snr_vs_stim_freq.png",
                                out / "fig4_snr_vs_stim_freq_interactive.html")
    plots.plot_band_hist_gauss(long, cfg.BANDS, colors, out / "fig5_hist_gauss_bands.png")
    plots.plot_relative_power_boxplot(subj, cfg.BANDS, colors,
                                      out / "fig6_relative_power_boxplot.png")
    plots.plot_amplitude_hist_subjects(store, colors, out / "fig7_amplitude_hist_subjects.png")
    plots.plot_amplitude_z_overlay(store, colors, out / "fig8_amplitude_zscore_overlay.png")
    print("Grafici salvati in:", out)


if __name__ == "__main__":
    main()