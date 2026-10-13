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
from descriptive_analysis.plots import _rgba, _save, _need_plotly, HAVE_PLOTLY

import numpy as np
import matplotlib.pyplot as plt
if HAVE_PLOTLY:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots


# ------------------------------------------------------------------ SNR
SNR_COLS = (("snr_h1_roi_dB", "Fondamentale (f)"), ("snr_h2_roi_dB", "Seconda armonica (2f)"))


def plot_snr_vs_stim_freq(snr_freq, colors, out_png, out_html=None, cols=SNR_COLS):
    """snr_freq: DataFrame con group, subject, stim_freq e le colonne SNR (media per frequenza)."""
    fig, axes = plt.subplots(1, len(cols), figsize=(6.5 * len(cols), 4.8), sharey=True,
                             squeeze=False)
    for ax, (col, ttl) in zip(axes[0], cols):
        for g, c in colors.items():
            piv = snr_freq[snr_freq.group == g].pivot(index="subject", columns="stim_freq",
                                                      values=col)
            if piv.empty:
                continue
            for _, r in piv.iterrows():
                ax.plot(piv.columns, r.values, color=c, lw=0.6, alpha=0.3)
            n = piv.count()
            sem = piv.std(ddof=1) / np.sqrt(n)
            ok = (n >= 2).values
            ax.errorbar(piv.columns[ok], piv.mean()[ok], yerr=sem[ok], color=c, lw=2,
                        marker="o", ms=3, capsize=2, label=f"{g} (n={piv.shape[0]})")
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set(title=ttl, xlabel="Frequenza di stimolazione (Hz)")
    axes[0][0].set_ylabel("SNR ROI (dB)")
    axes[0][0].legend()
    fig.suptitle("SNR a banda stretta: potenza al bin di stimolo / media dei bin vicini")
    _save(fig, out_png)

    if out_html and _need_plotly():
        fs = make_subplots(rows=1, cols=len(cols), shared_yaxes=True,
                           subplot_titles=[t for _, t in cols])
        for j, (col, _) in enumerate(cols, start=1):
            for (g, sid), d in snr_freq.groupby(["group", "subject"]):
                d = d.sort_values("stim_freq")
                fs.add_trace(go.Scatter(
                    x=d["stim_freq"], y=d[col], mode="lines+markers", name=sid,
                    legendgroup=sid, showlegend=(j == 1), marker=dict(size=5),
                    line=dict(color=_rgba(colors[g], 0.45), width=1.3),
                    hovertemplate="stim %{x:g} Hz<br>SNR %{y:.1f} dB<extra>"
                                  + sid + " | " + g + "</extra>"), row=1, col=j)
            for g, c in colors.items():
                mm = snr_freq[snr_freq.group == g].groupby("stim_freq")[col].mean()
                fs.add_trace(go.Scatter(
                    x=mm.index, y=mm.values, mode="lines+markers", name=f"MEDIA {g}",
                    legendgroup=f"MEDIA {g}", showlegend=(j == 1),
                    line=dict(color=_rgba(c, 1), width=4), marker=dict(size=7)), row=1, col=j)
        fs.update_layout(template="plotly_white", hovermode="closest",
                         title="SNR ROI alla frequenza di stimolazione (click in legenda = nascondi/mostra)")
        fs.update_xaxes(title_text="Frequenza di stimolazione (Hz)")
        fs.update_yaxes(title_text="SNR (dB)", row=1, col=1)
        fs.write_html(out_html, include_plotlyjs=True)


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
    plot_snr_vs_stim_freq(snr_freq, colors, out / "fig4_snr_vs_stim_freq.png",
                          out / "fig4_snr_vs_stim_freq_interactive.html")
    plots.plot_band_hist_gauss(long, cfg.BANDS, colors, out / "fig5_hist_gauss_bands.png")
    plots.plot_relative_power_boxplot(subj, cfg.BANDS, colors,
                                      out / "fig6_relative_power_boxplot.png")
    plots.plot_amplitude_hist_subjects(store, colors, out / "fig7_amplitude_hist_subjects.png")
    plots.plot_amplitude_z_overlay(store, colors, out / "fig8_amplitude_zscore_overlay.png")
    print("Grafici salvati in:", out)


if __name__ == "__main__":
    main()