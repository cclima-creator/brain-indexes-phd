# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 16:14:26 2026

@author: Chiara Clima
"""

"""Grafici database IPS: legge results.pkl prodotto da MAIN_IPS.py (nessun ricalcolo dell'analisi)."""
import sys
from pathlib import Path

# cartella principale del repository (brain-indexes-phd/) nel path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from photosensitive import config_ips as cfg
from photosensitive.report import FALLBACK_DIR, load_results
from descriptive_analysis import plots
from descriptive_analysis.plots import (_rgba, _save, _need_plotly, _label, HAVE_PLOTLY,
                                        PLOTLY_FONT)

import numpy as np
import matplotlib.pyplot as plt
if HAVE_PLOTLY:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots


# ------------------------------------------------------------------ SNR
SNR_COLS = (("snr_h1_roi_dB", "Fundamental (f)", 1), ("snr_h2_roi_dB", "Second harmonic (2f)", 2))


def plot_snr_vs_stim_freq(snr_freq, colors, out_png, out_html=None, cols=SNR_COLS, fmax=45.0):
    """
    snr_freq: DataFrame con group, subject, stim_freq e le colonne SNR (media per frequenza).
    Per l'armonica h si mostrano solo gli stimoli con h * f <= fmax (sopra e' rumore).
    """
    sub = {col: snr_freq[snr_freq.stim_freq * h <= fmax] for col, _, h in cols}
    fig, axes = plt.subplots(1, len(cols), figsize=(7 * len(cols), 5.5), sharey=True,
                             squeeze=False)
    for ax, (col, ttl, _) in zip(axes[0], cols):
        sf = sub[col]
        for g, c in colors.items():
            piv = sf[sf.group == g].pivot(index="subject", columns="stim_freq",
                                                      values=col)
            if piv.empty:
                continue
            for _, r in piv.iterrows():
                ax.plot(piv.columns, r.values, color=c, lw=0.7, alpha=0.3)
            n = piv.count()
            sem = piv.std(ddof=1) / np.sqrt(n)
            ok = (n >= 2).values
            ax.errorbar(piv.columns[ok], piv.mean()[ok], yerr=sem[ok], color=c, lw=2.5,
                        marker="o", ms=5, capsize=3, label=f"{_label(g)} (n = {piv.shape[0]})")
        ax.axhline(0, color="k", lw=1, ls="--")
        ax.set(title=ttl, xlabel="Stimulation frequency (Hz)", xlim=(0, fmax))
    axes[0][0].set_ylabel("SNR, ROI (dB)")
    axes[0][0].legend()
    fig.suptitle("Narrow-band SNR: power at stimulation bin / mean of neighbouring bins")
    _save(fig, out_png)

    if out_html and _need_plotly():
        fs = make_subplots(rows=1, cols=len(cols), shared_yaxes=True,
                           subplot_titles=[t for _, t, _ in cols])
        for j, (col, _, _) in enumerate(cols, start=1):
            for (g, sid), d in sub[col].groupby(["group", "subject"]):
                d = d.sort_values("stim_freq")
                fs.add_trace(go.Scatter(
                    x=d["stim_freq"], y=d[col], mode="lines+markers", name=sid,
                    legendgroup=sid, showlegend=(j == 1), marker=dict(size=5),
                    line=dict(color=_rgba(colors[g], 0.45), width=1.3),
                    hovertemplate="stim %{x:g} Hz<br>SNR %{y:.1f} dB<extra>"
                                  + sid + " | " + _label(g) + "</extra>"), row=1, col=j)
            for g, c in colors.items():
                mm = sub[col][sub[col].group == g].groupby("stim_freq")[col].mean()
                fs.add_trace(go.Scatter(
                    x=mm.index, y=mm.values, mode="lines+markers", name=f"MEAN {_label(g)}",
                    legendgroup=f"MEAN {g}", showlegend=(j == 1),
                    line=dict(color=_rgba(c, 1), width=4), marker=dict(size=7)), row=1, col=j)
        fs.update_layout(template="simple_white", hovermode="closest", font=PLOTLY_FONT,
                         title="ROI SNR at the stimulation frequency (click legend to hide/show)")
        fs.update_xaxes(title_text="Stimulation frequency (Hz)", range=[0, fmax])
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
    fmax = cfg.PSD_PLOT_FMAX
    plots.paper_style()

    plots.plot_psd_all_subjects(store, colors, out / "fig1_psd_all_subjects.png",
                                out / "fig1_psd_all_subjects_interactive.html",
                                cfg.COMMON_F, fmax)
    plots.plot_psd_each_subject(store, colors, out / "fig2_psd_each_subject.png", fmax)
    plots.plot_epoch_maps(store, colors, out / "fig3_epoch_maps.png", fmax)
    plot_snr_vs_stim_freq(snr_freq, colors, out / "fig4_snr_vs_stim_freq.png",
                          out / "fig4_snr_vs_stim_freq_interactive.html", fmax=fmax)
    t5 = plots.plot_band_hist_gauss(long, subj, cfg.BANDS, colors,
                                    out / "fig5_hist_gauss_bands.png")
    t6 = plots.plot_relative_power_boxplot(subj, cfg.BANDS, colors,
                                           out / "fig6_relative_power_boxplot.png")
    for name, t in (("fig5 log10 potenza", t5), ("fig6 potenza relativa", t6)):
        if len(t):
            print(f"\nTest tra gruppi, {name}:")
            print(t[["test", "p", "p_fdr"]].round(4).to_string())
    for g, c in colors.items():
        r = plots.plot_amplitude_z_group(store, g, c, out / f"fig8_amplitude_zscore_{g}.png")
        if r is not None:
            print(f"\nfig8 {g}: gaussiana di fit mu = {r['mu']:.3f}, sigma = {r['sigma']:.3f}")
            for t in r["tail"]:
                print(f"  oltre ±{t['sd']} SD: dati {t['data_pct']:.4g}%  gaussiana {t['gauss_pct']:.4g}%"
                      f"  rapporto {t['ratio']:.3g}")
    print("\nGrafici salvati in:", out)


if __name__ == "__main__":
    main()
