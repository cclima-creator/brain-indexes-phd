# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 16:14:40 2026

@author: Chiara Clima
"""

"""Tutti i grafici. Ogni funzione riceve dati e percorsi, non calcola nulla di nuovo."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAVE_PLOTLY = True
except ImportError:
    HAVE_PLOTLY = False


def _rgba(color, alpha):
    r, g, b = mcolors.to_rgb(color)
    return f"rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},{alpha})"


def _grid(n, ncols=5, h=2.6, w=3.4):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    return fig, axes.ravel()


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _need_plotly():
    if not HAVE_PLOTLY:
        print("plotly non installato: salto il grafico interattivo (pip install plotly)")
    return HAVE_PLOTLY


# ------------------------------------------------------------------ PSD
def plot_psd_all_subjects(store, colors, out_png, out_html=None, common_f=None):
    """PSD di ogni soggetto (linee sottili) + media di gruppo ± SEM; html interattivo opzionale."""
    if common_f is None:
        common_f = np.arange(1.0, 45.0, 0.1)
    means = {}
    fig, ax = plt.subplots(figsize=(9, 6))
    for s in store:
        ax.loglog(s["f"], s["psd"], color=colors[s["group"]], lw=0.6, alpha=0.4)
    for g, c in colors.items():
        A = np.array([np.interp(common_f, s["f"], s["psd"]) for s in store if s["group"] == g])
        if len(A) == 0:
            continue
        m = A.mean(0)
        sem = A.std(0, ddof=1) / np.sqrt(len(A)) if len(A) > 1 else 0 * m
        means[g] = m
        ax.loglog(common_f, m, color=c, lw=2.5, label=f"{g} (n={len(A)})")
        ax.fill_between(common_f, np.clip(m - sem, 1e-12, None), m + sem, color=c, alpha=0.2)
    ax.set(xlabel="Frequenza (Hz)", ylabel="PSD (uV²/Hz)",
           title="PSD per soggetto (linee sottili) e media di gruppo ± SEM")
    ax.legend()
    _save(fig, out_png)

    if out_html and _need_plotly():
        fi = go.Figure()
        for s in store:
            fi.add_trace(go.Scatter(
                x=s["f"], y=s["psd"], mode="lines", name=s["subject"], legendgroup=s["group"],
                line=dict(color=_rgba(colors[s["group"]], 0.45), width=1.3),
                hovertemplate="%{x:.2f} Hz<br>%{y:.3g} uV²/Hz<extra>"
                              + s["subject"] + " | " + s["group"] + "</extra>"))
        for g, m in means.items():
            fi.add_trace(go.Scatter(x=common_f, y=m, mode="lines", name=f"MEDIA {g}",
                                    line=dict(color=_rgba(colors[g], 1), width=4)))
        fi.update_layout(template="plotly_white", hovermode="closest",
                         xaxis=dict(type="log", title="Frequenza (Hz)"),
                         yaxis=dict(type="log", title="PSD (uV²/Hz)"),
                         title="PSD per soggetto (doppio click in legenda = isola un soggetto)")
        fi.write_html(out_html, include_plotlyjs=True)


def plot_psd_each_subject(store, colors, out_png):
    fig, axes = _grid(len(store))
    for ax, s in zip(axes, store):
        ax.loglog(s["f"], s["psd"], color=colors[s["group"]], lw=1)
        ax.set_title(s["subject"][:22], fontsize=7, color=colors[s["group"]])
        ax.tick_params(labelsize=6)
    fig.suptitle("PSD per soggetto (media su epoche e canali)")
    _save(fig, out_png)


def plot_epoch_maps(store, colors, out_png, fmax_plot=45.0):
    """Mappa epoca x frequenza (ROI), epoche ordinate per frequenza di stimolo."""
    items = [s for s in store if s.get("fstim") is not None]
    if not items:
        return
    fig, axes = _grid(len(items))
    for ax, s in zip(axes, items):
        order = np.argsort(np.where(np.isfinite(s["fstim"]), s["fstim"], np.inf))
        H = s["heat"][order]
        fm = s["f"] <= fmax_plot
        ax.imshow(H[:, fm], aspect="auto", origin="upper", cmap="viridis",
                  extent=[s["f"][fm][0], s["f"][fm][-1], len(order), 0],
                  vmin=np.percentile(H[:, fm], 5), vmax=np.percentile(H[:, fm], 99.5))
        ys = np.arange(len(order)) + 0.5
        ax.scatter(s["fstim"][order], ys, s=6, c="white", marker="|", linewidths=0.8)
        ax.set_yticks(ys[::2])
        ax.set_yticklabels([f"{v:g}" for v in s["fstim"][order][::2]], fontsize=5)
        ax.set_title(s["subject"][:22], fontsize=7, color=colors[s["group"]])
        ax.tick_params(axis="x", labelsize=6)
    fig.suptitle("log10 PSD (ROI): y = frequenza di stimolo (tacca bianca = attesa), x = Hz")
    _save(fig, out_png)


# ------------------------------------------------------------------ BANDE
def plot_band_hist_gauss(long, bands, colors, out_png):
    """Istogrammi log10 potenza per banda (canali x soggetti) con fit gaussiano."""
    fig, axes = plt.subplots(1, len(bands), figsize=(4 * len(bands), 3.8), squeeze=False)
    for ax, b in zip(axes[0], bands):
        for g, c in colors.items():
            v = long[(long.group == g) & (long.band == b)].log10_abs.dropna()
            if len(v) < 3:
                continue
            mu, sd = stats.norm.fit(v)
            xs = np.linspace(v.min(), v.max(), 200)
            ax.hist(v, bins=30, density=True, alpha=0.4, color=c)
            ax.plot(xs, stats.norm.pdf(xs, mu, sd), color=c, lw=2,
                    label=f"{g[:5]}.. μ={mu:.2f} σ={sd:.2f}")
        ax.set(title=b, xlabel="log10 potenza (uV²)")
        ax.legend(fontsize=7)
    axes[0][0].set_ylabel("Densità")
    fig.suptitle("Distribuzione della potenza per banda (canali x soggetti) con fit gaussiano")
    _save(fig, out_png)


def plot_relative_power_boxplot(subj, bands, colors, out_png):
    fig, ax = plt.subplots(figsize=(9, 5))
    w = 0.8 / max(len(colors), 1)
    for k, (g, c) in enumerate(colors.items()):
        data = [subj[(subj.group == g) & (subj.band == b)].rel.values for b in bands]
        pos = np.arange(len(bands)) + (k - (len(colors) - 1) / 2) * w
        bp = ax.boxplot(data, positions=pos, widths=w * 0.9, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(c)
            patch.set_alpha(0.5)
        ax.plot([], [], color=c, label=g)
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels(list(bands))
    ax.set(ylabel="Potenza relativa", title="Potenza relativa per banda (soggetti)")
    ax.legend()
    _save(fig, out_png)


# ------------------------------------------------------------------ AMPIEZZE
def plot_amplitude_hist_subjects(store, colors, out_png):
    """Istogramma delle ampiezze (uV) di ogni soggetto con la gaussiana di pari media e SD."""
    fig, axes = _grid(len(store))
    for ax, s in zip(axes, store):
        h = s["amp"]["hist_raw"]
        width = h["centers"][1] - h["centers"][0]
        ax.bar(h["centers"], h["density"], width=width, color=colors[s["group"]], alpha=0.5)
        ax.plot(h["centers"], stats.norm.pdf(h["centers"], h["mu"], h["sigma"]), "k", lw=1)
        ax.set_title(f'{s["subject"][:22]}\ncurtosi={s["amp"]["stats"]["excess_kurtosis"]:.1f}',
                     fontsize=7, color=colors[s["group"]])
        ax.tick_params(labelsize=6)
    fig.suptitle("Distribuzione delle ampiezze (uV) e fit gaussiano (linea nera)")
    _save(fig, out_png)


def plot_amplitude_z_overlay(store, colors, out_png):
    """Ampiezze z-scored per canale, un soggetto per linea, asse y logaritmico, vs N(0,1)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in store:
        h = s["amp"]["hist_z"]
        d = np.where(h["density"] > 0, h["density"], np.nan)
        ax.semilogy(h["centers"], d, color=colors[s["group"]], lw=0.8, alpha=0.5)
    xs = np.linspace(-6, 6, 400)
    ax.semilogy(xs, stats.norm.pdf(xs), "k--", lw=2, label="N(0,1)")
    for g, c in colors.items():
        ax.plot([], [], color=c, label=g)
    ax.set(ylim=(1e-6, 1), xlabel="Ampiezza (z-score per canale)", ylabel="Densità",
           title="Code della distribuzione delle ampiezze vs gaussiana")
    ax.legend()
    _save(fig, out_png)