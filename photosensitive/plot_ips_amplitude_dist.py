# -*- coding: utf-8 -*-
"""
Grafici della distribuzione delle ampiezze di tutti i campioni (MAIN_IPS_AMPLITUDE_DIST.py).
Rilegge gli stessi campioni |z| (stesso seed) e i risultati dei fit da amplitude_distribution.xlsx;
le curve dei modelli sono rifittate sopra lo stesso xmin con il pacchetto powerlaw (solo per
disegnarle: nessun bootstrap, i p-value e il miglior modello restano quelli della tabella figE1).

  figE2_amplitude_ccdf         per banda: P(|z| >= x) in log-log, una linea per soggetto, curva del
                               gruppo (tutti i campioni dei soggetti insieme) e |z| di un rumore
                               gaussiano (tratteggiato); linee verticali = xmin mediano del gruppo
  figE3_amplitude_fits_<banda> un pannello per soggetto: coda sopra xmin (punti) con power law,
                               power law troncata, esponenziale e lognormale; nel titolo il miglior
                               modello, alpha e p di Clauset
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from scipy.special import erfc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import powerlaw

from photosensitive import config_ips as base
from photosensitive import config_ips_avalan as av
from photosensitive.io_eeg import discover_files, load_epochs, get_data_uv
from photosensitive.MAIN_IPS import pick_channels
from photosensitive.MAIN_IPS_AVALAN import band_filter
from photosensitive.MAIN_IPS_AMPLITUDE_DIST import amplitude_sample, OUT_NAME, SEED
from photosensitive.plot_ips_avalan import _save
from avalanche_criticality.power_law import _tpl_log_norm_continuous
from descriptive_analysis import plots
from descriptive_analysis.plots import _label

warnings.filterwarnings("ignore")

MODEL_STYLE = {"power_law": ("PL", "k", "-"), "truncated_power_law": ("TPL", "tab:orange", "-"),
               "exponential": ("Exp", "tab:green", "--"), "lognormal": ("LN", "tab:purple", ":"),
               "stretched_exponential": ("SExp", "tab:olive", "-.")}


def ccdf(x):
    """(valori ordinati, P(X >= valore))."""
    s = np.sort(np.asarray(x))
    return s, 1 - np.arange(s.size) / s.size


def _log(ax):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())


def load_samples(channels):
    """Gli stessi campioni |z| dell'analisi: dict (gruppo, soggetto, banda) -> array."""
    out = {}
    for group, folder in av.GROUPS.items():
        for fpath in base.subject_files(group, discover_files(folder)):
            print("  ", fpath.name)
            ep = load_epochs(fpath, base.FIXED_LENGTH_S, base.ANALYSIS_WINDOW)
            if channels is not None:
                ep = pick_channels(ep, channels, base.OCCIPITAL_CHANNELS)
            sf, data = ep.info["sfreq"], get_data_uv(ep)
            for b, rng_b in av.BANDS.items():
                out[(group, fpath.stem, b)], _ = amplitude_sample(band_filter(data, sf, rng_b),
                                                                  np.random.default_rng(SEED))
    return out


def plot_ccdf_groups(samples, fit, bands, colors, out_png):
    """figE2: righe = gruppi, colonne = bande; soggetti sottili, gruppo spesso, gaussiana tratteggiata."""
    groups = [g for g in colors if any(k[0] == g for k in samples)]
    fig, axes = plt.subplots(len(groups), len(bands), figsize=(3.4 * len(bands), 3.4 * len(groups)),
                             squeeze=False, sharex=True, sharey=True)
    xg = np.logspace(-2, np.log10(15), 200)
    for i, g in enumerate(groups):
        for j, b in enumerate(bands):
            ax = axes[i, j]
            subs = [v for (gg, sid, bb), v in samples.items() if gg == g and bb == b]
            for v in subs:
                ax.plot(*ccdf(v), color=colors[g], lw=0.6, alpha=0.35)
            if subs:
                ax.plot(*ccdf(np.concatenate(subs)), color=colors[g], lw=2.5, label=_label(g))
            ax.plot(xg, erfc(xg / np.sqrt(2)), "k--", lw=1.2, label="Gaussian noise")
            xm = fit[(fit.group == g) & (fit.band == b)].xmin.median()
            if np.isfinite(xm):
                ax.axvline(xm, color=colors[g], ls=":", lw=1.2)
            _log(ax)
            ax.set(xlim=(0.05, 15), ylim=(1e-4, 1.2))
            if i == 0:
                ax.set_title(b)
            if i == len(groups) - 1:
                ax.set_xlabel("|z| (SD)")
            if j == 0:
                ax.set_ylabel(f"{_label(g)}\nP(|z| ≥ x)", color=colors[g])
    axes[0, 0].legend(fontsize=plots.BASE_FONT - 4, loc="lower left")
    fig.suptitle("Amplitude |z| of all samples: complementary cumulative distribution per subject (thin), "
                 "group (thick) and Gaussian noise (dashed); dotted = median Clauset xmin (start of the fitted tail)")
    _save(fig, out_png)


def plot_fits_band(samples, fit, band, colors, out_png, ncols=6):
    """figE3: un pannello per soggetto, coda sopra xmin con le quattro curve dei modelli."""
    rows = fit[fit.band == band]
    keys = [(r.group, r.subject) for r in rows.itertuples() if (r.group, r.subject, band) in samples]
    if not keys:
        return
    keys.sort(key=lambda k: list(colors).index(k[0]))
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows), squeeze=False)
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    for ax, (g, sid) in zip(axes.ravel(), keys):
        r = rows[(rows.group == g) & (rows.subject == sid)].iloc[0]
        x = samples[(g, sid, band)]
        xm = r.xmin
        t = x[x >= xm]
        if not np.isfinite(xm) or t.size < 10:
            ax.set_title(f"{sid.split('_')[0]}: no fit", fontsize=plots.BASE_FONT - 4)
            continue
        ax.plot(*ccdf(t), "o", color=colors[g], ms=2.5, alpha=0.6)
        f = powerlaw.Fit(t, xmin=xm, discrete=False, verbose=False)
        xs = np.logspace(np.log10(xm), np.log10(t.max()), 80)
        for m, (lab, col, ls) in MODEL_STYLE.items():
            try:
                if m == "power_law":              # fit di questo repo: il pacchetto (2.0.0) lascia alpha = 3
                    y = (xs / xm) ** (1 - r.alpha)
                elif m == "truncated_power_law":  # idem: fit_truncated_power_law_continuous
                    lz0 = _tpl_log_norm_continuous(r.tpl_alpha, r.tpl_lambda, xm)
                    y = np.exp([_tpl_log_norm_continuous(r.tpl_alpha, r.tpl_lambda, v) - lz0 for v in xs])
                else:
                    y = getattr(f, m).ccdf(xs)
                ax.plot(xs, y, color=col, ls=ls, lw=2.2 if m == r.best_model else 1.0,
                        label=lab + (" (best)" if m == r.best_model else ""))
            except Exception:
                pass
        _log(ax)
        ax.set_ylim(max(0.5 / t.size, 1e-4), 1.2)
        ax.set_title(f"{sid.split('_')[0]}: best {MODEL_STYLE.get(r.best_model, (r.best_model,))[0]}\n"
                     f"xmin {xm:.2f} SD ({100 * r.frac_tail:.0f}%), α {r.alpha:.1f}, p {r.p_gof:.2f}",
                     fontsize=plots.BASE_FONT - 4, color=colors[g])
        ax.legend(fontsize=plots.BASE_FONT - 7, loc="lower left")
    for ax in axes[-1]:
        ax.set_xlabel("|z| (SD)")
    for ax in axes[:, 0]:
        ax.set_ylabel("P(|z| ≥ x | tail)")
    fig.suptitle(f"{band}: tail of the amplitude distribution above the Clauset xmin with the fitted models "
                 "(thick = best model; % = share of samples in the tail; p = Clauset power-law p)")
    _save(fig, out_png)


def main():
    channel_set, channels, out_root = base.choose_channel_set()
    out = av.results_dir(channel_set, out_root) / OUT_NAME
    fit = pd.read_excel(out / "amplitude_distribution.xlsx", sheet_name="distribution_fit")
    print("Rileggo i campioni |z| (stesso seed dell'analisi)")
    samples = load_samples(channels)
    colors = {g: c for g, c in av.COLORS.items() if g in set(fit.group)}
    bands = list(av.BANDS)
    plots.paper_style()
    plot_ccdf_groups(samples, fit, bands, colors, out / "figE2_amplitude_ccdf.png")
    for b in bands:
        plot_fits_band(samples, fit, b, colors, out / f"figE3_amplitude_fits_{b}.png")
    print("Grafici salvati in:", out)


if __name__ == "__main__":
    main()
