# -*- coding: utf-8 -*-
"""
Grafici delle valanghe, database IPS: legge avalanche_results.pkl prodotto da MAIN_IPS_AVALAN.py
(nessun ricalcolo dell'analisi). Figure in OUT_DIR/avalanches (o OUT_DIR/occipital/avalanches).

  figA1_threshold_divergence      z-score per banda vs gaussiana di fit, punto di divergenza e soglia usata
  figA2_ccdf_sizes_durations      CCDF di dimensioni e durate, ogni soggetto, per banda
  figA3_powerlaw_fits_<banda>     PDF di S e T per soggetto con fit principale, range power law e cutoff
  figA4_size_duration_scaling     <S>(T) per soggetto con pendenza 1/(sigma nu z) e pendenza prevista
  figA5_exponents_dcc             tau, alpha, 1/(sigma nu z), DCC per banda e gruppo
  figA6_powerlaw_quality          p-value, decadi del range, MSE, cutoff per banda e gruppo
  figA7_branching_ratio_bins      sigma per durata del time bin (4-18 ms), media ± SEM per gruppo
  figA8_repertoire                repertorio (pattern unici/s e per valanga) e diversita' dei pattern
  figA9_fano                      Fano factor e curva FF(W)
  figA10_summary_heatmap          mediane per gruppo e differenza (effect size) metrica x banda
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from photosensitive import config_ips as base
from photosensitive import config_ips_avalan as cfg
from photosensitive.report import FALLBACK_DIR, load_results
from descriptive_analysis import plots
from descriptive_analysis.plots import _save, _label, _grid, _box_by_group, _stars
from avalanche_criticality.power_law import model_cdf


# ------------------------------------------------------------------ UTILITA'
def _logbin_pdf(x, n_bins=20):
    """PDF di dati interi con bin logaritmici: (centri, densita' per unita' di x)."""
    x = np.asarray(x, dtype=float)
    x = x[x > 0]
    if x.size == 0:
        return np.array([]), np.array([])
    edges = np.unique(np.floor(np.geomspace(x.min(), x.max() + 1, n_bins + 1)))
    if edges.size < 2:
        edges = np.array([x.min(), x.min() + 1])
    counts, _ = np.histogram(x, bins=edges)
    dens = counts / (x.size * np.diff(edges))
    centers = np.sqrt(edges[:-1] * (edges[1:] - 1).clip(min=edges[:-1]))
    ok = counts > 0
    return centers[ok], dens[ok]


def _model_pdf(xs, a, xmin, xmax, frac):
    """PDF discreta della power law su [xmin, xmax], scalata alla frazione di dati nel range."""
    xs = np.asarray(xs, dtype=float)
    p = model_cdf(xs, a, xmin, xmax, True) - model_cdf(xs - 1, a, xmin, xmax, True)
    return frac * p


def _ccdf(x):
    x = np.sort(np.asarray(x))
    return x, 1 - np.arange(x.size) / x.size


def _groups(df):
    return [g for g in cfg.COLORS if g in set(df.group)]


def _band_boxes(metrics, cols, bands, colors, out_png, title):
    """Griglia righe = misure, colonne = bande, box per gruppo con test (riusa plots._box_by_group)."""
    fig, axes = plt.subplots(len(cols), len(bands), figsize=(3.1 * len(bands), 3.4 * len(cols)),
                             squeeze=False)
    for i, (col, ylabel) in enumerate(cols):
        for j, b in enumerate(bands):
            ax = axes[i, j]
            d = metrics[metrics.band == b]
            if col not in d or d[col].notna().sum() == 0:
                ax.axis("off")
                continue
            _box_by_group(ax, d, col, colors, ylabel if j == 0 else "")
            ax.set_title(f"{b}\n{ax.get_title()}", fontsize=plots.BASE_FONT - 1)
    fig.suptitle(title)
    _save(fig, out_png)


# ------------------------------------------------------------------ FIGURE
def plot_threshold_divergence(curves, divergence, bands, colors, out_png):
    """figA1: densita' degli z-score di ogni gruppo vs la sua gaussiana di fit, per banda."""
    fig, axes = plt.subplots(len(colors), len(bands), figsize=(3.3 * len(bands), 3.2 * len(colors)),
                             squeeze=False, sharex=True)
    for i, g in enumerate(colors):
        for j, b in enumerate(bands):
            ax = axes[i, j]
            r = curves.get((g, b))
            if r is None:
                ax.axis("off")
                continue
            c = r["centers"]
            ax.fill_between(c, np.maximum(r["lo"], 1e-7), np.maximum(r["hi"], 1e-7),
                            color=colors[g], alpha=0.25, lw=0)
            ax.semilogy(c, np.maximum(r["density"], 1e-7), color=colors[g], lw=1.5, label="data")
            ax.semilogy(c, np.maximum(r["gauss"], 1e-7), "k--", lw=1, label="Gaussian fit")
            thr = cfg.threshold_for(g, b)
            for s in (-1, 1):
                ax.axvline(s * thr, color="k", lw=1.2, label="threshold used" if s == 1 else None)
                if np.isfinite(r["divergence"]):
                    ax.axvline(s * r["divergence"], color=colors[g], ls=":", lw=1.5,
                               label="divergence" if s == 1 else None)
            ax.set(xlim=(-8, 8), ylim=(1e-6, 1))
            ax.set_title(f"{b}: div. {r['divergence']:.1f} SD, used {thr:g}", fontsize=plots.BASE_FONT - 2)
            if j == 0:
                ax.set_ylabel(f"{_label(g)}\ndensity")
            if i == len(colors) - 1:
                ax.set_xlabel("z-score (SD)")
    axes[0, 0].legend(fontsize=plots.BASE_FONT - 4)
    fig.suptitle("Amplitude distribution vs Gaussian per band: divergence point and binarization threshold")
    _save(fig, out_png)


def plot_ccdf(details, bands, colors, out_png):
    """figA2: CCDF di dimensioni (sopra) e durate (sotto) per ogni soggetto, per banda."""
    fig, axes = plt.subplots(2, len(bands), figsize=(3.3 * len(bands), 6.4), squeeze=False)
    for j, b in enumerate(bands):
        for (g, sid, bb), det in details.items():
            if bb != b or g not in colors:
                continue
            for i, var in enumerate(("sizes", "durations")):
                if len(det[var]):
                    axes[i, j].loglog(*_ccdf(det[var]), color=colors[g], lw=0.9, alpha=0.6)
        axes[0, j].set_title(b)
        axes[0, j].set_xlabel("Size S (events)")
        axes[1, j].set_xlabel("Duration T (bins)")
    axes[0, 0].set_ylabel("P(S ≥ s)")
    axes[1, 0].set_ylabel("P(T ≥ t)")
    for g, c in colors.items():
        axes[0, -1].plot([], [], color=c, label=_label(g))
    axes[0, -1].legend(fontsize=plots.BASE_FONT - 3)
    fig.suptitle("Complementary cumulative distributions of avalanche size and duration")
    _save(fig, out_png)


def _fit_panel(ax, x, f, color, xlabel, sym):
    """Un pannello: PDF a bin log, fit principale (x >= xmin), range power law e cutoff."""
    c, d = _logbin_pdf(x)
    ax.loglog(c, d, "o", color=color, ms=4, mfc="none")
    if not np.isfinite(f.get("alpha", np.nan)):
        return
    xs = np.unique(np.round(np.geomspace(f["xmin"], max(x), 50)))
    ax.loglog(xs, _model_pdf(xs, f["alpha"], f["xmin"], None, f["frac_tail"]), "k-", lw=1.2)
    txt = f"{sym} = {f['alpha']:.2f}, p = {f['p_gof']:.2f}"
    if np.isfinite(f.get("xmax_cutoff", np.nan)):
        lo, hi = f["range_xmin"], f["xmax_cutoff"]
        xr = np.unique(np.round(np.geomspace(lo, hi, 30)))
        ax.loglog(xr, _model_pdf(xr, f["alpha_range"], lo, hi, f["n_range"] / len(x)), color="tab:orange",
                  lw=2)
        ax.axvspan(lo, hi, color="tab:orange", alpha=0.08)
        ax.axvline(hi, color="tab:orange", ls="--", lw=1)
        txt += f"\nrange {lo:g}-{hi:g}: {sym} = {f['alpha_range']:.2f}"
    else:
        txt += "\nno power-law range"
    ax.text(0.03, 0.03, txt, transform=ax.transAxes, fontsize=plots.BASE_FONT - 5, va="bottom")
    ax.set_xlabel(xlabel)


def plot_powerlaw_fits(details, band, colors, out_png):
    """figA3: per ogni soggetto, PDF di S e di T con i fit (nero = fit principale, arancio = range)."""
    items = [(k, v) for k, v in details.items() if k[2] == band and k[0] in colors]
    if not items:
        return
    fig, axes = _grid(2 * len(items), ncols=6, h=3.0, w=3.4)
    for i, ((g, sid, _), det) in enumerate(items):
        for k, (var, fit, lab, sym) in enumerate((("sizes", "size_fit", "S", "τ"),
                                                   ("durations", "duration_fit", "T", "α"))):
            ax = axes[2 * i + k]
            if len(det[var]):
                _fit_panel(ax, det[var], det[fit], colors[g], lab, sym)
            ax.set_title(f"{sid[:14]} ({_label(g)[:9]})", fontsize=plots.BASE_FONT - 5,
                         color=colors[g])
    fig.suptitle(f"{band}: power-law fits (black: x ≥ xmin, orange: range where the power law holds)")
    _save(fig, out_png)


def plot_scaling(details, metrics, bands, colors, out_png):
    """figA4: <S>(T) per soggetto, log-log, con pendenza mediana fittata e prevista dal DCC."""
    fig, axes = plt.subplots(1, len(bands), figsize=(3.4 * len(bands), 3.8), squeeze=False)
    for j, b in enumerate(bands):
        ax = axes[0, j]
        for (g, sid, bb), det in details.items():
            if bb == b and g in colors and len(det["third_fit"]["T"]):
                ax.loglog(det["third_fit"]["T"], det["third_fit"]["mean_S"], "-", color=colors[g],
                          lw=0.8, alpha=0.6)
        lines = []
        for g in _groups(metrics):
            d = metrics[(metrics.band == b) & (metrics.group == g)]
            lines.append(f"{_label(g)[:10]}: fit {d.third.median():.2f}, pred {d.predicted_third.median():.2f}")
        ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, va="top",
                fontsize=plots.BASE_FONT - 5)
        ax.set(title=b, xlabel="Duration T (bins)")
    axes[0, 0].set_ylabel("<S>(T)")
    fig.suptitle("Size-duration scaling: <S>(T) ~ T^(1/σνz) (median slope fitted vs predicted (α-1)/(τ-1))")
    _save(fig, out_png)


def plot_branching_bins(br, bands, colors, out_png):
    """figA7: sigma (stimatore 'bins' sopra, 'geometric' sotto) per durata del time bin, media ± SEM."""
    fig, axes = plt.subplots(2, len(bands), figsize=(3.3 * len(bands), 6.4), squeeze=False, sharex=True)
    for j, b in enumerate(bands):
        for i, col in enumerate(("sigma_bins", "sigma_geometric")):
            ax = axes[i, j]
            for g in _groups(br):
                d = br[(br.band == b) & (br.group == g)].groupby("bin_ms")[col]
                m, s = d.mean(), d.std() / np.sqrt(d.count())
                ax.errorbar(m.index, m.values, yerr=s.values, color=cfg.COLORS[g], marker="o", ms=4,
                            capsize=2, label=_label(g))
            ax.axhline(1, color="k", ls="--", lw=1)
            if i == 0:
                ax.set_title(b)
            else:
                ax.set_xlabel("Time bin (ms)")
        axes[0, 0].set_ylabel("σ (Beggs & Plenz)")
        axes[1, 0].set_ylabel("σ (Sorrentino, geometric)")
    axes[0, 0].legend(fontsize=plots.BASE_FONT - 3)
    fig.suptitle("Branching ratio vs time bin (mean ± SEM across subjects)")
    _save(fig, out_png)


def plot_fano(details, metrics, bands, colors, out_png):
    """figA9: sopra FF per banda (box per gruppo), sotto curve FF(W) per soggetto."""
    fig, axes = plt.subplots(2, len(bands), figsize=(3.3 * len(bands), 6.6), squeeze=False)
    for j, b in enumerate(bands):
        _box_by_group(axes[0, j], metrics[metrics.band == b], "ff", colors, "Fano factor" if j == 0 else "")
        axes[0, j].set_title(f"{b}\n{axes[0, j].get_title()}", fontsize=plots.BASE_FONT - 1)
        ax = axes[1, j]
        for (g, sid, bb), det in details.items():
            cur = det["fano_curve"]
            if bb == b and g in colors and len(cur["windows_s"]):
                ax.loglog(1000 * cur["windows_s"], cur["ff"], color=colors[g], lw=0.8, alpha=0.6)
        ax.axhline(1, color="k", ls="--", lw=1)
        ax.set_xlabel("Window W (ms)")
    axes[1, 0].set_ylabel("FF(W)")
    fig.suptitle("Fano factor of population activity (FF = 1: Poisson)")
    _save(fig, out_png)


def plot_summary_heatmap(metrics, bands, colors, out_png):
    """figA10: mediane per gruppo (z-score per riga) e differenza tra gruppi (r rank-biserial)."""
    cols = [("dcc", "DCC"), ("tau", "τ"), ("alpha", "α"), ("third", "1/σνz"), ("sigma_bins", "σ"),
            ("sigma_geometric", "σ geometric"), ("repertoire_rate", "repertoire /s"),
            ("diversity", "pattern diversity"), ("ff", "Fano factor"), ("size_decades", "size decades")]
    cols = [(c, l) for c, l in cols if c in metrics]
    groups = _groups(metrics)
    if len(groups) != 2:
        return
    med = {g: np.array([[metrics[(metrics.band == b) & (metrics.group == g)][c].median()
                         for b in bands] for c, _ in cols]) for g in groups}
    eff, ann = np.full((len(cols), len(bands)), np.nan), np.full((len(cols), len(bands)), "", object)
    for i, (c, _) in enumerate(cols):
        for j, b in enumerate(bands):
            d = metrics[metrics.band == b]
            a = d.loc[d.group == groups[0], c].dropna()
            z = d.loc[d.group == groups[1], c].dropna()
            if len(a) >= 2 and len(z) >= 2:
                u = stats.mannwhitneyu(a, z)
                eff[i, j] = 2 * u.statistic / (len(a) * len(z)) - 1   # r rank-biserial
                ann[i, j] = _stars(u.pvalue).replace("n.s.", "")
    fig, axes = plt.subplots(1, 3, figsize=(6 + 1.6 * len(bands) * 2, 0.55 * len(cols) + 2.5))
    for ax, g in zip(axes[:2], groups):
        m = med[g]
        allv = np.hstack([med[k] for k in groups])
        mu, sd = np.nanmean(allv, 1, keepdims=True), np.nanstd(allv, 1, keepdims=True)
        im = ax.imshow((m - mu) / np.where(sd > 0, sd, 1), cmap="viridis", vmin=-1.5, vmax=1.5,
                       aspect="auto")
        for i in range(len(cols)):
            for j in range(len(bands)):
                ax.text(j, i, f"{m[i, j]:.2g}", ha="center", va="center", fontsize=9, color="w")
        ax.set_title(f"{_label(g)} (median)", color=colors[g])
    im = axes[2].imshow(eff, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i in range(len(cols)):
        for j in range(len(bands)):
            axes[2].text(j, i, f"{eff[i, j]:.2f}{ann[i, j]}", ha="center", va="center", fontsize=9)
    axes[2].set_title(f"{_label(groups[0])} vs {_label(groups[1])}\n(rank-biserial r, MWU)")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.set_xticks(range(len(bands)))
        ax.set_xticklabels(bands, rotation=45)
        ax.set_yticks(range(len(cols)))
        ax.set_yticklabels([l for _, l in cols] if ax is axes[0] else [])
    fig.suptitle("Avalanche metrics: group medians and group difference")
    _save(fig, out_png)


# ------------------------------------------------------------------ MAIN
def find_results(out_dir):
    for d in (out_dir, FALLBACK_DIR / cfg.OUT_SUBDIR):
        if (Path(d) / "avalanche_results.pkl").exists():
            return Path(d)
    raise SystemExit("avalanche_results.pkl non trovato: lancia prima MAIN_IPS_AVALAN.py")


def main():
    channel_set, _, out_root = base.choose_channel_set()   # stessa scelta di MAIN_IPS_AVALAN
    out = find_results(Path(out_root) / cfg.OUT_SUBDIR)
    print("Risultati letti da:", out)
    res = load_results(out / "avalanche_results.pkl")
    metrics, details, br = res["metrics"], res["details"], res["br"]
    bands = list(res["bands"])
    colors = {g: c for g, c in cfg.COLORS.items() if g in set(metrics.group)}
    plots.paper_style()

    plot_threshold_divergence(res["divergence_curves"], res["divergence"], bands, colors,
                              out / "figA1_threshold_divergence.png")
    plot_ccdf(details, bands, colors, out / "figA2_ccdf_sizes_durations.png")
    for b in bands:
        plot_powerlaw_fits(details, b, colors, out / f"figA3_powerlaw_fits_{b}.png")
    plot_scaling(details, metrics, bands, colors, out / "figA4_size_duration_scaling.png")
    _band_boxes(metrics, [("tau", "τ (size)"), ("alpha", "α (duration)"), ("third", "1/σνz"),
                          ("dcc", "DCC"), ("dcc_clauset", "DCC (Clauset xmin)")], bands, colors,
                out / "figA5_exponents_dcc.png",
                "Critical exponents and deviation from criticality coefficient (Maschke et al. 2024)")
    _band_boxes(metrics, [("size_p_gof", "p power law (S)"), ("size_decades", "decades (S)"),
                          ("size_xmax_cutoff", "cutoff S"), ("size_mse", "MSE log-log (S)"),
                          ("duration_p_gof", "p power law (T)"), ("duration_mse", "MSE log-log (T)")],
                bands, colors, out / "figA6_powerlaw_quality.png",
                "How power-law are the distributions (Clauset p, range in decades, cutoff, MSE)")
    plot_branching_bins(br, bands, colors, out / "figA7_branching_ratio_bins.png")
    _band_boxes(metrics, [("repertoire_rate", "unique patterns / s"),
                          ("repertoire_per_avalanche", "unique / avalanches"),
                          ("diversity", "pattern diversity\n(median Hamming)")],
                bands, colors, out / "figA8_repertoire.png",
                "Avalanche repertoire (Sorrentino et al. 2021; Maschke et al. 2024)")
    plot_fano(details, metrics, bands, colors, out / "figA9_fano.png")
    plot_summary_heatmap(metrics, bands, colors, out / "figA10_summary_heatmap.png")
    print("\nGrafici salvati in:", out)


if __name__ == "__main__":
    main()
