# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 16:14:40 2026

@author: Chiara Clima
"""

"""
Tutti i grafici. Ogni funzione riceve dati e percorsi.
Stile da articolo (paper_style) e tutte le scritte nelle figure in inglese.
Unici calcoli qui: test tra gruppi da annotare sulle figure e IC bootstrap della fig. 8.
"""
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from scipy.optimize import curve_fit

from .summary_stats import compare_groups

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAVE_PLOTLY = True
except ImportError:
    HAVE_PLOTLY = False


# ------------------------------------------------------------------ STILE
BASE_FONT = 14


def paper_style(base=BASE_FONT):
    """Stile tipo Nature: sans-serif (Arial), assi puliti, font grandi, 300 dpi."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": base,
        "axes.titlesize": base + 2,
        "axes.labelsize": base + 2,
        "xtick.labelsize": base,
        "ytick.labelsize": base,
        "legend.fontsize": base - 1,
        "figure.titlesize": base + 4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.8,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,          # testo modificabile se si salva in pdf/svg
        "svg.fonttype": "none",
    })


PLOTLY_FONT = dict(family="Arial", size=16)


def _label(g):
    """Nome del gruppo leggibile: 'not_photosensitive' -> 'not photosensitive'."""
    return str(g).replace("_", " ")


def _rgba(color, alpha):
    r, g, b = mcolors.to_rgb(color)
    return f"rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},{alpha})"


def _grid(n, ncols=5, h=3.2, w=4.0):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    return fig, axes.ravel()


def _save(fig, path):
    """Salva la figura. Se il file vecchio e' bloccato (aperto in un visualizzatore, OneDrive...)
    prova a cancellarlo; se non si puo', salva con suffisso _new e avvisa invece di fermarsi."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # figure con gridspec annidati: avviso innocuo
        fig.tight_layout()
    path = Path(path)
    try:
        try:
            fig.savefig(path, dpi=300, bbox_inches="tight")
        except OSError:
            try:
                path.unlink(missing_ok=True)
                fig.savefig(path, dpi=300, bbox_inches="tight")
            except OSError as e:
                alt = path.with_name(path.stem + "_new" + path.suffix)
                print(f"ATTENZIONE: non riesco a scrivere {path.name} ({e}).\n"
                      f"  Il file e' aperto in un altro programma o bloccato da OneDrive? Salvo come {alt.name}")
                fig.savefig(alt, dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)


def _need_plotly():
    if not HAVE_PLOTLY:
        print("plotly non installato: salto il grafico interattivo (pip install plotly)")
    return HAVE_PLOTLY


def _p_str(p):
    if not np.isfinite(p):
        return "p = n.a."
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _stars(p):
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def band_group_tests(subj, value, bands, group_col="group"):
    """
    Test tra gruppi per ogni banda sui valori PER SOGGETTO (media sui canali): i canali
    dello stesso soggetto non sono indipendenti, quindi il test non si fa su canali x soggetti.
    2 gruppi: Mann-Whitney U; piu' gruppi: Kruskal-Wallis; p_fdr = FDR-BH sulle bande.
    Ritorna DataFrame indicizzato per banda (vuoto se non si puo' testare).
    """
    wide = subj.pivot_table(index=[group_col, "subject"], columns="band", values=value).reset_index()
    t = compare_groups(wide, [b for b in bands if b in wide.columns], group_col)
    return t.set_index("metric") if len(t) else t


def _test_note(tests):
    if len(tests) == 0:
        return "Group test not available (need ≥2 subjects per group)"
    return (f"{tests['test'].iloc[0]} on subject-level values; "
            "q = FDR-corrected (Benjamini–Hochberg) across bands")


# ------------------------------------------------------------------ PSD
def plot_psd_all_subjects(store, colors, out_png, out_html=None, common_f=None, fmax=45.0):
    """PSD di ogni soggetto (linee sottili) + media di gruppo ± SEM, fino a fmax Hz."""
    if common_f is None:
        common_f = np.arange(1.0, fmax + 1e-9, 0.1)
    common_f = common_f[common_f <= fmax]
    means = {}
    fig, ax = plt.subplots(figsize=(8, 6))
    for s in store:
        m = s["f"] <= fmax
        ax.loglog(s["f"][m], s["psd"][m], color=colors[s["group"]], lw=0.7, alpha=0.35)
    for g, c in colors.items():
        A = np.array([np.interp(common_f, s["f"], s["psd"]) for s in store if s["group"] == g])
        if len(A) == 0:
            continue
        m = A.mean(0)
        sem = A.std(0, ddof=1) / np.sqrt(len(A)) if len(A) > 1 else 0 * m
        means[g] = m
        ax.loglog(common_f, m, color=c, lw=3, label=f"{_label(g)} (n = {len(A)})")
        ax.fill_between(common_f, np.clip(m - sem, 1e-12, None), m + sem, color=c, alpha=0.2)
    ax.set_xlim(common_f[0], fmax)
    ax.set(xlabel="Frequency (Hz)", ylabel="PSD (µV²/Hz)",
           title="Power spectral density (group mean ± SEM)")
    ax.legend()
    _save(fig, out_png)

    if out_html and _need_plotly():
        fi = go.Figure()
        for s in store:
            m = s["f"] <= fmax
            fi.add_trace(go.Scatter(
                x=s["f"][m], y=s["psd"][m], mode="lines", name=s["subject"], legendgroup=s["group"],
                line=dict(color=_rgba(colors[s["group"]], 0.45), width=1.3),
                hovertemplate="%{x:.2f} Hz<br>%{y:.3g} µV²/Hz<extra>"
                              + s["subject"] + " | " + _label(s["group"]) + "</extra>"))
        for g, m in means.items():
            fi.add_trace(go.Scatter(x=common_f, y=m, mode="lines", name=f"MEAN {_label(g)}",
                                    line=dict(color=_rgba(colors[g], 1), width=4)))
        fi.update_layout(template="simple_white", hovermode="closest", font=PLOTLY_FONT,
                         xaxis=dict(type="log", title="Frequency (Hz)"),
                         yaxis=dict(type="log", title="PSD (µV²/Hz)"),
                         title="PSD per subject (double-click a legend entry to isolate it)")
        fi.write_html(out_html, include_plotlyjs=True)


def plot_psd_each_subject(store, colors, out_png, fmax=45.0):
    fig, axes = _grid(len(store))
    for ax, s in zip(axes, store):
        m = s["f"] <= fmax
        ax.loglog(s["f"][m], s["psd"][m], color=colors[s["group"]], lw=1.5)
        ax.set_xlim(s["f"][m][0], fmax)
        ax.set_title(s["subject"][:22], fontsize=BASE_FONT - 2, color=colors[s["group"]])
        ax.tick_params(labelsize=BASE_FONT - 3)
    fig.supxlabel("Frequency (Hz)")
    fig.supylabel("PSD (µV²/Hz)")
    fig.suptitle("PSD per subject (mean over epochs and channels)")
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
        ax.scatter(s["fstim"][order], ys, s=8, c="white", marker="|", linewidths=1)
        step = max(1, len(order) // 8)
        ax.set_yticks(ys[::step])
        ax.set_yticklabels([f"{v:g}" for v in s["fstim"][order][::step]], fontsize=BASE_FONT - 4)
        ax.set_title(s["subject"][:22], fontsize=BASE_FONT - 2, color=colors[s["group"]])
        ax.tick_params(axis="x", labelsize=BASE_FONT - 3)
    fig.supxlabel("Frequency (Hz)")
    fig.supylabel("Stimulation frequency of the epoch (Hz)")
    fig.suptitle("log10 PSD (ROI) per epoch; white tick = stimulation frequency")
    _save(fig, out_png)


# ------------------------------------------------------------------ BANDE
def plot_band_hist_gauss(long, subj, bands, colors, out_png):
    """
    Istogrammi log10 potenza per banda (canali x soggetti) con fit gaussiano per gruppo.
    Nel titolo di ogni pannello: test tra gruppi sui valori per soggetto (p e q FDR).
    """
    tests = band_group_tests(subj, "log10_abs", bands)
    fig, axes = plt.subplots(1, len(bands), figsize=(4.6 * len(bands), 5.0), squeeze=False)
    for ax, b in zip(axes[0], bands):
        for g, c in colors.items():
            v = long[(long.group == g) & (long.band == b)].log10_abs.dropna()
            if len(v) < 3:
                continue
            mu, sd = stats.norm.fit(v)
            xs = np.linspace(v.min(), v.max(), 200)
            ax.hist(v, bins=30, density=True, alpha=0.35, color=c)
            ax.plot(xs, stats.norm.pdf(xs, mu, sd), color=c, lw=2.5,
                    label=f"μ = {mu:.2f}, σ = {sd:.2f}")
        title = b
        if b in tests.index:
            title += f"\n{_p_str(tests.loc[b, 'p'])}, q = {tests.loc[b, 'p_fdr']:.3f}"
        ax.set_title(title)
        ax.set_xlabel("log10 power (µV²)")
        ax.legend(fontsize=BASE_FONT - 3, loc="upper left")
    axes[0][0].set_ylabel("Density")
    handles = [plt.Line2D([], [], color=c, lw=3, label=_label(g)) for g, c in colors.items()]
    fig.legend(handles=handles, loc="upper right", ncol=len(handles))
    fig.suptitle("Band power distribution (channels × subjects) with Gaussian fit", x=0.4)
    fig.text(0.5, -0.02, _test_note(tests), ha="center", fontsize=BASE_FONT - 2)
    _save(fig, out_png)
    return tests


def plot_relative_power_boxplot(subj, bands, colors, out_png, seed=0):
    """Potenza relativa per banda: box + soggetti, con test tra gruppi (p, q FDR) per banda."""
    tests = band_group_tests(subj, "rel", bands)
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(10, 6))
    w = 0.8 / max(len(colors), 1)
    tops = np.zeros(len(bands))
    for k, (g, c) in enumerate(colors.items()):
        data = [subj[(subj.group == g) & (subj.band == b)].rel.dropna().values for b in bands]
        pos = np.arange(len(bands)) + (k - (len(colors) - 1) / 2) * w
        bp = ax.boxplot(data, positions=pos, widths=w * 0.85, patch_artist=True,
                        showfliers=False, medianprops=dict(color="k", lw=2))
        for patch in bp["boxes"]:
            patch.set_facecolor(c)
            patch.set_alpha(0.45)
        for p, d in zip(pos, data):
            ax.scatter(p + rng.uniform(-w * 0.2, w * 0.2, len(d)), d, s=22, color=c,
                       edgecolor="k", linewidth=0.5, zorder=3)
        tops = np.maximum(tops, [d.max() if len(d) else 0 for d in data])
        ax.plot([], [], color=c, lw=8, alpha=0.6, label=_label(g))
    span = max(tops.max(), 1e-9)
    for i, b in enumerate(bands):
        if b not in tests.index:
            continue
        y = tops[i] + 0.05 * span
        ax.plot([i - 0.4 + w / 2, i - 0.4 + w / 2, i + 0.4 - w / 2, i + 0.4 - w / 2],
                [y, y + 0.02 * span, y + 0.02 * span, y], color="k", lw=1.2)
        ax.text(i, y + 0.03 * span, f"{_stars(tests.loc[b, 'p_fdr'])}\n{_p_str(tests.loc[b, 'p'])}",
                ha="center", va="bottom", fontsize=BASE_FONT - 3)
    ax.set_ylim(top=span * 1.3)
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels(list(bands))
    ax.set(ylabel="Relative power", title="Relative band power per subject")
    ax.legend(loc="upper right")
    fig.text(0.5, -0.02, _test_note(tests) + "; stars refer to q",
             ha="center", fontsize=BASE_FONT - 3)
    _save(fig, out_png)
    return tests


# ------------------------------------------------------------------ AMPIEZZE
def _runs(mask):
    """Indici (inizio, fine) dei tratti consecutivi True in mask."""
    edges = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))


def _keep_runs(mask, min_run):
    """Tiene solo i tratti di almeno min_run bin consecutivi (toglie i bin isolati dovuti al caso)."""
    out = np.zeros_like(mask)
    for i0, i1 in _runs(mask):
        if i1 - i0 + 1 >= min_run:
            out[i0:i1 + 1] = True
    return out


def amplitude_deviation_from_gaussian(store, group, n_boot=2000, seed=0, sd_levels=(1, 2, 3, 4, 5),
                                      min_run=3):
    """
    Distribuzione delle ampiezze z-scored di un gruppo vs la SUA gaussiana di fit.
    - le ampiezze sono z-scored per canale (ogni canale ha media 0 e SD 1): asse x in unita' di SD
    - densita' del gruppo = media delle densita' dei soggetti (ogni soggetto pesa uguale)
    - IC 95% = bootstrap sui soggetti
    - gaussiana di fit: minimi quadrati sulla densita' del gruppo (mu, sigma liberi), quindi
      ogni gruppo ha la propria gaussiana adattata alla parte centrale della propria distribuzione.
      (Con il metodo dei momenti uscirebbe sempre circa N(0,1) per costruzione dello z-score.)
    - deviazione: bin con dati in cui la gaussiana cade FUORI dall'IC 95% del gruppo
      ('heavier' = dati sopra la gaussiana, 'lighter' = sotto). Contano solo tratti di almeno
      min_run bin consecutivi: con ~200 bin, il 5% cadrebbe fuori dall'IC per puro caso.
    - tail: per k in sd_levels, % di campioni oltre ±k SD nei dati e nella gaussiana di fit.
    Ritorna dict (None se il gruppo non ha soggetti).
    """
    items = [s for s in store if s["group"] == group]
    if not items:
        return None
    c = items[0]["amp"]["hist_z"]["centers"]
    w = c[1] - c[0]
    D = np.array([s["amp"]["hist_z"]["density"] for s in items])      # (n_subj, n_bins)
    n = len(items)
    dens = D.mean(0)

    rng = np.random.default_rng(seed)
    W = np.zeros((n_boot, n))                     # quante volte ogni soggetto e' ripescato
    np.add.at(W, (np.repeat(np.arange(n_boot), n), rng.integers(0, n, n_boot * n)), 1)
    boot = W @ D / n
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)

    mass = (dens * w).sum()                       # punto di partenza: media e SD dei dati
    mu0 = (c * dens * w).sum() / mass
    sd0 = np.sqrt((((c - mu0) ** 2) * dens * w).sum() / mass)
    try:
        (mu, sigma), _ = curve_fit(lambda x, m, s: stats.norm.pdf(x, m, s), c, dens,
                                   p0=[mu0, sd0], bounds=([-np.inf, 1e-6], [np.inf, np.inf]))
    except RuntimeError:
        mu, sigma = mu0, sd0
    gauss = stats.norm.pdf(c, mu, sigma)

    has_data = dens > 0
    heavier = _keep_runs(has_data & (gauss < lo), min_run)
    lighter = _keep_runs(has_data & (gauss > hi), min_run)

    tail = []
    for k in sd_levels:
        data_pct = 100 * (dens * w)[np.abs(c) > k].sum()
        gauss_pct = 100 * (stats.norm.sf(k, mu, sigma) + stats.norm.cdf(-k, mu, sigma))
        tail.append(dict(sd=k, data_pct=data_pct, gauss_pct=gauss_pct,
                         ratio=data_pct / gauss_pct if gauss_pct > 0 else np.nan))

    st = [s["amp"]["stats"] for s in items]
    return dict(centers=c, D=D, density=dens, lo=lo, hi=hi, mu=mu, sigma=sigma, gauss=gauss,
                heavier=heavier, lighter=lighter, n=n, tail=tail,
                kurtosis=np.mean([x["excess_kurtosis"] for x in st]),
                ks=np.mean([x["ks_D_vs_gauss"] for x in st]))


def _fmt_pct(v):
    if not np.isfinite(v):
        return "n.a."
    return "0" if v == 0 else f"{v:.2g}%" if v < 0.01 else f"{v:.2f}%"


def plot_amplitude_z_group(store, group, color, out_png, n_boot=2000, seed=0,
                           sd_levels=(1, 2, 3, 4, 5)):
    """
    Una figura per gruppo. Sopra: densita' (asse y log) dei soggetti e del gruppo con IC 95%,
    gaussiana di fit del gruppo, zone dove i dati deviano dalla gaussiana, linee a ±1..5 SD.
    In mezzo: log10(dati / gaussiana). Sotto: tabella % campioni oltre ±k SD (dati vs gaussiana).
    """
    r = amplitude_deviation_from_gaussian(store, group, n_boot, seed, sd_levels)
    if r is None:
        return None
    c, g = r["centers"], r["gauss"]
    fig = plt.figure(figsize=(9, 11.5))
    outer = fig.add_gridspec(2, 1, height_ratios=[4.4, 1.0], hspace=0.2)
    inner = outer[0].subgridspec(2, 1, height_ratios=[3, 1.4], hspace=0.08)
    ax = fig.add_subplot(inner[0])
    ax2 = fig.add_subplot(inner[1], sharex=ax)
    ax3 = fig.add_subplot(outer[1])

    def shade(a):
        for mask, col in ((r["heavier"], "tab:orange"), (r["lighter"], "tab:purple")):
            for i0, i1 in _runs(mask):
                a.axvspan(c[i0] - (c[1] - c[0]) / 2, c[i1] + (c[1] - c[0]) / 2,
                          color=col, alpha=0.15, lw=0)
        for k in sd_levels:
            for sgn in (-1, 1):
                a.axvline(sgn * k, color="0.4", ls=":", lw=1)

    shade(ax)
    for d in r["D"]:
        ax.semilogy(c, np.where(d > 0, d, np.nan), color=color, lw=0.7, alpha=0.35)
    ax.fill_between(c, np.where(r["lo"] > 0, r["lo"], np.nan), np.where(r["hi"] > 0, r["hi"], np.nan),
                    color=color, alpha=0.3, lw=0)
    ax.semilogy(c, np.where(r["density"] > 0, r["density"], np.nan), color=color, lw=3,
                label=f"{_label(group)}, group mean (95% CI)")
    ax.semilogy(c, g, "k--", lw=2,
                label=f"Gaussian fit to this group (μ = {r['mu']:.2f}, σ = {r['sigma']:.2f})")
    ax.plot([], [], color="tab:orange", lw=8, alpha=0.3, label="data above Gaussian")
    ax.plot([], [], color="tab:purple", lw=8, alpha=0.3, label="data below Gaussian")
    ax.set(ylim=(1e-5, 3), ylabel="Density")
    ax.set_title(f"Amplitude distribution: {_label(group)} (n = {r['n']})")
    ax.legend(loc="upper left", fontsize=BASE_FONT - 4)
    ax.text(0.98, 0.97, f"excess kurtosis = {r['kurtosis']:.2f}\nKS D = {r['ks']:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=BASE_FONT - 3)
    ax.tick_params(labelbottom=False)

    shade(ax2)
    ok = (r["density"] > 0) & (g > 0)
    ratio = np.where(ok, np.log10(np.where(ok, r["density"], 1) / g), np.nan)
    rlo = np.where(ok & (r["lo"] > 0), np.log10(np.where(r["lo"] > 0, r["lo"], 1) / g), np.nan)
    rhi = np.where(ok & (r["hi"] > 0), np.log10(np.where(r["hi"] > 0, r["hi"], 1) / g), np.nan)
    ax2.fill_between(c, rlo, rhi, color=color, alpha=0.3, lw=0)
    ax2.plot(c, ratio, color=color, lw=2.5)
    ax2.axhline(0, color="k", ls="--", lw=1.5)
    ticks = [-k for k in sd_levels[::-1]] + [0] + list(sd_levels)
    ax2.set_xticks(ticks)
    ax2.set(xlabel="Amplitude (SD units, z-scored per channel)", ylabel="log10(data / Gaussian)",
            xlim=(c[0], c[-1]))

    ax3.axis("off")
    rows = [["% beyond ±k SD"] + [f"±{t['sd']} SD" for t in r["tail"]],
            ["data"] + [_fmt_pct(t["data_pct"]) for t in r["tail"]],
            ["Gaussian fit"] + [_fmt_pct(t["gauss_pct"]) for t in r["tail"]],
            ["data / Gaussian"] + [f"{t['ratio']:.3g}" if np.isfinite(t["ratio"]) else "n.a."
                                   for t in r["tail"]]]
    tb = ax3.table(cellText=rows, loc="center", cellLoc="center",
                   colWidths=[0.3] + [0.7 / len(sd_levels)] * len(sd_levels))
    tb.auto_set_font_size(False)
    tb.set_fontsize(BASE_FONT - 3)
    tb.scale(1, 1.8)
    for (i, j), cell in tb.get_celld().items():
        cell.set_linewidth(0.6)
        if i == 0 or j == 0:
            cell.set_text_props(fontweight="bold")
    fig.align_ylabels([ax, ax2])
    _save(fig, out_png)
    return r
