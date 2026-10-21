# -*- coding: utf-8 -*-
"""
Grafici della dinamica delle valanghe, database IPS: legge avalanche_dynamics.pkl prodotto da
MAIN_IPS_AVALDYN.py (nessun ricalcolo). Figure in criticality_results/avalanches_dynamics.

  figD1_top_patterns_<banda>      pattern piu' frequenti per gruppo sulla testa (canali attivi pieni)
  figD2_pattern_diversity_mds     mappa 2D dei pattern (MDS della distanza di Hamming) per gruppo
  figD3_pca_components            componenti della PCA comune dei pattern (modi di co-attivazione)
  figD4_repertoire_dimensionality participation ratio, varianza della PC1, pattern unici per soggetto
  figD5_phase_portrait            campo di deriva sul piano PC1-PC2 degli stati e punti fissi
  figD6_transition_matrices       matrici di transizione tra stati di valanghe consecutive
  figD7_states                    stati (centroidi k-means) usati nelle matrici di transizione
  figD8_dynamics_stats            lam_max, informazione mutua (z) e self-transition per soggetto
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from photosensitive import config_ips as base
from photosensitive import config_ips_avaldyn as cfg
from photosensitive.report import FALLBACK_DIR, load_results
from photosensitive.plot_ips_avalan import _band_boxes, _save
from descriptive_analysis import plots
from descriptive_analysis.plots import _label
from avalanches_dynamics.topo import positions_2d, plot_binary_pattern, plot_weights, draw_head

FIXED_STYLE = {"stable": dict(marker="o", mfc="k", mec="w"), "stable focus": dict(marker="o", mfc="k", mec="y"),
               "unstable": dict(marker="o", mfc="w", mec="k"), "unstable focus": dict(marker="o", mfc="w", mec="y"),
               "saddle": dict(marker="X", mfc="k", mec="w")}


def plot_top_patterns(res, xy, colors, out_png):
    """
    figD1: per ogni gruppo due righe: i pattern piu' frequenti fra tutte le valanghe e fra quelle con
    almeno MIN_MULTI_CHANNELS canali (frequenza media tra soggetti; soggetti in cui il pattern compare).
    """
    groups = res["groups"]
    rows = [(g, k) for g in groups for k in ("", "multi_") if f"{k}patterns" in res["top"][g]]
    n = max(len(res["top"][g][f"{k}patterns"]) for g, k in rows)
    fig, axes = plt.subplots(len(rows), n, figsize=(1.6 * n, 2.0 * len(rows)), squeeze=False)
    for i, (g, k) in enumerate(rows):
        t = res["top"][g]
        for j in range(n):
            ax = axes[i, j]
            if j >= len(t[f"{k}patterns"]):
                ax.axis("off")
                continue
            plot_binary_pattern(ax, xy, t[f"{k}patterns"][j], colors[g], size=18)
            ax.set_title(f"#{j + 1}  {100 * t[f'{k}freq'][j]:.1f}%\n"
                         f"{t[f'{k}n_subjects'][j]}/{t['n_groups_subjects']} subj", fontsize=8)
        what = "all avalanches" if not k else f"≥ {cfg.MIN_MULTI_CHANNELS} channels"
        axes[i, 0].text(-0.35, 0.5, f"{_label(g)}\n{what}\n{t[f'{k}n_unique']} unique",
                        transform=axes[i, 0].transAxes, rotation=90, ha="center", va="center",
                        color=colors[g], fontsize=9)
    fig.suptitle(f"{res['band']}: most frequent avalanche patterns (filled = active channels; "
                 "% = mean frequency across subjects within the row's avalanches)", fontsize=12)
    _save(fig, out_png)


def plot_mds(results, bands, colors, out_png):
    """figD2: pattern frequenti sul piano MDS (vicini = differiscono in pochi canali), area ~ frequenza."""
    fig, axes = plt.subplots(1, len(bands), figsize=(3.6 * len(bands), 4), squeeze=False)
    for ax, b in zip(axes[0], bands):
        r = results[b]
        xy = r["mds"]["xy"]
        for g in r["groups"]:
            f = r["mds"]["freq"][g]
            m = f > 0
            ax.scatter(xy[m, 0], xy[m, 1], s=4 + 3000 * f[m], color=colors[g], alpha=0.45,
                       edgecolors="none", label=f"{_label(g)} ({r['top'][g]['n_unique']} unique)")
        ax.set(title=f"{b}  (MDS {100 * r['mds']['explained']:.0f}% var.)", xticks=[], yticks=[])
        ax.legend(fontsize=7, loc="lower left", markerscale=0.3)
    fig.suptitle("Repertoire diversity: frequent patterns placed by Hamming distance (close = similar), "
                 "area ~ frequency")
    _save(fig, out_png)


def plot_pca(results, bands, xy, colors, out_png):
    """figD3: pesi dei canali di ogni componente (rosso/blu = si attivano insieme con segno opposto)."""
    k = cfg.N_COMPONENTS
    fig, axes = plt.subplots(len(bands), k, figsize=(2.6 * k, 2.6 * len(bands)), squeeze=False)
    for i, b in enumerate(bands):
        r = results[b]
        for j in range(k):
            ax = axes[i, j]
            if j >= len(r["pca"]["components"]):
                ax.axis("off")
                continue
            plot_weights(ax, xy, r["pca"]["components"][j], size=55)
            ev = " · ".join(f"{100 * r['group_evr'][g][j]:.0f}%" for g in r["groups"])
            ax.set_title(f"PC{j + 1}: {ev}", fontsize=9)
        axes[i, 0].text(-0.2, 0.5, b, transform=axes[i, 0].transAxes, rotation=90, ha="center",
                        va="center", fontsize=12)
    fig.suptitle("Co-activation modes: common PCA of avalanche patterns (red/blue = opposite weights)\n"
                 "% = variance of the group patterns explained by the component: "
                 + " · ".join(_label(g) for g in results[bands[0]]["groups"]), fontsize=12)
    _save(fig, out_png)


def plot_phase(results, bands, colors, out_png):
    """figD5: campo di deriva D1 (frecce), passaggi per cella (sfondo) e punti fissi del fit polinomiale."""
    groups = results[bands[0]]["groups"]
    fig, axes = plt.subplots(len(groups), len(bands), figsize=(3.7 * len(bands), 3.6 * len(groups)),
                             squeeze=False)
    for j, b in enumerate(bands):
        r = results[b]
        ex, ey = r["edges"]
        evr = r["state_pca"]["explained_ratio"]
        for i, g in enumerate(groups):
            ax = axes[i, j]
            d = r["dynamics"][g]
            fld = d["field"]
            cnt = np.where(fld["count"] > 0, fld["count"], np.nan)
            ax.pcolormesh(ex, ey, cnt, cmap="Greys", norm=LogNorm(), alpha=0.5, shading="flat")
            X, Y = np.meshgrid(fld["cx"], fld["cy"])
            ax.quiver(X, Y, fld["u"], fld["v"], color=colors[g], angles="xy", scale_units="xy", scale=1,
                      width=0.006)
            for f in d["fixed"]:
                ax.plot(f["x"], f["y"], ms=11, lw=0, mew=1.5, zorder=5, **FIXED_STYLE[f["kind"]])
            ax.set(xlim=(ex[0], ex[-1]), ylim=(ey[0], ey[-1]))
            ax.set_title(f"{b}" if i == 0 else "", fontsize=12)
            if j == 0:
                ax.set_ylabel(f"{_label(g)}\nPC2 ({100 * evr[1]:.0f}%)", color=colors[g])
            if i == len(groups) - 1:
                ax.set_xlabel(f"PC1 ({100 * evr[0]:.0f}%)")
            ax.text(0.02, 0.02, f"λmax {d['linear']['lam_max']:.2f}\nn = {d['n_transitions']}",
                    transform=ax.transAxes, fontsize=8, va="bottom")
    for kind, st in FIXED_STYLE.items():
        axes[0, -1].plot([], [], ms=9, lw=0, label=kind, **st)
    axes[0, -1].legend(fontsize=7, loc="upper right")
    fig.suptitle("Avalanche phase portrait: mean step to the next time bin (Kramers-Moyal drift) on the "
                 "state PCA plane; grey = visits per cell; markers = fixed points of the fitted field")
    _save(fig, out_png)


def plot_transitions(results, bands, colors, out_png):
    """figD6: P(stato della valanga successiva | stato della valanga corrente), media tra soggetti."""
    groups = results[bands[0]]["groups"]
    fig = plt.figure(figsize=(3.2 * len(bands) + 0.8, 3.2 * len(groups)))
    gs = fig.add_gridspec(len(groups), len(bands) + 1, width_ratios=[1] * len(bands) + [0.06])
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(len(bands))] for i in range(len(groups))])
    cax = fig.add_subplot(gs[:, -1])                  # barra dei colori in una colonna a parte
    for j, b in enumerate(bands):
        for i, g in enumerate(groups):
            ax = axes[i, j]
            im = ax.imshow(results[b]["group_P"][g], cmap="magma", vmin=0, vmax=0.6)
            ax.set(xticks=range(cfg.N_STATES), yticks=range(cfg.N_STATES))
            ax.tick_params(labelsize=7)
            ax.set_title(b if i == 0 else "", fontsize=12)
            if j == 0:
                ax.set_ylabel(f"{_label(g)}\nstate n", color=colors[g])
            if i == len(groups) - 1:
                ax.set_xlabel("state n+1")
    fig.colorbar(im, cax=cax, label="transition probability")
    fig.suptitle("Transitions between consecutive avalanches (k-means states, see figD7)")
    _save(fig, out_png)


def plot_states(results, bands, xy, out_png):
    """figD7: centroidi degli stati = probabilita' di ogni canale di essere attivo nello stato."""
    k = cfg.N_STATES
    fig, axes = plt.subplots(len(bands), k, figsize=(1.9 * k, 2.0 * len(bands)), squeeze=False)
    for i, b in enumerate(bands):
        r = results[b]
        for j in range(k):
            ax = axes[i, j]
            draw_head(ax)
            ax.scatter(xy[:, 0], xy[:, 1], c=r["state_centroids"][j], cmap="Reds", vmin=0, vmax=1, s=30,
                       edgecolors="k", lw=0.4, zorder=3)
            occ = " · ".join(f"{100 * r['state_occupancy'][g][j]:.0f}%" for g in r["groups"])
            ax.set_title(f"S{j} {occ}", fontsize=8)
        axes[i, 0].text(-0.25, 0.5, b, transform=axes[i, 0].transAxes, rotation=90, ha="center",
                        va="center", fontsize=11)
    fig.suptitle("States for the transition matrices: probability that each channel is active "
                 "(% = share of avalanches in the state per group)", fontsize=12)
    _save(fig, out_png)


def main():
    channel_set, _, out_root = base.choose_channel_set()
    out = cfg.results_dir(channel_set, out_root)
    if not (out / "avalanche_dynamics.pkl").exists():
        out = FALLBACK_DIR / cfg.OUT_NAME
        if not (out / "avalanche_dynamics.pkl").exists():
            raise SystemExit("avalanche_dynamics.pkl non trovato: lancia prima MAIN_IPS_AVALDYN.py")
    print("Risultati letti da:", out)
    res = load_results(out / "avalanche_dynamics.pkl")
    results, metrics, bands = res["results"], res["metrics"], list(res["bands"])
    colors = {g: c for g, c in cfg.COLORS.items() if g in set(metrics.group)}
    xy = positions_2d(res["ch_names"])
    plots.paper_style()

    for b in bands:
        plot_top_patterns(results[b], xy, colors, out / f"figD1_top_patterns_{b}.png")
    plot_mds(results, bands, colors, out / "figD2_pattern_diversity_mds.png")
    plot_pca(results, bands, xy, colors, out / "figD3_pca_components.png")
    _band_boxes(metrics, [("participation_ratio", "participation ratio\n(repertoire dimensionality)"),
                          ("pc1_explained", "variance explained\nby common PC1"),
                          ("n_unique_patterns", "unique patterns")], bands, colors,
                out / "figD4_repertoire_dimensionality.png",
                "Repertoire dimensionality per subject (Mann-Whitney U)")
    plot_phase(results, bands, colors, out / "figD5_phase_portrait.png")
    plot_transitions(results, bands, colors, out / "figD6_transition_matrices.png")
    plot_states(results, bands, xy, out / "figD7_states.png")
    _band_boxes(metrics, [("lam_max", "λmax of linear drift\n(<0: relaxes to a fixed point)"),
                          ("mi_z", "sequence memory\n(MI z vs shuffled)"),
                          ("self_transition", "P(same state\nnext avalanche)")], bands, colors,
                out / "figD8_dynamics_stats.png", "Avalanche dynamics per subject (Mann-Whitney U)")
    print("\nGrafici salvati in:", out)


if __name__ == "__main__":
    main()
