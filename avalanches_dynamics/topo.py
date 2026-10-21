# -*- coding: utf-8 -*-
"""
Disegno delle mappe dei canali sullo scalpo: pattern binari (canali attivi pieni, inattivi vuoti) e
mappe continue (pesi di una componente). Posizioni dal montaggio standard 10-20 di MNE, proiezione
azimutale come nelle topografie di MNE.
"""
import numpy as np
import matplotlib.pyplot as plt

from descriptive_analysis.topography import channel_info, ALIASES


def canonical_names(ch_names):
    """Nomi vecchi del 10-20 -> nomi nuovi (T3 -> T7, T4 -> T8, T5 -> P7, T6 -> P8)."""
    up = {k.upper(): v for k, v in ALIASES.items()}
    return [up.get(c.upper(), c) for c in ch_names]


def positions_2d(ch_names):
    """Coordinate 2D (n_canali, 2) normalizzate a raggio della testa ~1. NaN per canali sconosciuti."""
    info, found, _ = channel_info(list(ch_names))
    pos3 = {c: info["chs"][i]["loc"][:3] for i, c in enumerate(found)}
    xy = np.full((len(ch_names), 2), np.nan)
    for i, c in enumerate(ch_names):
        if c in pos3:
            x, y, z = pos3[c]
            r = np.sqrt(x * x + y * y + z * z)
            theta = np.arccos(np.clip(z / r, -1, 1))
            phi = np.arctan2(y, x)
            xy[i] = theta * np.cos(phi), theta * np.sin(phi)
    scale = np.nanmax(np.hypot(xy[:, 0], xy[:, 1]))
    return xy / (scale / 0.85) if np.isfinite(scale) and scale > 0 else xy


def draw_head(ax, lw=1.0, color="0.3"):
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(t), np.sin(t), color=color, lw=lw)
    ax.plot([-0.09, 0, 0.09], [0.995, 1.12, 0.995], color=color, lw=lw)          # naso
    for s in (-1, 1):                                                              # orecchie
        ax.plot(s * np.array([0.99, 1.06, 1.08, 1.06, 0.99]),
                [0.12, 0.10, 0.0, -0.10, -0.12], color=color, lw=lw)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.15, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_binary_pattern(ax, xy, active, color, size=40):
    """Un pattern: canali attivi pieni nel colore del gruppo, inattivi cerchi vuoti grigi."""
    draw_head(ax)
    active = np.asarray(active, bool)
    ax.scatter(xy[~active, 0], xy[~active, 1], s=size, facecolors="none", edgecolors="0.6", lw=0.8)
    ax.scatter(xy[active, 0], xy[active, 1], s=size * 1.3, color=color, edgecolors="k", lw=0.5, zorder=3)


def plot_weights(ax, xy, w, vlim=None, cmap="RdBu_r", size=110):
    """Pesi continui sui canali (es. una componente PCA): cerchi colorati, scala simmetrica."""
    draw_head(ax)
    lim = vlim or (np.nanmax(np.abs(w)) or 1)
    return ax.scatter(xy[:, 0], xy[:, 1], c=w, cmap=cmap, vmin=-lim, vmax=lim, s=size,
                      edgecolors="k", lw=0.5, zorder=3)
