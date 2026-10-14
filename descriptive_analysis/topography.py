# -*- coding: utf-8 -*-
"""
Mappe topografiche per gruppo e differenza tra gruppi (potenza per banda, varianza, ACFW,
esponente aperiodico...). Funzioni generali: ricevono una tabella con una riga per
soggetto x canale e disegnano con mne.viz.plot_topomap.
"""
import warnings

import numpy as np
import matplotlib.pyplot as plt
import mne

from .summary_stats import channel_group_tests
from .plots import _label, _save, BASE_FONT

# montaggi standard da provare in ordine: colin27_1020 (MNE >= 1.13) e standard_1020 (versioni
# precedenti) contengono anche i nomi vecchi T3/T4/T5/T6
MONTAGES = ("colin27_1020", "standard_1020")

# nomi vecchi del sistema 10-20 -> nomi nuovi (stessa posizione sullo scalpo)
ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


def unify_channel_names(chan, group_col="group"):
    """
    Unisce i canali che sono lo stesso punto sullo scalpo con nomi diversi (T3=T7, T4=T8,
    T5=P7, T6=P8), cosi' soggetti registrati con convenzioni diverse finiscono sullo stesso
    elettrodo. Se un soggetto ha entrambi i nomi, i due valori vengono mediati.
    Ritorna (tabella con nomi unificati, dict dei nomi rinominati).
    """
    up = {k.upper(): v for k, v in ALIASES.items()}
    renamed = {c: up[c.upper()] for c in chan["channel"].unique() if c.upper() in up}
    if not renamed:
        return chan, renamed
    out = chan.copy()
    out["channel"] = out["channel"].replace(renamed)
    num = out.select_dtypes("number").columns
    out = out.groupby([group_col, "subject", "channel"], sort=False)[list(num)].mean().reset_index()
    return out, renamed


def channel_info(ch_names, montages=MONTAGES):
    """
    mne.Info con le posizioni 2D dei canali che il montaggio conosce (maiuscole ignorate).
    Ritorna (info, canali_con_posizione, canali_scartati).
    """
    montage = None
    for name in montages:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                montage = mne.channels.make_standard_montage(name)
            break
        except ValueError:
            continue
    if montage is None:
        raise RuntimeError(f"Nessun montaggio disponibile tra {montages}")
    pos = montage.get_positions()["ch_pos"]
    up = {k.upper(): k for k in pos}
    found = [c for c in ch_names if c.upper() in up]
    missing = [c for c in ch_names if c.upper() not in up]
    # due canali nella stessa posizione bloccano plot_topomap: si tiene il primo e si avvisa
    seen, keep = {}, []
    for c in found:
        key = tuple(np.round(pos[up[c.upper()]], 6))
        if key in seen:
            missing.append(c)
            print(f"    topografia: {c} ha la stessa posizione di {seen[key]}, escluso")
        else:
            seen[key] = c
            keep.append(c)
    found = keep
    info = mne.create_info(found, 100.0, "eeg")
    dig = mne.channels.make_dig_montage({c: pos[up[c.upper()]] for c in found}, coord_frame="head")
    info.set_montage(dig)
    return info, found, missing


def plot_group_topomaps(chan, metrics, colors, out_png, title, group_col="group", alpha=0.05,
                        cmap="viridis"):
    """
    Una riga per misura: mappa media di ogni gruppo (stessa scala di colori) e, se i gruppi sono 2,
    mappa della differenza (primo gruppo - secondo) con i canali marcati:
      cerchio bianco grande = differenza significativa dopo FDR sui canali (q < alpha)
      punto nero piccolo    = p < alpha non corretto
    chan: una riga per soggetto x canale; metrics: lista di (colonna, etichetta in inglese).
    Ritorna la tabella dei test canale per canale.
    """
    groups = [g for g in colors if g in set(chan[group_col])]
    chan, renamed = unify_channel_names(chan, group_col)
    if renamed:
        print("    topografia: nomi vecchi unificati ai nuovi: "
              + ", ".join(f"{a}->{b}" for a, b in renamed.items()))
    names = list(dict.fromkeys(chan["channel"]))
    info, found, missing = channel_info(names)
    if missing:
        print(f"    topografia: canali senza posizione nel montaggio, esclusi: {missing}")
    if len(found) < 3:
        print("    topografia: meno di 3 canali con posizione, salto", out_png)
        return None
    tests = channel_group_tests(chan[chan.channel.isin(found)], [m for m, _ in metrics], group_col) \
        if len(groups) == 2 else None

    ncol = len(groups) + (1 if len(groups) == 2 else 0)
    fig, axes = plt.subplots(len(metrics), ncol, figsize=(3.6 * ncol + 1.2, 3.3 * len(metrics)),
                             squeeze=False)
    for r, (m, lab) in enumerate(metrics):
        means = {g: chan[chan[group_col] == g].groupby("channel")[m].mean().reindex(found).values
                 for g in groups}
        allv = np.concatenate([v[np.isfinite(v)] for v in means.values()])
        if allv.size == 0:
            for ax in axes[r]:
                ax.axis("off")
            continue
        vlim = (np.percentile(allv, 2), np.percentile(allv, 98))
        for c, g in enumerate(groups):
            ax = axes[r][c]
            im, _ = mne.viz.plot_topomap(np.nan_to_num(means[g], nan=np.nanmean(means[g])), info,
                                         axes=ax, show=False, cmap=cmap, vlim=vlim, contours=4,
                                         sensors=True)
            ax.set_title(_label(g) if r == 0 else "", color=colors[g], fontsize=BASE_FONT)
            if c == 0:
                ax.text(-0.25, 0.5, lab, transform=ax.transAxes, rotation=90, ha="center",
                        va="center", fontsize=BASE_FONT)
            if c == len(groups) - 1:
                fig.colorbar(im, ax=ax, shrink=0.75)
        if len(groups) == 2:
            ax = axes[r][2]
            diff = means[groups[0]] - means[groups[1]]
            lim = np.nanmax(np.abs(diff)) if np.isfinite(diff).any() else 1
            t = tests[tests.metric == m].set_index("channel").reindex(found) if tests is not None \
                and len(tests) else None
            q = t["p_fdr"].values if t is not None else np.full(len(found), np.nan)
            p = t["p"].values if t is not None else np.full(len(found), np.nan)
            im, _ = mne.viz.plot_topomap(np.nan_to_num(diff), info, axes=ax, show=False,
                                         cmap="RdBu_r", vlim=(-lim, lim), contours=4, sensors="k.")
            sens = [ln for ln in ax.lines if ln.get_xydata().shape[0] == len(found)]
            if sens:                                   # posizioni 2D dei canali, in ordine
                xy = sens[0].get_xydata()
                sens[0].set_markersize(2)
                sens[0].set_color("0.35")
                sig_q = np.nan_to_num(q, nan=1) < alpha
                sig_p = (np.nan_to_num(p, nan=1) < alpha) & ~sig_q
                ax.plot(xy[sig_p, 0], xy[sig_p, 1], "o", color="k", ms=4, zorder=5)
                ax.plot(xy[sig_q, 0], xy[sig_q, 1], "o", mfc="w", mec="k", mew=1.2, ms=9, zorder=6)
            ax.set_title(f"{_label(groups[0])} − {_label(groups[1])}" if r == 0 else "",
                         fontsize=BASE_FONT)
            fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle(title)
    if len(groups) == 2:
        fig.text(0.5, -0.01, f"Difference maps: white circles = q < {alpha} (Mann–Whitney U per "
                 f"channel, FDR across channels); black dots = p < {alpha} uncorrected",
                 ha="center", fontsize=BASE_FONT - 3)
    _save(fig, out_png)
    return tests
