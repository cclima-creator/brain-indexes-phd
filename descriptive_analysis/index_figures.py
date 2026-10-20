# -*- coding: utf-8 -*-
"""
One figure per index: one panel per frequency band (broadband first), subjects as points over a
box per group, Mann-Whitney p in the title of each panel (descriptive_analysis.plots._box_by_group)
and, under it, the p corrected with FDR across the bands of that index. Shared by the avalanche,
synchrony and edge-of-chaos figure scripts.

    plot_index_bands(long, "gs", "global synchrony", bands, colors, out_dir / "gs.png")

long: one row per subject x band (columns group, subject, band, <index>).
wide_to_long converts tables with one column per index x band (<index>_<band>) to that format.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .plots import _box_by_group, _save, BASE_FONT
from .summary_stats import compare_groups, fdr_bh


def wide_to_long(df, index, bands, id_cols=("group", "subject")):
    """Columns <index>_<band> -> long table (id_cols, band, <index>); bands without a column are skipped."""
    parts = [df[list(id_cols)].assign(band=b, **{index: df[f"{index}_{b}"]})
             for b in bands if f"{index}_{b}" in df]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def band_tests(long, index, bands):
    """Mann-Whitney per band and FDR across the bands of this index. Returns {band: (p, p_fdr)}."""
    rows = []
    for b in bands:
        t = compare_groups(long[long.band == b], [index])
        if len(t):
            rows.append((b, float(t.p.iloc[0])))
    if not rows:
        return {}
    q = fdr_bh(np.array([p for _, p in rows]))
    return {b: (p, qq) for (b, p), qq in zip(rows, q)}


def plot_index_bands(long, index, label, bands, colors, out_png, title=None, note=None):
    """One panel per band with data; returns the dict of band tests (p, p_fdr)."""
    if long is None or long.empty or index not in long:
        return {}
    present = [b for b in bands if long.loc[long.band == b, index].replace([np.inf, -np.inf], np.nan)
               .notna().any()]
    if not present:
        return {}
    tests = band_tests(long, index, present)
    fig, axes = plt.subplots(1, len(present), figsize=(3.3 * len(present) + 0.6, 4.4), squeeze=False)
    for ax, b in zip(axes[0], present):
        _box_by_group(ax, long[long.band == b], index, colors, label if ax is axes[0, 0] else "")
        p_title = ax.get_title()
        q = tests.get(b, (np.nan, np.nan))[1]
        q_txt = "" if not np.isfinite(q) else (f"FDR q < 0.001" if q < 0.001 else f"FDR q = {q:.3f}")
        ax.set_title(f"{b}\n{p_title}\n{q_txt}", fontsize=BASE_FONT - 1)
    fig.suptitle(title or label)
    if note:
        fig.text(0.5, -0.02, note, ha="center", va="top", fontsize=BASE_FONT - 3)
    _save(fig, out_png)
    return tests
