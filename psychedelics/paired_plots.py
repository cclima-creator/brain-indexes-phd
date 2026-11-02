# -*- coding: utf-8 -*-
"""
Figures for repeated measures (same subjects in every condition). Every figure gets the test
table from paired_stats, nothing is tested here except in plot_topomaps / plot_curves (per x value).

  paired_box          box per condition + one point per subject + grey lines joining the same
                      subject; title = Friedman p and FDR q; brackets = post hoc Wilcoxon,
                      stars from the FDR q, uncorrected p in grey when only p < 0.05
  plot_index_bands    one image per index: one panel per band (broadband first)
  plot_index_grid     several indices (rows) x bands (columns) in one image (overview figures)
  plot_summary_heatmap  index x band: Kendall's W with Friedman stars + rank-biserial r of each pair
  plot_curves         mean +- SEM per condition of a curve (PSD, MSE, DFA, CCDF, ...), Friedman
                      per x value with FDR across x (bar of significant x at the bottom)
  plot_topomaps       condition maps + difference map of every pair with significant channels
"""
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from descriptive_analysis.plots import _save, paper_style, BASE_FONT
from descriptive_analysis.summary_stats import fdr_bh
from psychedelics.paired_stats import paired_matrix, pairs_of, pair_name, friedman_row

STATS_NOTE = ("Friedman test across conditions (subjects with every condition); q = FDR across the bands of "
              "the index. Brackets: Wilcoxon signed-rank, stars from FDR across pairs × bands "
              "(* q<0.05, ** q<0.01, *** q<0.001); grey p = uncorrected p<0.05 only.")


def _p_txt(p, name="p"):
    if not np.isfinite(p):
        return f"{name} = n.a."
    return f"{name} < 0.001" if p < 0.001 else f"{name} = {p:.3f}"


def _stars(q):
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""


def _cond_label(c, labels):
    return labels.get(c, c) if labels else c


# ------------------------------------------------------------------ BOXES
def paired_box(ax, d, index, conditions, colors, labels=None, test=None, ylabel="", seed=0,
               show_title=True):
    """
    d: rows of one band (condition, subject, index). test: row of paired_stats.index_band_tests
    (Friedman p, p_fdr_bands, post hoc p_/q_ columns) or None.
    """
    rng = np.random.default_rng(seed)
    w = paired_matrix(d, index, conditions)
    allv = {c: d.loc[d.condition == c, index].replace([np.inf, -np.inf], np.nan).dropna().values
            for c in conditions}
    pos = np.arange(len(conditions))
    bp = ax.boxplot([allv[c] if len(allv[c]) else [np.nan] for c in conditions], positions=pos,
                    widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color="k", lw=2))
    for patch, c in zip(bp["boxes"], conditions):
        patch.set_facecolor(colors[c])
        patch.set_alpha(0.45)
    jit = {s: rng.uniform(-0.1, 0.1) for s in d.subject.unique()}
    for s, r in w.iterrows():                         # lines: subjects with every condition
        ax.plot(pos + jit[s], r.values, color="0.55", lw=0.7, alpha=0.6, zorder=2)
    for i, c in enumerate(conditions):
        dc = d[d.condition == c].replace([np.inf, -np.inf], np.nan).dropna(subset=[index])
        ax.scatter(i + dc.subject.map(jit).values, dc[index].values, s=22, color=colors[c],
                   edgecolor="k", linewidth=0.5, zorder=3)
    ax.set_xticks(pos)
    ax.set_xticklabels([_cond_label(c, labels) for c in conditions], fontsize=BASE_FONT - 3)
    ax.set_ylabel(ylabel)
    if test is None:
        return
    if show_title:
        q = test.get("p_fdr_bands", np.nan)
        ax.set_title(f"Friedman {_p_txt(test['p'])}\n{_p_txt(q, 'FDR q')}  W = {test['kendall_w']:.2f}"
                     f"  n = {int(test['n'])}", fontsize=BASE_FONT - 3)
    # brackets
    vals = np.concatenate([v for v in allv.values() if len(v)]) if any(len(v) for v in allv.values()) else np.array([0, 1])
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    span = (hi - lo) or 1.0
    level = 0
    for a, c in sorted(pairs_of(conditions), key=lambda ac: abs(conditions.index(ac[1]) - conditions.index(ac[0]))):
        p = test.get(f"p_{pair_name(a, c)}", np.nan)
        q = test.get(f"q_{pair_name(a, c)}", np.nan)
        if not (np.isfinite(p) and p < 0.05):
            continue
        txt, col = (_stars(q), "k") if np.isfinite(q) and q < 0.05 else (_p_txt(p).replace(" ", ""), "0.5")
        y = hi + span * (0.08 + 0.12 * level)
        i, j = conditions.index(a), conditions.index(c)
        ax.plot([i, i, j, j], [y - span * 0.02, y, y, y - span * 0.02], color=col, lw=1.1)
        ax.text((i + j) / 2, y, txt, ha="center", va="bottom", color=col,
                fontsize=BASE_FONT - 2 if col == "k" else BASE_FONT - 5)
        level += 1
    if level:
        ax.set_ylim(top=hi + span * (0.12 + 0.12 * level))


def _test_lookup(tests, index, band):
    if tests is None or tests.empty:
        return None
    t = tests[(tests["index"] == index) & (tests["band"] == (band or ""))]
    return t.iloc[0].to_dict() if len(t) else None


def plot_index_bands(long, index, label, bands, conditions, colors, tests, out_png, labels=None,
                     title=None, note=None):
    """One panel per band with data; long = rows subject x condition x band."""
    if long is None or long.empty or index not in long:
        return
    present = [b for b in bands if long.loc[long.band == b, index].replace([np.inf, -np.inf], np.nan)
               .notna().any()]
    if not present:
        return
    fig, axes = plt.subplots(1, len(present), figsize=(3.6 * len(present) + 0.6, 5.0), squeeze=False)
    for ax, b in zip(axes[0], present):
        paired_box(ax, long[long.band == b], index, conditions, colors, labels,
                   _test_lookup(tests, index, b), label if ax is axes[0, 0] else "")
        ax.set_title(f"{b}\n" + ax.get_title(), fontsize=BASE_FONT - 2)
    fig.suptitle(title or label)
    fig.text(0.5, -0.02, (note + "\n" if note else "") + STATS_NOTE, ha="center", va="top",
             fontsize=BASE_FONT - 5, wrap=True)
    _save(fig, out_png)


def plot_index_single(df, index, label, conditions, colors, tests, out_png, labels=None, title=None,
                      note=None):
    """One index without bands (one row per subject x condition)."""
    if index not in df or not df[index].replace([np.inf, -np.inf], np.nan).notna().any():
        return
    fig, ax = plt.subplots(figsize=(4.6, 5.0))
    paired_box(ax, df, index, conditions, colors, labels, _test_lookup(tests, index, None), label)
    fig.suptitle(title or label, fontsize=BASE_FONT + 1)
    fig.text(0.5, -0.02, (note + "\n" if note else "") + STATS_NOTE.replace("q = FDR across the bands of the index",
                                                                              "q = FDR across the indices of the figure"),
             ha="center", va="top", fontsize=BASE_FONT - 6, wrap=True)
    _save(fig, out_png)


def plot_index_grid(long, items, bands, conditions, colors, tests, out_png, labels=None, title=""):
    """items: list of (index, label) as rows; one column per band."""
    items = [(m, l) for m, l in items if m in long and long[m].replace([np.inf, -np.inf], np.nan).notna().any()]
    if not items:
        return
    present = [b for b in bands if (long.band == b).any()]
    fig, axes = plt.subplots(len(items), len(present), figsize=(3.2 * len(present), 3.6 * len(items)),
                             squeeze=False)
    for r, (m, lab) in enumerate(items):
        for c, b in enumerate(present):
            ax = axes[r, c]
            if not long.loc[long.band == b, m].replace([np.inf, -np.inf], np.nan).notna().any():
                ax.axis("off")
                continue
            paired_box(ax, long[long.band == b], m, conditions, colors, labels,
                       _test_lookup(tests, m, b), lab if c == 0 else "")
            t = ax.get_title()
            ax.set_title((f"{b}\n" if r == 0 else "") + t, fontsize=BASE_FONT - 4)
            if r < len(items) - 1:
                ax.set_xticklabels([])
    fig.suptitle(title)
    fig.text(0.5, -0.01, STATS_NOTE, ha="center", va="top", fontsize=BASE_FONT - 4, wrap=True)
    _save(fig, out_png)


# ------------------------------------------------------------------ SUMMARY HEATMAP
def plot_summary_heatmap(tests, bands, conditions, out_png, labels=None, index_labels=None, title="",
                         alpha=0.05):
    """
    Left: Kendall's W (Friedman effect size), stars = Friedman FDR q across the bands of the index,
    '+' = uncorrected p < alpha. Then one panel per pair: matched-pairs rank-biserial r
    (red = second condition higher), stars from the post hoc FDR q.
    """
    if tests is None or tests.empty:
        return
    idx = list(dict.fromkeys(tests["index"]))
    bnd = [b for b in bands if b in set(tests["band"])] or [""]
    pairs = pairs_of(conditions)

    def grid(col):
        return np.array([[tests.loc[(tests["index"] == m) & (tests["band"] == b), col].astype(float).mean()
                          if ((tests["index"] == m) & (tests["band"] == b)).any() else np.nan
                          for b in bnd] for m in idx])

    n_p = 1 + len(pairs)
    fig, axes = plt.subplots(1, n_p, figsize=(1.0 * len(bnd) * n_p + 4.5, 0.38 * len(idx) + 2.4),
                             squeeze=False)
    panels = [("kendall_w", "p_fdr_bands", "p", "Friedman: Kendall's W", "viridis", (0, 1))]
    panels += [(f"r_{pair_name(a, c)}", f"q_{pair_name(a, c)}", f"p_{pair_name(a, c)}",
                f"{_cond_label(c, labels)} vs {_cond_label(a, labels)}\nrank-biserial r", "RdBu_r", (-1, 1))
               for a, c in pairs]
    for k, (ax, (val, qcol, pcol, ttl, cmap, lim)) in enumerate(zip(axes[0], panels)):
        V, Q, P = grid(val), grid(qcol), grid(pcol)
        im = ax.imshow(V, cmap=cmap, vmin=lim[0], vmax=lim[1], aspect="auto")
        for i in range(len(idx)):
            for j in range(len(bnd)):
                s = _stars(Q[i, j]) if np.isfinite(Q[i, j]) else ""
                if not s and np.isfinite(P[i, j]) and P[i, j] < alpha:
                    s = "+"
                if s:
                    ax.text(j, i, s, ha="center", va="center", fontsize=9, color="k" if k or V[i, j] > 0.6 else "w",
                            fontweight="bold")
        ax.set_xticks(range(len(bnd)))
        ax.set_xticklabels(bnd, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([(index_labels or {}).get(m, m) for m in idx] if k == 0 else [], fontsize=9)
        ax.set_title(ttl.replace("\n", "\n"), fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.6)
    fig.suptitle(title, fontsize=BASE_FONT)
    fig.text(0.5, -0.01, "Stars: FDR q (Friedman: across the bands of each index; pairs: across pairs × bands of "
             "each index): * <0.05 ** <0.01 *** <0.001;  + = uncorrected p < 0.05 only. "
             "r > 0 (red): the second condition of the pair is higher.", ha="center", va="top", fontsize=8)
    _save(fig, out_png)


# ------------------------------------------------------------------ CURVES
def curve_tests(curves, conditions):
    """curves: {cond: DataFrame (subjects x points)}. Friedman per point, FDR across points."""
    common = None
    for c in conditions:
        idx = set(curves[c].dropna(how="all").index) if c in curves else set()
        common = idx if common is None else common & idx
    common = sorted(common or [])
    if len(common) < 3:
        return None, None
    X = np.stack([curves[c].loc[common].values for c in conditions], -1)   # (n, x, k)
    p = np.full(X.shape[1], np.nan)
    for j in range(X.shape[1]):
        w = pd.DataFrame(X[:, j, :]).dropna()
        if len(w) >= 3:
            p[j] = friedman_row(w)["p"]
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        q[ok] = fdr_bh(p[ok])
    return p, q


def plot_curves(ax, x, curves, conditions, colors, labels=None, logx=False, logy=False, test=True,
                alpha=0.05, sem=True):
    """
    curves: {cond: DataFrame index=subject, columns=x}. Mean +- SEM per condition; at the bottom a
    black bar where Friedman FDR q < alpha (grey where only p < alpha).
    """
    for c in conditions:
        if c not in curves or curves[c].empty:
            continue
        Y = curves[c].values.astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = np.nanmean(Y, 0)
            s = np.nanstd(Y, 0, ddof=1) / np.sqrt(np.sum(np.isfinite(Y), 0))
        ax.plot(x, m, color=colors[c], lw=2, label=f"{_cond_label(c, labels).replace(chr(10), ' ')} (n={len(Y)})")
        if sem:
            ax.fill_between(x, m - s, m + s, color=colors[c], alpha=0.2, lw=0)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    if test:
        p, q = curve_tests(curves, conditions)
        if p is not None:
            y0, y1 = ax.get_ylim()
            if logy:
                ybar = 10 ** (np.log10(y0) + 0.02 * (np.log10(y1) - np.log10(y0)))
            else:
                ybar = y0 + 0.02 * (y1 - y0)
            xs = np.asarray(x, float)
            ax.scatter(xs[(p < alpha) & ~(q < alpha)], np.full(((p < alpha) & ~(q < alpha)).sum(), ybar),
                       marker="s", s=10, color="0.65", lw=0)
            ax.scatter(xs[q < alpha], np.full((q < alpha).sum(), ybar), marker="s", s=10, color="k", lw=0)
            ax.set_ylim(y0, y1)
    return ax


CURVE_NOTE = ("Mean ± SEM over subjects. Squares at the bottom: Friedman test at each point "
              "(black: FDR q < 0.05 across points; grey: uncorrected p < 0.05 only).")


# ------------------------------------------------------------------ TOPOGRAPHIES
def plot_topomaps(chan, metrics, conditions, colors, ctests, out_png, labels=None, title="", alpha=0.05,
                  cmap="viridis"):
    """
    One row per metric: mean map of every condition (same colour scale) and difference map of every
    pair (second - first); white circles = post hoc q < alpha (FDR across channels), black dots =
    uncorrected p < alpha. Title of each row: number of channels with Friedman q < alpha.
    chan: rows subject x condition x channel; metrics: list of (column, label).
    """
    import mne
    from descriptive_analysis.topography import channel_info
    names = list(dict.fromkeys(chan["channel"]))
    info, found, missing = channel_info(names)
    if len(found) < 3:
        return
    pairs = pairs_of(conditions)
    ncol = len(conditions) + len(pairs)
    metrics = [(m, l) for m, l in metrics if m in chan and chan[m].notna().any()]
    if not metrics:
        return
    fig, axes = plt.subplots(len(metrics), ncol, figsize=(3.0 * ncol + 1.5, 3.0 * len(metrics)), squeeze=False)
    for r, (m, lab) in enumerate(metrics):
        means = {c: chan[chan.condition == c].groupby("channel")[m].mean().reindex(found).values
                 for c in conditions}
        allv = np.concatenate([v[np.isfinite(v)] for v in means.values()])
        if allv.size == 0:
            continue
        vlim = (np.percentile(allv, 2), np.percentile(allv, 98))
        t = ctests[ctests.metric == m].set_index("channel").reindex(found) if ctests is not None and len(ctests) else None
        for k, c in enumerate(conditions):
            ax = axes[r][k]
            im, _ = mne.viz.plot_topomap(np.nan_to_num(means[c], nan=np.nanmean(means[c])), info, axes=ax,
                                         show=False, cmap=cmap, vlim=vlim, contours=4, sensors=True)
            ax.set_title(_cond_label(c, labels).replace("\n", " ") if r == 0 else "", color=colors[c],
                         fontsize=BASE_FONT - 3)
            if k == 0:
                nfr = int((t["p_fdr"] < alpha).sum()) if t is not None else 0
                ax.text(-0.3, 0.5, f"{lab}\n(Friedman q<{alpha}: {nfr} ch)", transform=ax.transAxes,
                        rotation=90, ha="center", va="center", fontsize=BASE_FONT - 5)
            if k == len(conditions) - 1:
                fig.colorbar(im, ax=ax, shrink=0.7)
        for k, (a, c) in enumerate(pairs):
            ax = axes[r][len(conditions) + k]
            diff = means[c] - means[a]
            lim = np.nanmax(np.abs(diff)) if np.isfinite(diff).any() and np.nanmax(np.abs(diff)) > 0 else 1
            im, _ = mne.viz.plot_topomap(np.nan_to_num(diff), info, axes=ax, show=False, cmap="RdBu_r",
                                         vlim=(-lim, lim), contours=4, sensors="k.")
            if t is not None:
                q = t[f"q_{pair_name(a, c)}"].values
                p = t[f"p_{pair_name(a, c)}"].values
                sens = [ln for ln in ax.lines if ln.get_xydata().shape[0] == len(found)]
                if sens:
                    xy = sens[0].get_xydata()
                    sens[0].set_markersize(2)
                    sens[0].set_color("0.35")
                    sq = np.nan_to_num(q, nan=1) < alpha
                    sp = (np.nan_to_num(p, nan=1) < alpha) & ~sq
                    ax.plot(xy[sp, 0], xy[sp, 1], "o", color="k", ms=4, zorder=5)
                    ax.plot(xy[sq, 0], xy[sq, 1], "o", mfc="w", mec="k", mew=1.2, ms=9, zorder=6)
            ax.set_title(f"{_cond_label(c, labels)} − {_cond_label(a, labels)}".replace("\n", " ") if r == 0 else "",
                         fontsize=BASE_FONT - 4)
            fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(title)
    fig.text(0.5, -0.01, f"Difference maps: white circles = q < {alpha} (Wilcoxon signed-rank per channel, "
             f"FDR across channels); black dots = p < {alpha} uncorrected. Row label: channels with Friedman "
             f"q < {alpha} (FDR across channels).", ha="center", fontsize=BASE_FONT - 5)
    _save(fig, out_png)
