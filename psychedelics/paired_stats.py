# -*- coding: utf-8 -*-
"""
Repeated-measures statistics (same subjects in every condition), on long tables with one row per
subject x condition [x band] and one column per index.

  friedman_row      Friedman chi2 across the conditions (only subjects with all conditions),
                    Kendall's W = chi2 / (n (k - 1)) as effect size
  wilcoxon_pair     Wilcoxon signed-rank test on one pair of conditions, matched-pairs
                    rank-biserial r (positive = second condition higher)
  index_band_tests  one row per index x band: Friedman + the three post hoc pairs.
                    FDR (Benjamini-Hochberg):
                      p_fdr_bands   Friedman p across the bands of that index (as in the figures)
                      p_fdr_all     Friedman p across every index x band of the table
                      q_<a>_vs_<b>  post hoc p across the pairs x bands of that index (figures)
  summary_table     n, median, IQR, mean, SD per condition and index x band
"""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from descriptive_analysis.summary_stats import fdr_bh


def paired_matrix(df, value, conditions, subject_col="subject", cond_col="condition"):
    """(n_subjects, n_conditions) of value, only subjects with a finite value in every condition."""
    w = (df.replace([np.inf, -np.inf], np.nan)
         .pivot_table(index=subject_col, columns=cond_col, values=value, aggfunc="mean"))
    w = w.reindex(columns=conditions).dropna()
    return w


def friedman_row(w):
    """Friedman test on a complete (n, k) table. Returns dict(n, chi2, p, kendall_w)."""
    n, k = w.shape
    if n < 3 or k < 3:
        return dict(n=n, chi2=np.nan, p=np.nan, kendall_w=np.nan)
    x = w.values
    if np.allclose(x, x[:, :1]):                        # identical in every condition
        return dict(n=n, chi2=0.0, p=1.0, kendall_w=0.0)
    chi2, p = stats.friedmanchisquare(*x.T)
    return dict(n=n, chi2=float(chi2), p=float(p), kendall_w=float(chi2 / (n * (k - 1))))


def wilcoxon_pair(a, b):
    """Wilcoxon signed-rank test b vs a (paired arrays). Returns (p, rank-biserial r, median diff)."""
    d = np.asarray(b, float) - np.asarray(a, float)
    d = d[np.isfinite(d)]
    if d.size < 3:
        return np.nan, np.nan, np.nan
    nz = d[d != 0]
    if nz.size == 0:
        return 1.0, 0.0, 0.0
    p = stats.wilcoxon(nz).pvalue
    r = stats.rankdata(np.abs(nz))
    rb = (r[nz > 0].sum() - r[nz < 0].sum()) / r.sum()
    return float(p), float(rb), float(np.median(d))


def pairs_of(conditions):
    return list(combinations(conditions, 2))


def pair_name(a, b):
    return f"{a}_vs_{b}"


def index_band_tests(long, indices, conditions, bands=None, band_col="band"):
    """
    long: one row per subject x condition [x band], columns = indices.
    bands None = no band column (one test per index). Returns one row per index x band.
    """
    rows = []
    band_list = [None] if bands is None else bands
    for m in indices:
        if m not in long:
            continue
        for b in band_list:
            d = long if b is None else long[long[band_col] == b]
            w = paired_matrix(d, m, conditions)
            if w.empty:
                continue
            row = dict(index=m, band=b if b is not None else "", **friedman_row(w))
            for c in conditions:
                row[f"median_{c}"] = float(w[c].median())
            for a, c in pairs_of(conditions):
                p, rb, md = wilcoxon_pair(w[a], w[c])
                row[f"p_{pair_name(a, c)}"] = p
                row[f"r_{pair_name(a, c)}"] = rb
                row[f"diff_{pair_name(a, c)}"] = md
            rows.append(row)
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t["p_fdr_bands"] = np.nan
    pcols = [f"p_{pair_name(a, c)}" for a, c in pairs_of(conditions)]
    for c in pcols:
        t["q" + c[1:]] = np.nan
    for m, g in t.groupby("index", sort=False):
        ok = g["p"].notna()
        if ok.any():
            t.loc[g.index[ok], "p_fdr_bands"] = fdr_bh(g.loc[ok, "p"].values)
        # post hoc: one FDR over every pair x band of this index
        vals = g[pcols].values.ravel()
        fin = np.isfinite(vals)
        q = np.full(vals.shape, np.nan)
        if fin.any():
            q[fin] = fdr_bh(vals[fin])
        t.loc[g.index, ["q" + c[1:] for c in pcols]] = q.reshape(len(g), len(pcols))
    ok = t["p"].notna()
    t["p_fdr_all"] = np.nan
    t.loc[ok, "p_fdr_all"] = fdr_bh(t.loc[ok, "p"].values)
    first = ["index", "band", "n", "chi2", "kendall_w", "p", "p_fdr_bands", "p_fdr_all"]
    return t[first + [c for c in t.columns if c not in first]]


def summary_table(long, indices, conditions, bands=None, band_col="band"):
    """Descriptive statistics per condition x index [x band] (all available subjects)."""
    rows = []
    for m in indices:
        if m not in long:
            continue
        for b in ([None] if bands is None else bands):
            d = long if b is None else long[long[band_col] == b]
            for c in conditions:
                x = d.loc[d.condition == c, m].replace([np.inf, -np.inf], np.nan).dropna()
                if not len(x):
                    continue
                rows.append(dict(index=m, band=b if b is not None else "", condition=c, n=len(x),
                                 median=x.median(), q25=x.quantile(.25), q75=x.quantile(.75),
                                 mean=x.mean(), sd=x.std()))
    return pd.DataFrame(rows)


def wide_to_long(df, index, bands, id_cols=("condition", "subject")):
    """Columns <index>_<band> -> long table (id_cols, band, <index>)."""
    parts = [df[list(id_cols)].assign(band=b, **{index: df[f"{index}_{b}"]})
             for b in bands if f"{index}_{b}" in df]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def channel_tests(chan, metrics, conditions):
    """
    Per channel (topographies): Friedman across conditions and Wilcoxon per pair, FDR across
    channels for each metric (Friedman) and each metric x pair (post hoc).
    chan: one row per subject x condition x channel.
    """
    out = []
    for m in metrics:
        rows = []
        for ch, d in chan.groupby("channel", sort=False):
            w = paired_matrix(d, m, conditions)
            if w.empty:
                continue
            r = dict(metric=m, channel=ch, **friedman_row(w))
            for a, c in pairs_of(conditions):
                p, rb, md = wilcoxon_pair(w[a], w[c])
                r[f"p_{pair_name(a, c)}"], r[f"r_{pair_name(a, c)}"], r[f"diff_{pair_name(a, c)}"] = p, rb, md
            rows.append(r)
        if not rows:
            continue
        t = pd.DataFrame(rows)
        for col in ["p"] + [f"p_{pair_name(a, c)}" for a, c in pairs_of(conditions)]:
            ok = t[col].notna()
            t["q" + col[1:] if col != "p" else "p_fdr"] = np.nan
            if ok.any():
                t.loc[ok, "q" + col[1:] if col != "p" else "p_fdr"] = fdr_bh(t.loc[ok, col].values)
        out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
