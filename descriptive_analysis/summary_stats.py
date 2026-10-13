# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 14:11:51 2026

@author: Chiara Clima
"""

"""Statistiche descrittive, fit gaussiano e confronti tra gruppi (su tabelle pandas)."""
import numpy as np
import pandas as pd
from scipy import stats


def describe(x):
    x = np.asarray(x, dtype=float)                 ##converte x in numeri decimali
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(n=0, mean=np.nan, median=np.nan, sd=np.nan, skew=np.nan, kurtosis=np.nan)
    return dict(n=len(x), mean=x.mean(), median=np.median(x),
                sd=x.std(ddof=1) if len(x) > 1 else np.nan,
                skew=stats.skew(x) if len(x) > 2 else np.nan,           ##Asimmetria della distribuzione
                kurtosis=stats.kurtosis(x) if len(x) > 3 else np.nan)    ##importanza delle code


def gaussian_fit(x):
    """(mu, sigma) della gaussiana che meglio approssima x."""
    x = np.asarray(x, dtype=float)
    return stats.norm.fit(x[np.isfinite(x)])


def shapiro_p(x):                                ##how close to a nomal distribution
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return stats.shapiro(x).pvalue if len(x) >= 3 else np.nan


def band_group_summary(long, subj, group_col="group"):
    """
    Riassunto per gruppo e banda: media/mediana/sd (livello soggetto) della potenza
    assoluta, relativa e log10; fit gaussiano su log10 (canali x soggetti); skewness
    lineare vs log10; Shapiro sui soggetti (lineare vs log10).
    long: tabella per canale; subj: tabella per soggetto (media sui canali).
    """
    rows = []
    for (g, b), s in subj.groupby([group_col, "band"]):
        ch = long[(long[group_col] == g) & (long.band == b)]
        mu, sd = gaussian_fit(ch.log10_abs)
        a, r, lg = describe(s.abs_uV2), describe(s.rel), describe(s.log10_abs)
        rows.append(dict(
            group=g, band=b, n_subjects=a["n"],
            abs_mean=a["mean"], abs_median=a["median"], abs_sd=a["sd"],
            rel_mean=r["mean"], rel_median=r["median"], rel_sd=r["sd"],
            log10_mean=lg["mean"], log10_median=lg["median"], log10_sd=lg["sd"],
            gauss_mu_log10_channels=mu, gauss_sigma_log10_channels=sd,
            skew_channels_linear=stats.skew(ch.abs_uV2),
            skew_channels_log10=stats.skew(ch.log10_abs),
            shapiro_p_subjects_linear=shapiro_p(s.abs_uV2),
            shapiro_p_subjects_log10=shapiro_p(s.log10_abs)))
    return pd.DataFrame(rows)


def fdr_bh(p):
    """Correzione Benjamini-Hochberg dei p-value."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1.0)
    return out


def compare_groups(df, metrics, group_col="group"):
    """
    Confronto tra gruppi per ogni colonna in `metrics` (una riga per soggetto in df).
    2 gruppi: Mann-Whitney U; piu' gruppi: Kruskal-Wallis. Aggiunge p corretto (FDR-BH).
    """
    groups = list(df[group_col].dropna().unique())
    rows = []
    for m in metrics:
        samples = [df.loc[df[group_col] == g, m].dropna().values for g in groups]
        if len(groups) < 2 or any(len(s) < 2 for s in samples):
            continue
        if len(groups) == 2:
            test, p = "Mann-Whitney U", stats.mannwhitneyu(*samples).pvalue
        else:
            test, p = "Kruskal-Wallis", stats.kruskal(*samples).pvalue
        row = dict(metric=m, test=test, p=p)
        for g, s in zip(groups, samples):
            row[f"n_{g}"] = len(s)
            row[f"median_{g}"] = float(np.median(s))
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out["p_fdr"] = fdr_bh(out["p"].values)
    return out