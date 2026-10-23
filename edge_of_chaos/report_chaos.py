# -*- coding: utf-8 -*-
"""
Tables of the edge-of-chaos analysis and group comparisons (from the per-subject results of
MAIN_chaos.analyze_subject).

  subject_metrics  one row per subject: channel means of every measure + whole-montage measures
  channel_metrics  one row per subject x channel
  epoch_metrics    one row per subject x epoch (channel means of the per-epoch measures)
  group_tests      Mann-Whitney (2 groups) / Kruskal-Wallis per measure, FDR across measures
                   (all measures x bands together)
  group_summary    n, median, IQR, mean, SD per group and measure
  channel_tests    per measure and channel, FDR across channels (topographies)
  mse_scale_tests  MSE (channel mean) compared at every scale, FDR across scales within band
"""
import numpy as np
import pandas as pd

from descriptive_analysis.summary_stats import compare_groups, channel_group_tests, fdr_bh
from descriptive_analysis.topography import unify_channel_names

META = {"group", "subject", "channel", "epoch", "label", "n_channels", "n_epochs", "duration_s",
        "sfreq", "aperiodic_offset", "aperiodic_r2"}
# diagnostics of the covariance width (kept in subject_metrics, not tested or plotted)
META_PREFIX = ("cov_mean_diag_", "cov_sd_offdiag_", "cov_mean_offdiag_", "event_rate_")


def metric_columns(df):
    """Numeric columns that are measures (not bookkeeping)."""
    return [c for c in df.columns if c not in META and not c.startswith(META_PREFIX)
            and pd.api.types.is_numeric_dtype(df[c])]


def group_summary(df, metrics, group_col="group"):
    rows = []
    for g, d in df.groupby(group_col, sort=False):
        for m in metrics:
            x = d[m].replace([np.inf, -np.inf], np.nan).dropna()
            rows.append(dict(group=g, metric=m, n=len(x), median=x.median(), q25=x.quantile(0.25),
                             q75=x.quantile(0.75), mean=x.mean(), sd=x.std()))
    return pd.DataFrame(rows)


def mse_table(results):
    """Long table subject x signal band x scale of the MSE curve averaged over channels."""
    rows = []
    for r in results:
        d = r["detail"]
        for key, curve in d.items():
            if not key.startswith("mse_curve"):
                continue
            band = key[len("mse_curve_"):] or "broadband"
            curve = np.nanmean(curve, axis=0)
            rows += [dict(group=r["row"]["group"], subject=r["row"]["subject"], band=band,
                          scale=int(s), mse=float(v)) for s, v in zip(d["mse_scales"], curve)]
    return pd.DataFrame(rows)


def mse_scale_tests(mse_long):
    """MSE compared between groups at every scale; p_fdr = FDR across scales within each band."""
    if mse_long.empty:
        return pd.DataFrame()
    out = []
    for b, db in mse_long.groupby("band", sort=False):
        part = []
        for s, d in db.groupby("scale"):
            t = compare_groups(d, ["mse"])
            if len(t):
                t.insert(0, "scale", s)
                t.insert(0, "band", b)
                part.append(t.drop(columns="p_fdr"))
        if part:
            t = pd.concat(part, ignore_index=True)
            t["p_fdr"] = fdr_bh(t["p"].values)
            out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def build_tables(results, cfg=None):
    """All tables from the list of per-subject results."""
    subj = pd.DataFrame([r["row"] for r in results])
    chan = pd.DataFrame([c for r in results for c in r["chans"]])
    epochs = pd.DataFrame([e for r in results for e in r["epochs"]])
    metrics = metric_columns(subj)
    chan_u, _ = unify_channel_names(chan)
    mse_long = mse_table(results)
    return dict(subject_metrics=subj,
                group_tests=compare_groups(subj, metrics),
                group_summary=group_summary(subj, metrics),
                channel_tests=channel_group_tests(chan_u, metric_columns(chan_u)),
                channel_metrics=chan,
                epoch_metrics=epochs,
                mse_long=mse_long,
                mse_scale_tests=mse_scale_tests(mse_long))
