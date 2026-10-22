# -*- coding: utf-8 -*-
"""
Stationarity with eyes open and eyes closed, IPS database.

Every epoch (times -1 ... 11 s, as in mne ep.times) is split into two windows:
    eyes open    EO = [1, 6) s
    eyes closed  EC = [6, 11) s
and in each window the Reverse Arrangement Test of the preliminary analysis is applied
(descriptive_analysis.summary_stats.stationarity_test: trend of the variance of STAT_N_SEGMENTS
consecutive segments; p < STAT_ALPHA = epoch x channel non-stationary). Unfiltered signal in uV,
same files, groups, excluded subjects and channel choice as MAIN_IPS.py.

Statistics
  - fraction of non-stationary epoch x channel per subject, per condition
  - between groups, per condition: Mann-Whitney on the fractions; KS permutation test on the
    distributions of p (labels permuted between subjects, descriptive_analysis.ks_permutation_test)
  - within group, EO vs EC: Wilcoxon signed-rank on the fractions (paired by subject)

Outputs in <OUT_DIR>/stationarity_eyes (or <OUT_DIR>/occipital/stationarity_eyes):
  stationarity_eyes.xlsx               epoch x channel p, per-subject fractions, tests
  fig_stationarity_eyes_open.png       fraction non-stationary + cumulative probability of p (as fig10)
  fig_stationarity_eyes_closed.png     same for eyes closed
  fig_stationarity_eyes_compare.png    EO vs EC side by side: paired fractions and CDFs of both
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from photosensitive import config_ips as cfg
from photosensitive.io_eeg import discover_files, load_epochs, get_data_uv, epoch_labels
from photosensitive.report import check_output_dir, save_tables_excel, FALLBACK_DIR
from photosensitive.MAIN_IPS import pick_channels
from descriptive_analysis.summary_stats import stationarity_test, compare_groups, ks_permutation_test
from descriptive_analysis import plots
from descriptive_analysis.plots import _save, _label, _p_str, BASE_FONT

warnings.filterwarnings("ignore")

CONDITIONS = {"eyes_open": (1.0, 6.0), "eyes_closed": (6.0, 11.0)}   # seconds, ep.times
COND_LABELS = {"eyes_open": "eyes open (1-6 s)", "eyes_closed": "eyes closed (6-11 s)"}
OUT_SUBDIR = "stationarity_eyes"
OUTPUT_NAMES = ["stationarity_eyes.xlsx", "fig_stationarity_eyes_open.png",
                "fig_stationarity_eyes_closed.png", "fig_stationarity_eyes_compare.png"]


# ================================================================== ONE SUBJECT
def window(data, times, t0, t1):
    """Samples with t0 <= time < t1. data: (n_epochs, n_channels, n_times)."""
    keep = (times >= t0) & (times < t1)
    if not keep.any():
        raise ValueError(f"no samples in {t0}-{t1} s (epoch times {times[0]:.2f}-{times[-1]:.2f} s)")
    return data[..., keep]


def analyze_subject(fpath, group, channels):
    """Reverse Arrangement Test p of every epoch x channel in each condition window."""
    ep = load_epochs(fpath, cfg.FIXED_LENGTH_S, None)
    if channels is not None:
        ep = pick_channels(ep, channels, cfg.OCCIPITAL_CHANNELS)
    data, times, labels = get_data_uv(ep), ep.times, epoch_labels(ep)
    fstim = np.array([cfg.STIM_FREQ_MAP.get(l, cfg.parse_stim_freq(l)) for l in labels], float)
    rows = []
    for cond, (t0, t1) in CONDITIONS.items():
        x = window(data, times, t0, t1)
        p = stationarity_test(x, cfg.STAT_N_SEGMENTS)                 # (n_epochs, n_channels)
        n_ep, n_ch = p.shape
        rows.append(pd.DataFrame(dict(
            group=group, subject=fpath.stem, condition=cond, epoch=np.repeat(np.arange(n_ep), n_ch),
            stim_freq=np.repeat(fstim, n_ch), channel=np.tile(ep.ch_names, n_ep), stat_p=p.ravel(),
            window_s=x.shape[-1] / ep.info["sfreq"])))
    return pd.concat(rows, ignore_index=True)


# ================================================================== TABLES
def subject_fractions(ec):
    """Fraction of non-stationary epoch x channel (p < STAT_ALPHA) per subject and condition."""
    ec = ec.assign(nonstationary=ec.stat_p < cfg.STAT_ALPHA)
    return (ec.groupby(["group", "subject", "condition"], sort=False)
            .agg(frac_nonstationary=("nonstationary", "mean"), n_epoch_channel=("stat_p", "size"),
                 median_p=("stat_p", "median"))
            .reset_index())


def condition_tests(ec, frac, n_perm):
    """Between groups per condition (Mann-Whitney, KS permutation) and EO vs EC per group (Wilcoxon)."""
    between, ks_rows, within = [], [], []
    for cond in CONDITIONS:
        f = frac[frac.condition == cond]
        t = compare_groups(f, ["frac_nonstationary"])
        if len(t):
            between.append(t.drop(columns="p_fdr").assign(condition=cond))
        ks = ks_permutation_test(ec[ec.condition == cond], "stat_p", n_perm=n_perm)
        if ks:
            ks_rows.append(dict(condition=cond, **ks))
    for g, d in frac.groupby("group", sort=False):
        w = d.pivot(index="subject", columns="condition", values="frac_nonstationary").dropna()
        if len(w) >= 2 and set(CONDITIONS) <= set(w.columns):
            eo, ec_ = w["eyes_open"], w["eyes_closed"]
            p = stats.wilcoxon(eo, ec_).pvalue if (eo != ec_).any() else 1.0
            within.append(dict(group=g, test="Wilcoxon signed-rank", n=len(w), p=p,
                               median_eyes_open=eo.median(), median_eyes_closed=ec_.median()))
    return (pd.concat(between, ignore_index=True) if between else pd.DataFrame(),
            pd.DataFrame(ks_rows), pd.DataFrame(within))


# ================================================================== FIGURES
def _cdf_axis(ax, ec, colors, conds, title):
    styles = {"eyes_open": "-", "eyes_closed": "--"}
    for cond in conds:
        for g, c in colors.items():
            v = np.sort(ec.loc[(ec.group == g) & (ec.condition == cond), "stat_p"].dropna().values)
            if len(v):
                lab = _label(g) if len(conds) == 1 else f"{_label(g)}, {COND_LABELS[cond]}"
                ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", lw=2.3, color=c,
                        ls=styles[cond], label=lab)
    ax.plot([0, 1], [0, 1], color="k", ls=":", lw=1.2, label="all stationary (uniform p)")
    ax.axvline(cfg.STAT_ALPHA, color="0.6", ls=":", lw=1.2)
    ax.set(xlabel="Reverse Arrangement Test p (epoch × channel)", ylabel="Cumulative probability",
           title=title, xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(fontsize=BASE_FONT - 4, loc="lower right")


def plot_condition(ec, frac, ks, cond, colors, out_png):
    """As fig10 for one condition: fraction non-stationary per subject and CDF of p per group."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    plots._box_by_group(axes[0], frac[frac.condition == cond], "frac_nonstationary", colors,
                        f"Fraction non-stationary (p < {cfg.STAT_ALPHA})")
    axes[0].axhline(cfg.STAT_ALPHA, color="k", ls="--", lw=1.2)
    axes[0].text(0.98, cfg.STAT_ALPHA, " chance level", transform=axes[0].get_yaxis_transform(),
                 ha="right", va="bottom", fontsize=BASE_FONT - 3)
    k = ks[ks.condition == cond] if len(ks) else ks
    title = "p-value distribution"
    if len(k):
        title += f"\nKS D = {k.D.iloc[0]:.3f}, permutation {_p_str(k.p_perm.iloc[0])}"
    _cdf_axis(axes[1], ec, colors, [cond], title)
    fig.suptitle(f"Stationarity, {COND_LABELS[cond]} (variance trend in {cfg.STAT_N_SEGMENTS} segments)")
    _save(fig, out_png)


def plot_compare(ec, frac, within, colors, out_png):
    """EO vs EC: paired fractions per subject (lines) for each group, and the four CDFs together."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    ax = axes[0]
    conds = list(CONDITIONS)
    groups = [g for g in colors if g in set(frac.group)]
    rng = np.random.default_rng(0)
    for gi, g in enumerate(groups):
        w = frac[frac.group == g].pivot(index="subject", columns="condition", values="frac_nonstationary")
        xs = np.array([gi * 2.6, gi * 2.6 + 1])
        for _, r in w.iterrows():
            ax.plot(xs + rng.uniform(-0.06, 0.06), r[conds].values, color=colors[g], alpha=0.45, lw=1,
                    marker="o", ms=4)
        bp = ax.boxplot([w[c].dropna() for c in conds], positions=xs, widths=0.5, patch_artist=True,
                        showfliers=False, medianprops=dict(color="k", lw=2))
        for patch in bp["boxes"]:
            patch.set_facecolor(colors[g])
            patch.set_alpha(0.3)
        t = within[within.group == g] if len(within) else within
        if len(t):
            ax.text(xs.mean(), 0.96, f"EO vs EC {_p_str(t.p.iloc[0])}", ha="center", va="top",
                    transform=ax.get_xaxis_transform(), fontsize=BASE_FONT - 3)
    ax.set_xticks([x for gi in range(len(groups)) for x in (gi * 2.6, gi * 2.6 + 1)])
    ax.set_xticklabels([f"{_label(g).replace(' ', chr(10), 1)}\n{'EO' if c == 'eyes_open' else 'EC'}"
                        for g in groups for c in conds], fontsize=BASE_FONT - 3)
    ax.axhline(cfg.STAT_ALPHA, color="k", ls="--", lw=1.2)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.12)                       # room for the EO vs EC labels
    ax.set_ylabel(f"Fraction non-stationary (p < {cfg.STAT_ALPHA})")
    ax.set_title("Eyes open vs eyes closed (Wilcoxon, paired)")
    _cdf_axis(axes[1], ec, colors, conds, "p-value distribution (solid EO, dashed EC)")
    fig.suptitle("Stationarity, eyes open (1-6 s) vs eyes closed (6-11 s)")
    _save(fig, out_png)


# ================================================================== MAIN
def main():
    channel_set, channels, out_root = cfg.choose_channel_set()
    fallback = (FALLBACK_DIR if channels is None else FALLBACK_DIR / cfg.OCCIPITAL_SUBDIR) / OUT_SUBDIR
    out = check_output_dir(Path(out_root) / OUT_SUBDIR, OUTPUT_NAMES, fallback)
    print("Output folder:", out)
    print(f"Windows: {CONDITIONS} s | {cfg.STAT_N_SEGMENTS} segments | alpha {cfg.STAT_ALPHA}")
    parts = []
    for group, folder in cfg.GROUPS.items():
        files = cfg.subject_files(group, discover_files(folder))
        print(f"{group}: {len(files)} files")
        for f in files:
            try:
                parts.append(analyze_subject(f, group, channels))
            except Exception as e:
                print("  ", f.name, "ERROR ->", e)
    if not parts:
        raise SystemExit("No file analysed: check the paths in config_ips.py")
    ec = pd.concat(parts, ignore_index=True)
    frac = subject_fractions(ec)
    between, ks, within = condition_tests(ec, frac, cfg.KS_N_PERM)
    summary = frac.groupby(["condition", "group"], sort=False).frac_nonstationary.agg(
        ["count", "median", "mean", "std"]).reset_index()

    print("\n=== Fraction non-stationary per group ===")
    print(summary.round(3).to_string(index=False))
    print("\n=== Between groups (Mann-Whitney on fractions) ===")
    print(between.round(4).to_string(index=False) if len(between) else "none")
    print("\n=== Between groups (KS permutation on p distributions) ===")
    print(ks.round(4).to_string(index=False) if len(ks) else "none")
    print("\n=== Eyes open vs eyes closed (Wilcoxon) ===")
    print(within.round(4).to_string(index=False) if len(within) else "none")

    params = pd.DataFrame([dict(parameter=k, value=str(v)) for k, v in dict(
        CONDITIONS=CONDITIONS, STAT_N_SEGMENTS=cfg.STAT_N_SEGMENTS, STAT_ALPHA=cfg.STAT_ALPHA,
        KS_N_PERM=cfg.KS_N_PERM, channel_set=channel_set).items()])
    save_tables_excel({"subject_fractions": frac, "group_summary": summary, "tests_between_groups": between,
                       "ks_permutation": ks, "tests_eo_vs_ec": within, "epoch_channel_p": ec,
                       "parameters": params}, out / "stationarity_eyes.xlsx")

    colors = {g: c for g, c in cfg.COLORS.items() if g in set(ec.group)}
    plots.paper_style()
    for cond in CONDITIONS:
        plot_condition(ec, frac, ks, cond, colors, out / f"fig_stationarity_{cond}.png")
    plot_compare(ec, frac, within, colors, out / "fig_stationarity_eyes_compare.png")
    print("\nDone. Tables and figures in:", out)


if __name__ == "__main__":
    main()
