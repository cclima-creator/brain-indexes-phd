# -*- coding: utf-8 -*-
"""
Regional synchrony, IPS database: within-region and between-region values of every synchrony index,
to see WHERE on the scalp the group differences are located (functional connectivity by region).

Indices
  global synchrony (GS) and metastability (M)    regional Kuramoto (synchrony_criticality/regional.py),
                                                 recomputed from the phases of the .set files
  PLV, PPC, mean phase coherence (PC), PLI,      regional means of the subject matrices saved by
  wPLI, debiased wPLI, imaginary coherence       MAIN_IPS_SYNC.py in syncron_results/subjects/
Regions: config_ips_sync.REGIONS. "within r" = pairs inside region r; "a-b" = pairs between a and b.

Statistics: Mann-Whitney per band, index and region cell (6 within + 15 between = 21 cells);
p_fdr = Benjamini-Hochberg over the 21 cells of that band and index (question: where, for this
index in this band?). Effect size: rank-biserial r (> 0: photosensitive higher).

Outputs in syncron_results/regions/ (occipital ROI: not meaningful, all channels are needed):
  regional_synchrony.xlsx        subject x band x index x cell values, tests, group medians
  fig_regions_<index>.png        rows = bands; group-mean region matrices and effect-size matrix
                                 (diagonal = within-region, off-diagonal = between-region)
  fig_regions_overview.png       effect size of every index x band, cells with p_fdr < 0.05 marked
Run MAIN_IPS_SYNC.py first (it creates the subject caches with the matrices).
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

from photosensitive import config_ips as base
from photosensitive import config_ips_sync as cfg
from photosensitive.io_eeg import discover_files, load_epochs, get_data_uv
from photosensitive.report import load_results, save_tables_excel, check_output_dir
from descriptive_analysis import plots
from descriptive_analysis.plots import _save, _label
from descriptive_analysis.summary_stats import fdr_bh
from synchrony_criticality.phase import band_analytic, drop_flat_channels
from synchrony_criticality.regional import region_indices, region_pairs, within_between, regional_kuramoto

warnings.filterwarnings("ignore")

INDICES = [("gs", "global synchrony (GS)"), ("metastability", "metastability (M)"),
           ("plv", "phase locking value"), ("ppc", "PPC (unbiased PLV²)"), ("pc", "mean phase coherence"),
           ("pli", "phase lag index"), ("wpli", "weighted PLI"), ("dwpli", "debiased wPLI"),
           ("icoh", "imaginary coherence")]
MATRIX_INDICES = ["plv", "ppc", "pc", "pli", "wpli", "dwpli", "icoh"]
SHORT = {"frontopolar": "Fp", "frontal": "F", "central": "C", "temporal": "T", "parietal": "P", "occipital": "O"}


def cell_name(a, b):
    return f"within {a}" if a == b else f"{a}-{b}"


# ================================================================== ONE SUBJECT
def regional_kuramoto_subject(fpath, ch_names, bands):
    """Regional GS / M of one subject in every band, from the band-passed phases."""
    ep = load_epochs(fpath, base.FIXED_LENGTH_S, base.ANALYSIS_WINDOW)
    data = get_data_uv(ep)
    keep = drop_flat_channels(data)
    data, names = data[:, keep], [ep.ch_names[i] for i in keep]
    idx = region_indices(names, cfg.REGIONS)
    out = {}
    for b, band in bands.items():
        phases = np.angle(band_analytic(data, ep.info["sfreq"], band, cfg.FILTER_ORDER, cfg.EDGE_TRIM_S))
        out[b] = regional_kuramoto(phases, idx)
    return out


def subject_rows(cache, fpath, group):
    """Long rows (group, subject, band, index, region_a, region_b, value) for one subject."""
    res = load_results(cache)
    idx = region_indices(res["ch_names"], cfg.REGIONS)
    rows = []
    kur = regional_kuramoto_subject(fpath, res["ch_names"], cfg.BANDS)
    for b in cfg.BANDS:
        det = res["details"][b]
        for m in MATRIX_INDICES:
            if m in det:
                for (ra, rb), v in within_between(det[m], idx).items():
                    rows.append(dict(group=group, subject=fpath.stem, band=b, index=m, region_a=ra,
                                     region_b=rb, value=v))
        for (ra, rb), (gs, ms) in kur[b].items():
            rows += [dict(group=group, subject=fpath.stem, band=b, index="gs", region_a=ra, region_b=rb, value=gs),
                     dict(group=group, subject=fpath.stem, band=b, index="metastability", region_a=ra,
                          region_b=rb, value=ms)]
    return rows


# ================================================================== STATISTICS
def cell_tests(long, groups):
    """Mann-Whitney per band x index x cell; FDR over the cells of each band x index; rank-biserial r."""
    out = []
    for (b, m), d in long.groupby(["band", "index"], sort=False):
        part = []
        for (ra, rb), dc in d.groupby(["region_a", "region_b"], sort=False):
            x = [dc.loc[dc.group == g, "value"].dropna().values for g in groups]
            if min(len(v) for v in x) < 2:
                continue
            u = stats.mannwhitneyu(*x)
            part.append(dict(band=b, index=m, region_a=ra, region_b=rb, cell=cell_name(ra, rb),
                             kind="within" if ra == rb else "between", p=u.pvalue,
                             r_rank_biserial=2 * u.statistic / (len(x[0]) * len(x[1])) - 1,
                             **{f"median_{g}": float(np.median(v)) for g, v in zip(groups, x)}))
        if part:
            for r, q in zip(part, fdr_bh(np.array([r["p"] for r in part]))):
                r["p_fdr"] = q
            out += part
    return pd.DataFrame(out)


# ================================================================== FIGURES
def _matrix(df, regions, col):
    """Symmetric region x region matrix from the cell rows of one band x index."""
    M = np.full((len(regions), len(regions)), np.nan)
    pos = {r: k for k, r in enumerate(regions)}
    for _, r in df.iterrows():
        i, j = pos[r.region_a], pos[r.region_b]
        M[i, j] = M[j, i] = r[col]
    return M


def _ticks(ax, regions, ylabels=True):
    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels([SHORT.get(r, r) for r in regions], fontsize=10)
    ax.set_yticks(range(len(regions)))
    ax.set_yticklabels([SHORT.get(r, r) for r in regions] if ylabels else [], fontsize=10)


def _marks(ax, t, regions, alpha=0.05):
    """o = p_fdr < alpha (white ring), · = p < alpha uncorrected."""
    pos = {r: k for k, r in enumerate(regions)}
    for _, r in t.iterrows():
        i, j = pos[r.region_a], pos[r.region_b]
        for (y, x) in {(i, j), (j, i)}:
            if r.p_fdr < alpha:
                ax.plot(x, y, "o", mfc="none", mec="k", mew=1.6, ms=13)
            elif r.p < alpha:
                ax.plot(x, y, ".", color="k", ms=6)


def plot_index(long, tests, index, label, regions, colors, out_png):
    """Rows = bands: mean matrix of each group (same scale) and rank-biserial r with significance."""
    groups = [g for g in colors if g in set(long.group)]
    bands = list(cfg.BANDS)
    fig, axes = plt.subplots(len(bands), 3, figsize=(11.5, 3.3 * len(bands)), squeeze=False)
    for i, b in enumerate(bands):
        d = long[(long.band == b) & (long["index"] == index)]
        means = {g: _matrix(d[d.group == g].groupby(["region_a", "region_b"], sort=False).value.mean()
                            .reset_index(), regions, "value") for g in groups}
        vals = np.concatenate([m[np.isfinite(m)] for m in means.values()])
        vmin, vmax = (np.percentile(vals, 2), np.percentile(vals, 98)) if vals.size else (0, 1)
        for c, g in enumerate(groups):
            ax = axes[i, c]
            im = ax.imshow(means[g], cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(f"{_label(g)}, {b}", color=colors[g], fontsize=plots.BASE_FONT - 2)
            _ticks(ax, regions, c == 0)
            fig.colorbar(im, ax=ax, fraction=0.046)
        t = tests[(tests.band == b) & (tests["index"] == index)]
        ax = axes[i, 2]
        im = ax.imshow(_matrix(t, regions, "r_rank_biserial"), cmap="RdBu_r", vmin=-1, vmax=1)
        _marks(ax, t, regions)
        ax.set_title(f"effect size r, {b}", fontsize=plots.BASE_FONT - 2)
        _ticks(ax, regions, False)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{label}: within-region (diagonal) and between-region (off-diagonal)\n"
                 f"r > 0: {_label(groups[0])} higher; o p_fdr < 0.05 (FDR over the 21 cells), · p < 0.05 uncorrected")
    _save(fig, out_png)


def plot_overview(tests, regions, out_png):
    """Effect-size matrix of every index (rows) x band (columns), significant cells marked."""
    bands = list(cfg.BANDS)
    fig, axes_all = plt.subplots(len(INDICES), len(bands) + 1, figsize=(2.5 * len(bands) + 1.2, 2.5 * len(INDICES)),
                                 squeeze=False, gridspec_kw=dict(width_ratios=[1] * len(bands) + [0.08]))
    axes = axes_all[:, :-1]
    for ax in axes_all[:, -1]:                     # last narrow column: only the colorbar
        ax.axis("off")
    for i, (m, lab) in enumerate(INDICES):
        for j, b in enumerate(bands):
            ax = axes[i, j]
            t = tests[(tests.band == b) & (tests["index"] == m)]
            im = ax.imshow(_matrix(t, regions, "r_rank_biserial"), cmap="RdBu_r", vmin=-1, vmax=1)
            _marks(ax, t, regions)
            _ticks(ax, regions, j == 0)
            if i == 0:
                ax.set_title(b, fontsize=plots.BASE_FONT)
            if j == 0:
                ax.set_ylabel(lab, fontsize=plots.BASE_FONT - 3)
    cax = axes_all[len(INDICES) // 2, -1]
    cax.axis("on")
    fig.colorbar(im, cax=cax, label="rank-biserial r\n(photosensitive higher > 0)")
    fig.suptitle("Where are the group differences? Within-region (diagonal) and between-region synchrony\n"
                 "o p_fdr < 0.05 (FDR over the 21 region cells of each index and band), · p < 0.05 uncorrected",
                 y=1.0)
    _save(fig, out_png)


# ================================================================== MAIN
def main():
    out_sync = cfg.results_dir("all", base.OUT_DIR)
    cache_dir = out_sync / "subjects"
    if not cache_dir.exists():
        raise SystemExit("syncron_results/subjects not found: run MAIN_IPS_SYNC.py (all channels) first")
    out = check_output_dir(out_sync / "regions", ["regional_synchrony.xlsx"])
    print("Output folder:", out)
    rows = []
    for group, folder in cfg.GROUPS.items():
        for f in base.subject_files(group, discover_files(folder)):
            cache = cache_dir / f"{f.stem}.pkl"
            if not cache.exists():
                print("  no synchrony cache for", f.stem, "- skipped")
                continue
            print("  ", f.stem)
            rows += subject_rows(cache, f, group)
    long = pd.DataFrame(rows)
    groups = [g for g in cfg.GROUPS if g in set(long.group)]
    regions = [r for r in cfg.REGIONS if r in set(long.region_a)]
    long["cell"] = [cell_name(a, b) for a, b in zip(long.region_a, long.region_b)]
    tests = cell_tests(long, groups)
    medians = long.groupby(["band", "index", "cell", "group"], sort=False).value.median().unstack("group").reset_index()

    sig = tests[tests.p_fdr < 0.05]
    print(f"\n=== Region cells with p_fdr < 0.05 (FDR over 21 cells per band and index): {len(sig)} ===")
    if len(sig):
        print(sig[["band", "index", "cell", "r_rank_biserial", "p", "p_fdr"]].round(4).to_string(index=False))
    params = pd.DataFrame([dict(parameter=k, value=str(v)) for k, v in
                           dict(REGIONS=cfg.REGIONS, **cfg.params()).items()])
    save_tables_excel({"tests": tests, "group_medians": medians, "subject_values": long, "parameters": params},
                      out / "regional_synchrony.xlsx")

    colors = {g: c for g, c in cfg.COLORS.items() if g in groups}
    plots.paper_style()
    for m, lab in INDICES:
        plot_index(long, tests, m, lab, regions, colors, out / f"fig_regions_{m}.png")
    plot_overview(tests, regions, out / "fig_regions_overview.png")
    print("\nDone. Tables and figures in:", out)


if __name__ == "__main__":
    main()
