# -*- coding: utf-8 -*-
"""
Drug x time comparisons (Ayahuasca vs placebo, same subjects, 3 time points each), on the long
tables of every module (one row per subject x condition [x band], one column per index).

Per index x band, subjects with all six values:
  interaction  Friedman test on the difference Ayahuasca - placebo across the 3 time points
               ("does the drug effect change over time?"), Kendall's W
  drug         Wilcoxon signed-rank on the mean over the 3 time points, Ayahuasca vs placebo
  time_aya / time_pla   Friedman across time within each arm (as in analyses 1-4)
  t1..t3       Wilcoxon Ayahuasca vs placebo at each time point (r > 0: Ayahuasca higher)
FDR (Benjamini-Hochberg): interaction and drug p across the bands of the index (q_interaction,
q_drug, used in the figures); per-time p across times x bands of the index (q_t1..q_t3);
q_interaction_all across every index x band of the module.

Outputs in <comparison>/<module>/: <module>_aya_vs_placebo.xlsx, boxplots_per_index/<index>.png
(per band: Ayahuasca and placebo boxes at each time point, median trajectories), figures/summary_heatmap.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from psychedelics import config_aya as cfg
from psychedelics import runner
from psychedelics.paired_stats import friedman_row, wilcoxon_pair
from descriptive_analysis.summary_stats import fdr_bh
from descriptive_analysis.plots import _save, paper_style, BASE_FONT

NOTE = ("Boxes: Ayahuasca (red) and placebo (blue) at each time point, lines = median trajectory. Title: interaction = "
        "Friedman on (Ayahuasca − placebo) across time, drug = Wilcoxon on the mean over time; q = FDR across the bands "
        "of the index. Stars over a time point: Wilcoxon Ayahuasca vs placebo, FDR across times × bands "
        "(* q<0.05, ** q<0.01, *** q<0.001); grey p = uncorrected p<0.05 only.")


def _stars(q):
    return "***" if q < .001 else "**" if q < .01 else "*" if q < .05 else ""


def _p(p, name="p"):
    return f"{name} = n.a." if not np.isfinite(p) else (f"{name} < 0.001" if p < .001 else f"{name} = {p:.3f}")


def _matrices(d, index, aya, pla):
    w = (d.replace([np.inf, -np.inf], np.nan)
         .pivot_table(index="subject", columns="condition", values=index, aggfunc="mean")
         .reindex(columns=aya + pla).dropna())
    return w[aya].values, w[pla].values, w.index


def drug_time_tests(long, indices, aya, pla, bands=None):
    rows = []
    for m in indices:
        if m not in long:
            continue
        for b in ([None] if bands is None else bands):
            d = long if b is None else long[long.band == b]
            A, P, subs = _matrices(d, m, aya, pla)
            if len(subs) < 3:
                continue
            fi = friedman_row(pd.DataFrame(A - P))
            pd_, rd, _ = wilcoxon_pair(P.mean(1), A.mean(1))
            row = dict(index=m, band=b or "", n=len(subs), p_interaction=fi["p"], W_interaction=fi["kendall_w"],
                       p_drug=pd_, r_drug=rd, p_time_aya=friedman_row(pd.DataFrame(A))["p"],
                       p_time_pla=friedman_row(pd.DataFrame(P))["p"])
            for t in range(3):
                pt, rt, dt = wilcoxon_pair(P[:, t], A[:, t])
                row.update({f"p_t{t + 1}": pt, f"r_t{t + 1}": rt, f"median_aya_t{t + 1}": float(np.median(A[:, t])),
                            f"median_pla_t{t + 1}": float(np.median(P[:, t]))})
            rows.append(row)
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    for col in ("q_interaction", "q_drug", "q_t1", "q_t2", "q_t3"):
        t[col] = np.nan
    for m, g in t.groupby("index", sort=False):
        for p, q in (("p_interaction", "q_interaction"), ("p_drug", "q_drug")):
            ok = g[p].notna()
            if ok.any():
                t.loc[g.index[ok], q] = fdr_bh(g.loc[ok, p].values)
        v = g[["p_t1", "p_t2", "p_t3"]].values.ravel()
        q = np.full(v.shape, np.nan)
        ok = np.isfinite(v)
        if ok.any():
            q[ok] = fdr_bh(v[ok])
        t.loc[g.index, ["q_t1", "q_t2", "q_t3"]] = q.reshape(len(g), 3)
    ok = t.p_interaction.notna()
    t["q_interaction_all"] = np.nan
    t.loc[ok, "q_interaction_all"] = fdr_bh(t.loc[ok, "p_interaction"].values)
    first = ["index", "band", "n", "W_interaction", "p_interaction", "q_interaction", "r_drug", "p_drug", "q_drug"]
    return t[first + [c for c in t.columns if c not in first]]


def _lookup(tests, m, b):
    t = tests[(tests["index"] == m) & (tests["band"] == (b or ""))]
    return t.iloc[0].to_dict() if len(t) else None


def drug_box(ax, d, m, aya, pla, times, test, ylabel="", seed=0):
    rng = np.random.default_rng(seed)
    A, P, subs = _matrices(d, m, aya, pla)
    x = np.arange(3)
    for arr, off, key in ((A, -0.2, "aya"), (P, 0.2, "pla")):
        col = cfg.DRUG_COLORS[key]
        if not len(arr):
            continue
        bp = ax.boxplot(list(arr.T), positions=x + off, widths=0.32, patch_artist=True, showfliers=False,
                        medianprops=dict(color="k", lw=1.6))
        for patch in bp["boxes"]:
            patch.set_facecolor(col)
            patch.set_alpha(0.4)
        for t in range(3):
            ax.scatter(np.full(len(arr), t + off) + rng.uniform(-.06, .06, len(arr)), arr[:, t], s=10, color=col,
                       edgecolor="k", linewidth=0.3, zorder=3)
        ax.plot(x + off, np.median(arr, 0), color=col, lw=2, marker="o", ms=4, zorder=4,
                label="Ayahuasca" if key == "aya" else "placebo")
    ax.set_xticks(x)
    ax.set_xticklabels(times, fontsize=BASE_FONT - 4)
    ax.set_ylabel(ylabel)
    if test is None or not len(A):
        return
    ax.set_title(f"interaction {_p(test['p_interaction'])}, {_p(test['q_interaction'], 'q')}\n"
                 f"drug {_p(test['p_drug'])}, {_p(test['q_drug'], 'q')}  n = {int(test['n'])}", fontsize=BASE_FONT - 5)
    hi = np.nanmax(np.r_[A.ravel(), P.ravel()])
    lo = np.nanmin(np.r_[A.ravel(), P.ravel()])
    span = (hi - lo) or 1
    for t in range(3):
        p, q = test[f"p_t{t + 1}"], test[f"q_t{t + 1}"]
        if np.isfinite(p) and p < .05:
            s, c = (_stars(q), "k") if q < .05 else (_p(p).replace(" ", ""), "0.5")
            ax.text(t, hi + span * .04, s, ha="center", va="bottom", color=c,
                    fontsize=BASE_FONT - 2 if c == "k" else BASE_FONT - 6)
    ax.set_ylim(top=hi + span * .16)


def plot_index(long, m, label, bands, aya, pla, times, tests, out_png, title=""):
    present = [b for b in (bands or [None]) if (long if b is None else long[long.band == b])[m]
               .replace([np.inf, -np.inf], np.nan).notna().any()] if m in long else []
    if not present:
        return
    fig, axes = plt.subplots(1, len(present), figsize=(3.9 * len(present) + 0.6, 5.0), squeeze=False)
    for ax, b in zip(axes[0], present):
        d = long if b is None else long[long.band == b]
        drug_box(ax, d, m, aya, pla, times, _lookup(tests, m, b), label if ax is axes[0, 0] else "")
        if b:
            ax.set_title(f"{b}\n" + ax.get_title(), fontsize=BASE_FONT - 3)
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle(f"{title} — {label}")
    fig.text(0.5, -0.02, NOTE, ha="center", va="top", fontsize=BASE_FONT - 6, wrap=True)
    _save(fig, out_png)


def plot_heatmap(tests, labels, bands, times, out_png, title=""):
    if tests.empty:
        return
    idx = list(dict.fromkeys(tests["index"]))
    bnd = [b for b in (bands or []) if b in set(tests.band)] or [""]

    def grid(col):
        return np.array([[tests.loc[(tests["index"] == m) & (tests.band == b), col].astype(float).mean()
                          if ((tests["index"] == m) & (tests.band == b)).any() else np.nan for b in bnd] for m in idx])

    panels = [("W_interaction", "q_interaction", "p_interaction", "interaction (drug × time)\nKendall's W", "viridis", (0, 1)),
              ("r_drug", "q_drug", "p_drug", "drug main effect\nr (red: Ayahuasca higher)", "RdBu_r", (-1, 1))]
    panels += [(f"r_t{t + 1}", f"q_t{t + 1}", f"p_t{t + 1}", f"{times[t]}: Ayahuasca vs placebo\nr", "RdBu_r", (-1, 1))
               for t in range(3)]
    fig, axes = plt.subplots(1, len(panels), figsize=(1.0 * len(bnd) * len(panels) + 4.5, 0.38 * len(idx) + 2.4),
                             squeeze=False)
    for k, (ax, (v, qc, pc, ttl, cmap, lim)) in enumerate(zip(axes[0], panels)):
        V, Q, P = grid(v), grid(qc), grid(pc)
        im = ax.imshow(V, cmap=cmap, vmin=lim[0], vmax=lim[1], aspect="auto")
        for i in range(len(idx)):
            for j in range(len(bnd)):
                s = _stars(Q[i, j]) if np.isfinite(Q[i, j]) else ""
                if not s and np.isfinite(P[i, j]) and P[i, j] < .05:
                    s = "+"
                if s:
                    ax.text(j, i, s, ha="center", va="center", fontsize=9, fontweight="bold",
                            color="w" if k == 0 and V[i, j] < .6 else "k")
        ax.set_xticks(range(len(bnd)))
        ax.set_xticklabels(bnd, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([labels.get(m, m) for m in idx] if k == 0 else [], fontsize=9)
        ax.set_title(ttl, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.6)
    fig.suptitle(title, fontsize=BASE_FONT)
    fig.text(0.5, -0.01, "Stars: FDR q (interaction / drug: across the bands of the index; time points: across "
             "times × bands): * <0.05 ** <0.01 *** <0.001; + = uncorrected p < 0.05 only.", ha="center", fontsize=8)
    _save(fig, out_png)


def run(tables, comp, module, xlsx_name, subdir=None):
    """
    tables: list of (long table, {index: label}, bands or None) from the module's index_tables.
    Writes the comparison folder of one module (in its subfolder subdir, if given).
    """
    paper_style()
    aya, pla, times = cfg.COMPARISONS[comp]
    out = runner.analysis_dir(comp, module)
    if subdir:
        out = out / subdir
        out.mkdir(exist_ok=True)
    box_dir, fig_dir = out / "boxplots_per_index", out / "figures"
    box_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)
    title = cfg.COMPARISON_TITLES[comp]
    all_tests, labels = [], {}
    for long, idx_labels, bands in tables:
        t = drug_time_tests(long, list(idx_labels), aya, pla, bands)
        all_tests.append(t)
        labels.update(idx_labels)
        for m, lab in idx_labels.items():
            plot_index(long, m, lab, bands, aya, pla, times, t, box_dir / f"{m}.png", title)
        plot_heatmap(t, idx_labels, bands, times, fig_dir / f"summary_heatmap{'' if bands else '_scalar'}.png",
                     f"{title} — {module}")
    tests = pd.concat([t for t in all_tests if len(t)], ignore_index=True)
    if len(tests):
        ok = tests.p_interaction.notna()
        tests.loc[ok, "q_interaction_all"] = fdr_bh(tests.loc[ok, "p_interaction"].values)
    runner.save_tables_excel(dict(tests=tests, time_points=pd.DataFrame(dict(time=times, ayahuasca=aya, placebo=pla))),
                             out / xlsx_name)
    sig = tests[(tests.q_interaction < cfg.ALPHA) | (tests.q_drug < cfg.ALPHA)] if len(tests) else tests
    print(f"   {comp} {module}: interaction or drug q < {cfg.ALPHA}: "
          + (", ".join(f"{r['index']}{'_' + r['band'] if r['band'] else ''}" for _, r in sig.iterrows()) or "none"))
    return tests
