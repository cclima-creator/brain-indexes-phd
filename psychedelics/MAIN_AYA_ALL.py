# -*- coding: utf-8 -*-
"""
Runs the four Ayahuasca analyses in order and writes the overview of the results folder:

  0 statistical analysis  MAIN_AYA_STATS.py
  1 avalanches            MAIN_AYA_AVALAN.py
  2 synchrony             MAIN_AYA_SYNC.py
  3 edge of chaos         MAIN_AYA_CHAOS.py   (the slow one: ~2-3 h with 7 processes)

Every module caches each subject x condition, so a re-run only redoes figures and tables.
At the end: <analysis>/significant_results.xlsx (every index x band with Friedman p < 0.05 in any
module, with FDR q and the post hoc pairs) and README.txt in the results folder.

    python psychedelics/MAIN_AYA_ALL.py            (all modules)
    python psychedelics/MAIN_AYA_ALL.py stats sync (only some)
"""
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from psychedelics import config_aya as cfg
from psychedelics import io_aya, runner

MODULES = {"stats": ("MAIN_AYA_STATS", "stats_results.xlsx"),
           "avalanches": ("MAIN_AYA_AVALAN", "avalanche_results.xlsx"),
           "synchrony": ("MAIN_AYA_SYNC", "new_method/synchrony_results.xlsx"),
           "chaos": ("MAIN_AYA_CHAOS", "chaos_results.xlsx")}

README = """AYAHUASCA vs PLACEBO - criticality indices (code: brain-indexes-phd/psychedelics)

Every participant did the ceremony twice, with Ayahuasca and with placebo.

ANALYSES WITH THREE CONDITIONS (repeated measures, the same subjects in the three conditions)
{analyses}
  Subjects without one of the conditions are left out of that analysis.
  "during" = minutes 27-33 of the hour (min 7-13 of the 20-40 slot).
  rest (analysis 7) = the 6-min pre-ceremony rest, taken at random 50/50 from the Ayahuasca or the
  placebo session for every subject who has both (seed {seed}); see rest_assignment.xlsx.

DRUG x TIME COMPARISONS (Ayahuasca and placebo trajectories, subjects with both ceremonies)
{comparisons}

FOLDERS (inside each analysis)
  0_statistical_analysis   power spectrum, band power, amplitude distribution, variance, ACF width,
                           stationarity, aperiodic 1/f, alpha peak, SNR
  1_avalanches             neuronal avalanches: rate, size, duration, branching ratio, repertoire,
                           Fano factor (power-law exponents, goodness of fit and DCC left out)
  2_synchrony              Kuramoto GS / metastability, PC, PLV / PPC, PLI, wPLI, dwPLI, iCOH
  3_edge_of_chaos          0-1 test K, lambda_max, LZC, Katz FD, Renyi entropy, MSE, entropy production,
                           DFA, fE/I, aperiodic exponent
  each with
    figures/                 explanatory figures (three-condition analyses) or summary heatmaps
                             (comparisons)
    boxplots_per_index/      one image per index, one panel per band (broadband first)
    *.xlsx                   tests, summary, subject values
  significant_results.xlsx   every index x band with p < 0.05 (Friedman, or interaction / drug), all modules

STATISTICS
  Three conditions: Friedman test (Kendall's W), post hoc Wilcoxon signed-rank on each pair
  (rank-biserial r > 0: second condition of the pair higher).
  Drug x time: interaction = Friedman on (Ayahuasca - placebo) across the 3 time points ("does the
  drug effect change over time?"); drug = Wilcoxon on the mean over time; at each time point
  Wilcoxon Ayahuasca vs placebo (r > 0: Ayahuasca higher).
  Benjamini-Hochberg FDR across the bands of the same index (the "q" of the figures); post hoc /
  time points across pairs (times) x bands of the index; *_all columns across every index x band.
  In the figures: * q<0.05 ** q<0.01 *** q<0.001; a grey p is an uncorrected p<0.05 only.

CAVEATS
  - 20-min slots ~110-240 epochs, pre/post/during 27-33 ~40-72 epochs (5 s each, bad epochs removed):
    counts depend on the duration; rates are the comparable measures. "n_epochs" is tested in
    0_statistical_analysis.
  - 16 channels at sensor level: GS, M, PC, PLV are inflated by volume conduction; PLI / wPLI /
    dwPLI / iCOH are lag-insensitive.
  - Avalanche threshold 1.9 SD for every condition; time bin 4 ms (delta 8 ms) at 250 Hz.
  - Edge of chaos: 0-1 test with 100 random c, MSE scales 1-10 on at most 60 epochs.

cache/ holds one result per subject x condition: delete a module folder there to recompute it.
"""


def significant(name, comparison=False):
    rows = []
    for module, (_, xlsx) in MODULES.items():
        if comparison:
            xlsx = xlsx.replace("_results.xlsx", "_aya_vs_placebo.xlsx")
        p = cfg.OUT_DIR / name / cfg.MODULE_DIRS[module] / xlsx
        if not p.exists():
            continue
        t = pd.read_excel(p, sheet_name="tests")
        keep = (t.p_interaction < cfg.ALPHA) | (t.p_drug < cfg.ALPHA) if comparison else t.p < cfg.ALPHA
        t = t[keep].copy()
        t.insert(0, "module", cfg.MODULE_DIRS[module])
        rows.append(t)
    if not rows:
        return
    t = pd.concat(rows, ignore_index=True)
    qcol, pcol = ("q_interaction", "p_interaction") if comparison else ("p_fdr_bands", "p")
    t = t.sort_values(["module", qcol, pcol])
    runner.save_tables_excel({"p_below_0.05": t}, cfg.OUT_DIR / name / "significant_results.xlsx")
    n_q = int((t[qcol] < cfg.ALPHA).sum()) + (int((t.q_drug < cfg.ALPHA).sum()) if comparison else 0)
    print(f"{name}: {len(t)} index x band with p < {cfg.ALPHA}, {n_q} FDR-significant "
          + ("(interaction + drug)" if comparison else "(Friedman)"))


def write_readme():
    ana = "\n".join(f"  {a:<30} {' vs '.join(c)}   ({len(io_aya.analysis_subjects(a))} subjects)"
                    for a, c in cfg.ANALYSES.items())
    com = "\n".join(f"  {c:<38} {cfg.COMPARISON_TITLES[c]}   ({len(io_aya.comparison_subjects(c))} subjects)"
                    for c in cfg.COMPARISONS)
    (cfg.OUT_DIR / "README.txt").write_text(README.format(analyses=ana, comparisons=com, seed=cfg.REST_SEED),
                                            encoding="utf-8")
    assign = io_aya.rest_assignment()
    runner.save_tables_excel({"rest_pre": pd.DataFrame(dict(subject=list(assign), rest_taken_from=[
        "Ayahuasca pre" if c == "pre" else "placebo pre" for c in assign.values()]))},
        cfg.OUT_DIR / "7_rest_vs_aya_vs_placebo" / "rest_assignment.xlsx")


def main(which):
    import importlib
    t0 = time.time()
    (cfg.OUT_DIR / "7_rest_vs_aya_vs_placebo").mkdir(parents=True, exist_ok=True)
    write_readme()
    for module in which:
        print(f"\n################ {module}", flush=True)
        importlib.import_module(f"psychedelics.{MODULES[module][0]}").main()
        print(f"################ {module} done ({(time.time() - t0) / 60:.0f} min total)", flush=True)
    for analysis in cfg.ANALYSES:
        significant(analysis)
    for comp in cfg.COMPARISONS:
        significant(comp, comparison=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(MODULES))
