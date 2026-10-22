# -*- coding: utf-8 -*-
"""
Valanghe con time bin FISSI (4, 8, 20 ms) per tutte le bande, database IPS: quanto cambiano numero,
dimensione e durata delle valanghe, diversita' e repertorio con la scelta del bin.
Stessa binarizzazione e soglie di MAIN_IPS_AVALAN.py (config_ips_avalan.py); qui niente power law
(solo indici descrittivi), quindi e' veloce. Analisi e figure in un solo script.

Risultati in criticality_results/bin_comparison (cache per soggetto in subjects/):
  bin_comparison.xlsx  : una riga per soggetto x banda x bin, test tra gruppi (Mann-Whitney,
                         FDR dentro ogni banda e bin)
  figB1-figB6          : una figura per indice, righe = bin, colonne = bande, box per gruppo
"""
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from photosensitive import config_ips as base
from photosensitive import config_ips_avalan as av
from photosensitive.io_eeg import discover_files, load_epochs, get_data_uv
from photosensitive.report import check_output_dir, save_tables_excel, load_results, FALLBACK_DIR
from photosensitive.MAIN_IPS import pick_channels
from photosensitive.MAIN_IPS_AVALAN import band_filter
import matplotlib.pyplot as plt

from photosensitive.plot_ips_avalan import _save
from descriptive_analysis import plots
from descriptive_analysis.plots import _box_by_group, _stars
from descriptive_analysis.summary_stats import compare_groups
from avalanche_criticality.binarization import zscore_channels, avalanches_from_epochs
from avalanche_criticality.repertoire import repertoire

warnings.filterwarnings("ignore")

BINS_MS = (4, 8, 20)
OUT_NAME = "bin_comparison"
OVERWRITE = False
# (colonna, etichetta, nome del file): una figura per indice
INDICES = [("n_avalanches", "number of avalanches", "figB1_n_avalanches"),
           ("mean_size", "mean avalanche size\n(events)", "figB2_mean_size"),
           ("mean_duration_ms", "mean avalanche duration\n(ms)", "figB3_mean_duration"),
           ("diversity", "pattern diversity\n(median Hamming)", "figB4_diversity"),
           ("repertoire_rate", "repertoire\n(unique patterns / s)", "figB5_repertoire_rate"),
           ("repertoire_per_avalanche", "repertoire\n(unique / avalanches)", "figB6_repertoire_per_avalanche")]


def params():
    return dict(BINS_MS=BINS_MS, THRESHOLD_SD=av.THRESHOLD_SD, EVENT_MODE=av.EVENT_MODE,
                THRESH_TYPE=av.THRESH_TYPE, BANDS=av.BANDS, REPERTOIRE_MIN_CHANNELS=av.REPERTOIRE_MIN_CHANNELS,
                FIXED_LENGTH_S=base.FIXED_LENGTH_S, ANALYSIS_WINDOW=base.ANALYSIS_WINDOW)


def analyze_subject(fpath, group, channels):
    """Indici delle valanghe per ogni banda e ogni bin fisso. Ritorna la lista delle righe."""
    ep = load_epochs(fpath, base.FIXED_LENGTH_S, base.ANALYSIS_WINDOW)
    if channels is not None:
        ep = pick_channels(ep, channels, base.OCCIPITAL_CHANNELS)
    sfreq = ep.info["sfreq"]
    data = get_data_uv(ep)
    rows = []
    for band, rng in av.BANDS.items():
        x = band_filter(data, sfreq, rng)
        z = zscore_channels(x)
        thr = av.threshold_for(group, band)
        for ms in BINS_MS:
            a = avalanches_from_epochs(x, sfreq, thr, ms / 1000, av.EVENT_MODE, av.THRESH_TYPE, z=z)
            s, d = a["sizes"], a["durations"]
            bin_ms = 1000 * a["bin_len"] / sfreq
            rep, _, _ = repertoire(a["patterns"], a["duration_s"], av.REPERTOIRE_MIN_CHANNELS,
                                   av.REPERTOIRE_MAX_PATTERNS)
            rows.append(dict(group=group, subject=fpath.stem, band=band, bin_ms=ms,
                             bin_samples=a["bin_len"], threshold_sd=thr, n_avalanches=len(s),
                             avalanche_rate_hz=len(s) / a["duration_s"],
                             mean_size=float(s.mean()) if len(s) else np.nan,
                             mean_duration_bins=float(d.mean()) if len(d) else np.nan,
                             mean_duration_ms=float(d.mean()) * bin_ms if len(d) else np.nan,
                             **{k: v for k, v in rep.items() if k != "n_avalanches"}))
        print(f"    {band:>10}: " + " | ".join(f"{r['bin_ms']} ms {r['n_avalanches']} val."
                                                for r in rows[-len(BINS_MS):]))
    return rows


def load_or_run(fpath, group, channels, cache_dir):
    cache = cache_dir / f"{fpath.stem}.pkl"
    if cache.exists() and not OVERWRITE:
        res = load_results(cache)
        if res.get("params") == params():
            print("    (letto dalla cache)")
            return res["rows"]
    rows = analyze_subject(fpath, group, channels)
    with open(cache, "wb") as f:
        pickle.dump(dict(rows=rows, params=params()), f)
    return rows


def plot_index(wide, tests, col, lab, bands, colors, out_png):
    """
    Una figura per indice: righe = bin, colonne = bande, box per gruppo (plots._box_by_group).
    Titolo di ogni pannello: p di Mann-Whitney non corretto e p_FDR (Benjamini-Hochberg sugli indici
    testati nella stessa banda e nello stesso bin, foglio group_tests); asterischi dal p_FDR.
    """
    fig, axes = plt.subplots(len(BINS_MS), len(bands), figsize=(3.1 * len(bands), 3.5 * len(BINS_MS)),
                             squeeze=False)
    for i, ms in enumerate(BINS_MS):
        c = f"{col}_{ms}ms"
        for j, b in enumerate(bands):
            ax = axes[i, j]
            d = wide[wide.band == b]
            if c not in d or d[c].notna().sum() == 0:
                ax.axis("off")
                continue
            _box_by_group(ax, d, c, colors, f"{lab}\n{ms} ms bin" if j == 0 else "")
            t = tests[(tests.band == b) & (tests.bin_ms == ms) & (tests.metric == col)] if len(tests) else []
            if len(t):
                p, q = t["p"].iloc[0], t["p_fdr"].iloc[0]
                ax.set_title(f"{b}\np = {p:.3f}, p_FDR = {q:.3f} {_stars(q)}", fontsize=plots.BASE_FONT - 2)
            else:
                ax.set_title(b, fontsize=plots.BASE_FONT - 2)
    fig.suptitle(f"{lab.replace(chr(10), ' ')} with fixed time bins {', '.join(map(str, BINS_MS))} ms (rows) "
                 "per band (columns)\nMann-Whitney U; p_FDR = Benjamini-Hochberg over the indices tested "
                 "in the same band and bin; stars from p_FDR")
    _save(fig, out_png)


def main():
    channel_set, channels, out_root = base.choose_channel_set()
    out_dir = check_output_dir(av.results_dir(channel_set, out_root) / OUT_NAME, ["bin_comparison.xlsx"],
                               (FALLBACK_DIR if channels is None else FALLBACK_DIR / base.OCCIPITAL_SUBDIR) / OUT_NAME)
    cache_dir = out_dir / "subjects"
    cache_dir.mkdir(exist_ok=True)
    print("Cartella di output:", out_dir)
    print(f"Bin fissi: {BINS_MS} ms | soglie (SD): {av.THRESHOLD_SD}")

    rows, t0 = [], time.time()
    for group, folder in av.GROUPS.items():
        files = base.subject_files(group, discover_files(folder))
        print(f"{group}: {len(files)} file")
        for fpath in files:
            print("  ", fpath.name)
            try:
                rows += load_or_run(fpath, group, channels, cache_dir)
            except Exception as e:
                print("    ERRORE ->", e)
    if not rows:
        raise SystemExit("Nessun file analizzato: controlla i percorsi in config_ips.py")
    print(f"\nAnalisi: {(time.time() - t0) / 60:.1f} min")

    df = pd.DataFrame(rows)
    tests = []
    for (b, ms), d in df.groupby(["band", "bin_ms"], sort=False):
        t = compare_groups(d, [c for c, _, _ in INDICES] + ["avalanche_rate_hz", "mean_duration_bins"])
        if len(t):
            t.insert(0, "bin_ms", ms)
            t.insert(0, "band", b)
            tests.append(t)
    tests = pd.concat(tests, ignore_index=True) if tests else pd.DataFrame()
    save_tables_excel({"subject_band_bin": df, "group_tests": tests}, out_dir / "bin_comparison.xlsx",
                      ["subject_band_bin", "group_tests"])

    # figure: una per indice, righe = bin, colonne = bande (colonne larghe "indice_<bin>ms")
    wide = df.pivot_table(index=["group", "subject", "band"], columns="bin_ms",
                          values=[c for c, _, _ in INDICES]).reset_index()
    wide.columns = [f"{a}_{b}ms" if b != "" else a for a, b in wide.columns]
    colors = {g: c for g, c in av.COLORS.items() if g in set(df.group)}
    bands = list(av.BANDS)
    plots.paper_style()
    for col, lab, name in INDICES:
        plot_index(wide, tests, col, lab, bands, colors, out_dir / f"{name}.png")
    if len(tests):
        print("\n=== Test tra gruppi (Mann-Whitney, p_fdr dentro banda e bin), solo p < 0.05 ===")
        sig = tests[tests.p < 0.05]
        print(sig[["band", "bin_ms", "metric", "p", "p_fdr"]].round(4).to_string(index=False) if len(sig) else "nessuno")
    print("\nFatto. bin_comparison.xlsx e figure in:", out_dir)


if __name__ == "__main__":
    main()
