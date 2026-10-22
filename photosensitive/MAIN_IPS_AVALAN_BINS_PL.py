# -*- coding: utf-8 -*-
"""
Confronto dei fit delle distribuzioni delle valanghe con time bin FISSI (4, 8, 20 ms), database IPS.
Come la tabella figA15 di plot_ips_avalan.py (che usa i bin caratteristici di ogni banda), ma una
tabella per ogni bin fisso. Stessa binarizzazione, soglie e metodi di fit di MAIN_IPS_AVALAN.py.

Per soggetto, banda, bin e variabile (dimensione S, durata T):
  - range in cui la power law regge e % di valanghe nel range (power_law.fit_power_law)
  - plausibilita' della power law: p di Clauset >= 0.1
  - miglior modello tra power law, troncata, lognormale, esponenziale, esponenziale stirata e
    confidenza (power_law.compare_all_distributions)
Risultati in criticality_results/bin_comparison: bin_powerlaw.xlsx e figB7_distribution_table_<bin>ms.png.
Cache per soggetto in bin_comparison/subjects_pl/.
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
from photosensitive.plot_ips_avalan import plot_distribution_table
from descriptive_analysis import plots
from avalanche_criticality.binarization import zscore_channels, avalanches_from_epochs
from avalanche_criticality.power_law import fit_power_law, compare_all_distributions, MODELS

warnings.filterwarnings("ignore")

BINS_MS = (4, 8, 20)
OUT_NAME = "bin_comparison"
OVERWRITE = False


def params():
    return dict(BINS_MS=BINS_MS, THRESHOLD_SD=av.THRESHOLD_SD, EVENT_MODE=av.EVENT_MODE,
                THRESH_TYPE=av.THRESH_TYPE, BANDS=av.BANDS, XMIN=av.XMIN, N_BOOT=av.N_BOOT, N_SIM=av.N_SIM,
                MIN_TAIL=av.MIN_TAIL, FIXED_LENGTH_S=base.FIXED_LENGTH_S, ANALYSIS_WINDOW=base.ANALYSIS_WINDOW)


def fit_row(x, group, sid, band, ms, var):
    """Range power law, plausibilita' e miglior modello di una distribuzione (stessi campi di figA15)."""
    f = fit_power_law(x, True, av.XMIN, av.N_BOOT, av.N_SIM, av.MIN_TAIL, lr_tests=False)
    c = compare_all_distributions(x, av.XMIN)
    n = len(x)
    return dict(group=group, subject=sid, band=band, bin_ms=ms, variable=var, n_avalanches=n,
                pct_in_pl_range=100 * f.get("n_range", 0) / n if n else np.nan,
                range_xmin=f.get("range_xmin"), xmax_cutoff=f.get("xmax_cutoff"), decades=f.get("decades"),
                alpha=f.get("alpha"), alpha_range=f.get("alpha_range"), p_gof=f.get("p_gof"),
                pl_plausible=bool(f.get("plausible")), best_model=c["best"], confidence=c["confidence"],
                tpl_cutoff=c.get("tpl_cutoff"), **{k: v for k, v in c.items() if k.startswith(("R_", "p_"))})


def analyze_subject(fpath, group, channels):
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
            t0 = time.time()
            a = avalanches_from_epochs(x, sfreq, thr, ms / 1000, av.EVENT_MODE, av.THRESH_TYPE, z=z)
            for var, vals in (("size", a["sizes"]), ("duration", a["durations"])):
                rows.append(fit_row(vals, group, fpath.stem, band, ms, var))
            s, d = rows[-2], rows[-1]
            print(f"    {band:>10} {ms:>2} ms: {s['n_avalanches']:6d} val. | S {s['pct_in_pl_range']:3.0f}% "
                  f"{s['best_model']:>21} | T {d['pct_in_pl_range']:3.0f}% {d['best_model']:>21} "
                  f"({time.time() - t0:.0f} s)")
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


def summary(fit):
    """Una riga per banda x gruppo x variabile x bin, stessi campi di distribution_summary (figA15)."""
    out = []
    for (ms, b, g, var), d in fit.groupby(["bin_ms", "band", "group", "variable"], sort=False):
        pct = d.pct_in_pl_range
        row = dict(bin_ms=ms, band=b, group=g, variable=var, n_subjects=len(d),
                   pct_in_pl_range_median=pct.median(), pct_in_pl_range_q25=pct.quantile(.25),
                   pct_in_pl_range_q75=pct.quantile(.75), decades_median=d.decades.median(),
                   n_pl_plausible=int(d.pl_plausible.sum()), n_strong=int((d.confidence == "strong").sum()))
        for m in MODELS:
            row[f"n_best_{m}"] = int((d.best_model == m).sum())
        row["n_ambiguous"] = int((d.confidence == "ambiguous").sum())
        out.append(row)
    return pd.DataFrame(out)


def main():
    channel_set, channels, out_root = base.choose_channel_set()
    out_dir = check_output_dir(av.results_dir(channel_set, out_root) / OUT_NAME, ["bin_powerlaw.xlsx"],
                               (FALLBACK_DIR if channels is None else FALLBACK_DIR / base.OCCIPITAL_SUBDIR) / OUT_NAME)
    cache_dir = out_dir / "subjects_pl"
    cache_dir.mkdir(exist_ok=True)
    print("Cartella di output:", out_dir)
    print(f"Bin fissi: {BINS_MS} ms | soglie (SD): {av.THRESHOLD_SD} | xmin {av.XMIN} | bootstrap {av.N_BOOT}")

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

    fit = pd.DataFrame(rows)
    summ = summary(fit)
    save_tables_excel({"distribution_summary": summ, "distribution_fit": fit}, out_dir / "bin_powerlaw.xlsx",
                      ["distribution_summary", "distribution_fit"])
    colors = {g: c for g, c in av.COLORS.items() if g in set(fit.group)}
    plots.paper_style()
    for ms in BINS_MS:
        plot_distribution_table(summ[summ.bin_ms == ms], list(av.BANDS), colors,
                                out_dir / f"figB7_distribution_table_{ms}ms.png",
                                title_extra=f"fixed time bin {ms} ms")
    print("\nFatto. bin_powerlaw.xlsx e figB7_distribution_table_*.png in:", out_dir)


if __name__ == "__main__":
    main()
