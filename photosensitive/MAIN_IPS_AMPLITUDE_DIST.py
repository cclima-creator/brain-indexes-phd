# -*- coding: utf-8 -*-
"""
Distribuzione delle ampiezze di TUTTI i campioni (senza soglia e senza valanghe), database IPS:
quale distribuzione descrive |z| e quanta parte dei dati segue una power law. Stessa tabella della
fig. A15 delle valanghe, con una sola variabile: l'ampiezza.

Per soggetto e banda:
  1. filtro passa-banda e z-score per canale su tutte le epoche (come MAIN_IPS_AVALAN)
  2. |z| di tutti i canali e campioni; se ne prendono N_SAMPLES a caso (seed fisso). Con milioni di
     campioni ogni test rifiuterebbe ogni modello per minime deviazioni; campioni presi a caso sono
     anche meno correlati di campioni consecutivi.
  3. fit di Clauset per dati continui con xmin scelto dal minimo di KS (per le ampiezze xmin = minimo
     dei dati, ~0, non ha senso: una power law non parte da 0); p-value bootstrap; range in cui la
     power law regge e % dei campioni in quel range (power_law.fit_power_law, discrete=False)
  4. miglior modello tra power law, troncata, lognormale, esponenziale, esponenziale stirata sopra
     lo stesso xmin (power_law.compare_all_distributions, discrete=False)
Risultati in criticality_results/amplitude_distribution: amplitude_distribution.xlsx e
figE1_amplitude_distribution_table.png. Cache per soggetto in subjects/.
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
from avalanche_criticality.binarization import zscore_channels
from avalanche_criticality.power_law import fit_power_law, compare_all_distributions, MODELS

warnings.filterwarnings("ignore")

OUT_NAME = "amplitude_distribution"
N_SAMPLES = 5000                   # campioni |z| per soggetto e banda (a caso, seed fisso)
SEED = 0
FIT_VERSION = 2                    # 2: power law troncata continua scritta qui (bug del pacchetto powerlaw)
OVERWRITE = False


def params():
    return dict(FIT_VERSION=FIT_VERSION, N_SAMPLES=N_SAMPLES, SEED=SEED, BANDS=av.BANDS, N_BOOT=av.N_BOOT, N_SIM=av.N_SIM,
                MIN_TAIL=av.MIN_TAIL, FIXED_LENGTH_S=base.FIXED_LENGTH_S, ANALYSIS_WINDOW=base.ANALYSIS_WINDOW)


def amplitude_sample(x, rng):
    """|z| di tutti i canali e campioni (z-score per canale su tutte le epoche), N_SAMPLES a caso."""
    z = np.abs(zscore_channels(x)).ravel()
    z = z[np.isfinite(z) & (z > 0)]
    return z[rng.choice(z.size, min(N_SAMPLES, z.size), replace=False)], z.size


def analyze_subject(fpath, group, channels):
    ep = load_epochs(fpath, base.FIXED_LENGTH_S, base.ANALYSIS_WINDOW)
    if channels is not None:
        ep = pick_channels(ep, channels, base.OCCIPITAL_CHANNELS)
    sfreq = ep.info["sfreq"]
    data = get_data_uv(ep)
    rows = []
    for band, rng_b in av.BANDS.items():
        t0 = time.time()
        a, n_total = amplitude_sample(band_filter(data, sfreq, rng_b), np.random.default_rng(SEED))
        f = fit_power_law(a, False, None, av.N_BOOT, av.N_SIM, av.MIN_TAIL, lr_tests=False)
        c = compare_all_distributions(a, f["xmin"], discrete=False)
        n = len(a)
        rows.append(dict(group=group, subject=fpath.stem, band=band, variable="amplitude",
                         n_samples_total=n_total, n_samples_used=n,
                         pct_in_pl_range=100 * f.get("n_range", 0) / n if n else np.nan,
                         range_xmin=f.get("range_xmin"), xmax_cutoff=f.get("xmax_cutoff"),
                         decades=f.get("decades"), xmin=f.get("xmin"), frac_tail=f.get("frac_tail"),
                         alpha=f.get("alpha"), alpha_range=f.get("alpha_range"), p_gof=f.get("p_gof"),
                         pl_plausible=bool(f.get("plausible")), best_model=c["best"],
                         confidence=c["confidence"], tpl_cutoff=c.get("tpl_cutoff"),
                         tpl_alpha=c.get("tpl_alpha"), tpl_lambda=c.get("tpl_lambda"),
                         **{k: v for k, v in c.items() if k.startswith(("R_", "p_"))}))
        r = rows[-1]
        print(f"    {band:>10}: xmin {r['xmin']:.2f} SD ({100 * r['frac_tail']:.0f}% in coda) | "
              f"PL p {r['p_gof']:.2f} | {r['pct_in_pl_range']:3.0f}% nel range PL | "
              f"{r['best_model']} ({r['confidence']})  ({time.time() - t0:.0f} s)")
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
    """Una riga per banda x gruppo (stessi campi di distribution_summary della fig. A15)."""
    out = []
    for (b, g, var), d in fit.groupby(["band", "group", "variable"], sort=False):
        pct = d.pct_in_pl_range
        row = dict(band=b, group=g, variable=var, n_subjects=len(d),
                   pct_in_pl_range_median=pct.median(), pct_in_pl_range_q25=pct.quantile(.25),
                   pct_in_pl_range_q75=pct.quantile(.75), decades_median=d.decades.median(),
                   xmin_median=d.xmin.median(), frac_tail_median=d.frac_tail.median(),
                   alpha_median=d.alpha.median(),
                   n_pl_plausible=int(d.pl_plausible.sum()), n_strong=int((d.confidence == "strong").sum()))
        for m in MODELS:
            row[f"n_best_{m}"] = int((d.best_model == m).sum())
        row["n_ambiguous"] = int((d.confidence == "ambiguous").sum())
        out.append(row)
    return pd.DataFrame(out)


def main():
    channel_set, channels, out_root = base.choose_channel_set()
    out_dir = check_output_dir(av.results_dir(channel_set, out_root) / OUT_NAME, ["amplitude_distribution.xlsx"],
                               (FALLBACK_DIR if channels is None else FALLBACK_DIR / base.OCCIPITAL_SUBDIR) / OUT_NAME)
    cache_dir = out_dir / "subjects"
    cache_dir.mkdir(exist_ok=True)
    print("Cartella di output:", out_dir)
    print(f"|z| di tutti i campioni, {N_SAMPLES} per soggetto e banda | xmin di Clauset | bootstrap {av.N_BOOT}")

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
    save_tables_excel({"distribution_summary": summ, "distribution_fit": fit},
                      out_dir / "amplitude_distribution.xlsx", ["distribution_summary", "distribution_fit"])
    colors = {g: c for g, c in av.COLORS.items() if g in set(fit.group)}
    plots.paper_style()
    plot_distribution_table(summ, list(av.BANDS), colors, out_dir / "figE1_amplitude_distribution_table.png",
                            variables=(("amplitude", f"Amplitude |z| of all samples ({N_SAMPLES} random samples "
                                                     "per subject and band; fits above the Clauset xmin)"),),
                            what="samples",
                            heading="How the amplitude distribution of ALL samples is fitted (no threshold, "
                                    "no avalanches): share in the power-law range and best model")
    print("\nFatto. amplitude_distribution.xlsx e figE1_amplitude_distribution_table.png in:", out_dir)


if __name__ == "__main__":
    main()
