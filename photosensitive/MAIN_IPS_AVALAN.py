# -*- coding: utf-8 -*-
"""
MAIN valanghe, database IPS: binarizzazione, neuronal avalanches, power law + DCC, branching ratio,
repertorio, Fano factor. Separato da MAIN_IPS.py: non rifa' l'analisi preliminare, legge gli stessi
file .set e (se c'e') il suo results.pkl solo per confrontare le soglie.
I grafici si fanno dopo con plot_ips_avalan.py (senza rifare l'analisi).

Per ogni soggetto i risultati sono salvati in OUT_DIR/avalanches/subjects/<file>.pkl: rilanciando,
i soggetti gia' analizzati con gli stessi parametri (config_ips_avalan.params) vengono solo riletti.
Per rifare tutto: OVERWRITE = True in config_ips_avalan.py.

Pipeline per soggetto e banda (moduli in avalanche_criticality/):
  filtro passa-banda -> z-score per canale -> eventi oltre ±soglia SD (soglia del gruppo)
  -> valanghe (IEI 8 ms) -> tau, alpha, 1/(sigma nu z), DCC, cutoff -> branching ratio (+ 4-18 ms)
  -> repertorio e diversita' dei pattern -> Fano factor
"""
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# cartella principale del repository (brain-indexes-phd/) nel path, come in MAIN_IPS.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mne

from photosensitive import config_ips as base
from photosensitive import config_ips_avalan as cfg
from photosensitive.io_eeg import discover_files, load_epochs, get_data_uv
from photosensitive.report import (check_output_dir, save_tables_excel, save_results, load_results,
                                   FALLBACK_DIR)
from photosensitive.MAIN_IPS import pick_channels
from descriptive_analysis.summary_stats import compare_groups
from descriptive_analysis.plots import amplitude_deviation_from_gaussian
from avalanche_criticality.binarization import (zscore_channels, zscore_histogram,
                                                avalanches_from_epochs, divergence_threshold)
from avalanche_criticality.branching_ratio import branching_ratio, branching_ratio_sweep
from avalanche_criticality.power_law import avalanche_criticality
from avalanche_criticality.repertoire import repertoire, shared_repertoire
from avalanche_criticality.fano_factor import fano_factor, fano_curve

warnings.filterwarnings("ignore")

OUTPUT_NAMES = ["avalanche_results.xlsx", "avalanche_results.pkl"]
SHEET_ORDER = ["subject_band_metrics", "group_tests", "group_summary", "threshold_divergence",
               "branching_ratio_bins", "power_law_cutoff_scan", "shared_repertoire", "parameters"]

# colonne della tabella per soggetto x banda che vengono confrontate tra gruppi
TEST_METRICS = ["n_avalanches", "avalanche_rate_hz", "mean_size", "mean_duration_bins",
                "tau", "alpha", "third", "dcc", "dcc_clauset", "dcc_range", "tau_range", "alpha_range",
                "size_xmax_cutoff", "size_decades", "duration_xmax_cutoff", "duration_decades",
                "size_p_gof", "duration_p_gof", "size_mse", "duration_mse",
                "sigma_bins", "sigma_geometric", "sigma_ancestors",
                "repertoire_rate", "repertoire_per_avalanche", "diversity", "ff", "ff_beta"]


# ================================================================== UN SOGGETTO
def band_filter(data, sfreq, band):
    """Filtro passa-banda FIR a fase zero (MNE) su ogni epoca e canale. data: (n_ep, n_ch, n_t)."""
    lo, hi = band
    hi = min(hi, sfreq / 2 - 1)
    return mne.filter.filter_data(data, sfreq, lo, hi, verbose="ERROR")


def _fit_cols(prefix, f):
    """Colonne riassuntive di un fit di power_law.fit_power_law."""
    keys = ["xmin", "p_gof", "plausible", "n_tail", "frac_tail", "D", "alpha_se", "mse", "chi2_red",
            "residual_trend_rho", "residual_trend_p", "clauset_xmin", "clauset_alpha", "clauset_D",
            "clauset_n_tail", "range_xmin", "xmax_cutoff", "decades", "alpha_range_se", "p_range",
            "n_range", "mse_range", "chi2_red_range", "residual_trend_rho_range",
            "tpl_alpha", "tpl_cutoff", "best_fit"]
    out = {f"{prefix}_{k}": f.get(k, np.nan) for k in keys}
    for k, v in f.items():
        if k.startswith(("R_vs_", "p_vs_")):
            out[f"{prefix}_{k}"] = v
    return out


def analyze_band(data, sfreq, group, sid, band_name, band):
    """Tutte le misure di valanga per un soggetto in una banda. Ritorna (riga, dettagli)."""
    x = band_filter(data, sfreq, band)
    z = zscore_channels(x)
    zc, zd = zscore_histogram(z)
    thr = cfg.threshold_for(group, band_name)
    iei_s = cfg.MAX_IEI_MS / 1000
    av = avalanches_from_epochs(x, sfreq, thr, iei_s, cfg.EVENT_MODE, cfg.THRESH_TYPE, z=z)
    if cfg.IEI_MODE == "mean_iei" and np.isfinite(av["mean_iei_s"]):
        iei_s = av["mean_iei_s"]                      # l'IEI medio non dipende dal bin: secondo passaggio
        av = avalanches_from_epochs(x, sfreq, thr, iei_s, cfg.EVENT_MODE, cfg.THRESH_TYPE, z=z)
    sizes, durs = av["sizes"], av["durations"]
    row = dict(group=group, subject=sid, band=band_name, threshold_sd=thr, time_bin_ms=1000 * iei_s,
               n_channels=av["n_channels"], duration_s=av["duration_s"],
               n_events=av["n_events"], n_avalanches=len(sizes),
               avalanche_rate_hz=len(sizes) / av["duration_s"],
               mean_iei_ms=1000 * av["mean_iei_s"],
               mean_size=float(sizes.mean()) if sizes.size else np.nan,
               max_size=int(sizes.max()) if sizes.size else 0,
               mean_duration_bins=float(durs.mean()) if durs.size else np.nan,
               max_duration_bins=int(durs.max()) if durs.size else 0)

    crit = avalanche_criticality(sizes, durs, cfg.XMIN, cfg.N_BOOT, cfg.N_SIM, cfg.MIN_TAIL,
                                 cfg.THIRD_METHOD)
    row.update({k: crit[k] for k in ("tau", "alpha", "third", "third_se", "third_r2",
                                      "predicted_third", "dcc", "scaling_diff", "tau_range",
                                      "alpha_range", "third_range", "predicted_third_range",
                                      "dcc_range", "tau_clauset", "alpha_clauset", "third_clauset",
                                      "dcc_clauset")})
    row.update(_fit_cols("size", crit["size_fit"]))
    row.update(_fit_cols("duration", crit["duration_fit"]))

    profiles = [a["profile"] for a in av["avalanches"]]
    row.update(branching_ratio(av["events"], sfreq, iei_s, profiles))
    sweep = branching_ratio_sweep(av["events"], sfreq, cfg.BR_BINS_MS)

    rep, uniq, counts = repertoire(av["patterns"], av["duration_s"], cfg.REPERTOIRE_MIN_CHANNELS,
                                   cfg.REPERTOIRE_MAX_PATTERNS)
    row.update({k: v for k, v in rep.items() if k != "n_avalanches"})

    ff, ff_mu, ff_var = fano_factor(av["events"], av["bin_len"])
    curve = fano_curve(av["events"], sfreq)
    row.update(ff=ff, ff_mean=ff_mu, ff_var=ff_var, ff_beta=curve["beta"], ff_beta_se=curve["beta_se"])

    detail = dict(sizes=sizes.astype(np.int32), durations=durs.astype(np.int32),
                  size_fit=crit["size_fit"], duration_fit=crit["duration_fit"],
                  third_fit=crit["third_fit"], third_fit_range=crit["third_fit_range"],
                  br_sweep=sweep, fano_curve=curve, zhist=dict(centers=zc, density=zd),
                  unique_patterns=uniq, pattern_counts=counts)
    return row, detail


def analyze_subject(fpath, group, channels):
    """Carica un .set e analizza tutte le bande. Ritorna dict(rows, details, ch_names, params)."""
    ep = load_epochs(fpath, base.FIXED_LENGTH_S, base.ANALYSIS_WINDOW)
    if channels is not None:
        ep = pick_channels(ep, channels, base.OCCIPITAL_CHANNELS)
    sfreq = ep.info["sfreq"]
    data = get_data_uv(ep)
    print(f"    {data.shape[0]} epoche x {data.shape[1]} canali x {data.shape[2] / sfreq:.1f} s, "
          f"sfreq {sfreq:g} Hz")
    rows, details = [], {}
    for band_name, band in cfg.BANDS.items():
        t0 = time.time()
        row, det = analyze_band(data, sfreq, group, fpath.stem, band_name, band)
        rows.append(row)
        details[band_name] = det
        print(f"    {band_name:>10}: {row['n_avalanches']:6d} valanghe | tau {row['tau']:.2f} "
              f"alpha {row['alpha']:.2f} DCC {row['dcc']:.3f} | sigma {row['sigma_bins']:.3f} | "
              f"repertorio {row['repertoire_size']} | FF {row['ff']:.2f}  ({time.time() - t0:.0f} s)")
    return dict(rows=rows, details=details, ch_names=list(ep.ch_names), params=cfg.params(),
                sfreq=sfreq)


def load_or_run(fpath, group, channels, cache_dir):
    """Rilegge la cache del soggetto se ha gli stessi parametri, altrimenti analizza e salva."""
    cache = cache_dir / f"{fpath.stem}.pkl"
    if cache.exists() and not cfg.OVERWRITE:
        res = load_results(cache)
        if res.get("params") == cfg.params():
            print("    (gia' analizzato: letto dalla cache)")
            return res
        print("    parametri cambiati: rianalizzo")
    res = analyze_subject(fpath, group, channels)
    with open(cache, "wb") as f:
        pickle.dump(res, f)
    return res


# ================================================================== SOGLIE
def preliminary_divergence(out_root):
    """
    Punto di divergenza dalla gaussiana dell'analisi preliminare (fig8 di plot_ips.py), letto dal
    results.pkl di MAIN_IPS.py senza rifare nulla. Ritorna lista di righe (vuota se manca il file).
    """
    path = Path(out_root) / "results.pkl"
    if not path.exists():
        return []
    store = load_results(path)["store"]
    rows = []
    for g in cfg.GROUPS:
        r = amplitude_deviation_from_gaussian(store, g)
        if r is None:
            continue
        thr, pos, neg = divergence_threshold(r["centers"], r["heavier"], cfg.DIVERGENCE_MIN_SD)
        rows.append(dict(source="preliminary results.pkl (raw signal)", group=g, band="raw",
                         divergence_sd=thr, divergence_pos_sd=pos, divergence_neg_sd=neg,
                         threshold_used_sd=np.nan, n_subjects=r["n"]))
    return rows


def band_divergence(subjects):
    """
    Stesso metodo della fig8 (gaussiana di fit sulla densita' media del gruppo, IC 95% bootstrap sui
    soggetti, tratti 'heavier') sugli z-score filtrati in ogni banda di questa analisi: dice se le
    soglie del gruppo valgono anche banda per banda. Ritorna (righe, dati per il grafico).
    """
    rows, curves = [], {}
    for band_name in cfg.BANDS:
        for g in cfg.GROUPS:
            store = [dict(group=g, amp=dict(hist_z=s["details"][band_name]["zhist"],
                                            stats=dict(excess_kurtosis=np.nan, ks_D_vs_gauss=np.nan)))
                     for s in subjects if s["group"] == g]
            r = amplitude_deviation_from_gaussian(store, g)
            if r is None:
                continue
            thr, pos, neg = divergence_threshold(r["centers"], r["heavier"], cfg.DIVERGENCE_MIN_SD)
            rows.append(dict(source="this analysis (band-passed signal)", group=g, band=band_name,
                             divergence_sd=thr, divergence_pos_sd=pos, divergence_neg_sd=neg,
                             threshold_used_sd=cfg.threshold_for(g, band_name), n_subjects=r["n"]))
            curves[(g, band_name)] = {k: r[k] for k in ("centers", "density", "lo", "hi", "gauss",
                                                        "heavier", "mu", "sigma")} | dict(divergence=thr)
    return rows, curves


# ================================================================== TABELLE DI GRUPPO
def group_tables(df):
    """Test tra gruppi banda per banda (Mann-Whitney, FDR dentro ogni banda) e mediane per gruppo."""
    tests, summ = [], []
    metrics = [m for m in TEST_METRICS if m in df.columns]
    for b in cfg.BANDS:
        d = df[df.band == b]
        t = compare_groups(d, metrics)
        if len(t):
            t.insert(0, "band", b)
            tests.append(t)
        s = d.groupby("group")[metrics].agg(["median", "mean", "std", "count"])
        s.columns = [f"{m}_{a}" for m, a in s.columns]
        summ.append(s.reset_index().assign(band=b))
    tests = pd.concat(tests, ignore_index=True) if tests else pd.DataFrame()
    return tests, pd.concat(summ, ignore_index=True) if summ else pd.DataFrame()


def long_tables(subjects):
    """Branching ratio per time bin e scansione del cutoff, una riga per soggetto x banda x valore."""
    br, scan = [], []
    for s in subjects:
        for b, det in s["details"].items():
            for r in det["br_sweep"]:
                br.append(dict(group=s["group"], subject=s["subject"], band=b, **r))
            for var in ("size", "duration"):
                rng = det[f"{var}_fit"]
                scan.append(dict(group=s["group"], subject=s["subject"], band=b, variable=var,
                                 range_xmin=rng.get("range_xmin"), xmax_cutoff=rng.get("xmax_cutoff"),
                                 decades=rng.get("decades"), alpha_range=rng.get("alpha_range"),
                                 p_range=rng.get("p_range"), clauset_xmin=rng.get("xmin"),
                                 clauset_alpha=rng.get("alpha"), clauset_p=rng.get("p_gof"),
                                 tpl_cutoff=rng.get("tpl_cutoff")))
    return pd.DataFrame(br), pd.DataFrame(scan)


def shared_patterns(subjects):
    """Pattern condivisi tra i due gruppi in ogni banda (Sorrentino 2021), se i canali coincidono."""
    if len({tuple(s["ch_names"]) for s in subjects}) != 1:
        print("I soggetti non hanno tutti gli stessi canali: salto il confronto dei pattern tra gruppi")
        return pd.DataFrame()
    groups = list(cfg.GROUPS)
    rows = []
    for b in cfg.BANDS:
        reps = {g: np.vstack([s["details"][b]["unique_patterns"] for s in subjects
                              if s["group"] == g and len(s["details"][b]["unique_patterns"])]
                             or [np.zeros((0, len(subjects[0]["ch_names"])), bool)])
                for g in groups}
        if len(groups) == 2:
            r = shared_repertoire(reps[groups[0]], reps[groups[1]])
            rows.append(dict(band=b, group_a=groups[0], group_b=groups[1], **r))
    return pd.DataFrame(rows)


# ================================================================== MAIN
def main():
    channel_set, channels, out_root = base.choose_channel_set()
    print(f"Canali: {'tutti i canali EEG' if channels is None else base.OCCIPITAL_CHANNELS}")
    fallback = FALLBACK_DIR if channels is None else FALLBACK_DIR / base.OCCIPITAL_SUBDIR
    out_dir = check_output_dir(Path(out_root) / cfg.OUT_SUBDIR, OUTPUT_NAMES, fallback / cfg.OUT_SUBDIR)
    cache_dir = out_dir / "subjects"
    cache_dir.mkdir(exist_ok=True)
    print("Cartella di output:", out_dir)
    print(f"Soglie (SD): {cfg.THRESHOLD_SD} | IEI {cfg.MAX_IEI_MS} ms | eventi '{cfg.EVENT_MODE}' | "
          f"bootstrap {cfg.N_BOOT}")

    subjects, t_start = [], time.time()
    for group, folder in cfg.GROUPS.items():
        files = discover_files(folder)
        print(f"{group}: {len(files)} file")
        for fpath in files:
            print("  ", fpath.name)
            try:
                res = load_or_run(fpath, group, channels, cache_dir)
                subjects.append(dict(group=group, subject=fpath.stem, **res))
            except Exception as e:
                print("    ERRORE ->", e)
    if not subjects:
        raise SystemExit("Nessun file analizzato: controlla i percorsi in config_ips.py")
    print(f"\nAnalisi: {(time.time() - t_start) / 60:.1f} min")

    df = pd.DataFrame([r for s in subjects for r in s["rows"]])
    tests, summary = group_tables(df)
    br_long, scan = long_tables(subjects)
    div_rows, div_curves = band_divergence(subjects)
    prelim = preliminary_divergence(out_root)
    divergence = pd.DataFrame(prelim + div_rows)
    shared = shared_patterns(subjects)

    print("\n=== Punto di divergenza dalla gaussiana (SD) ===")
    if len(divergence):
        print(divergence[["source", "group", "band", "divergence_sd", "threshold_used_sd"]]
              .round(2).to_string(index=False))
    if len(tests):
        print("\n=== Test tra gruppi (Mann-Whitney, p_fdr dentro ogni banda), solo p < 0.05 ===")
        sig = tests[tests.p < 0.05]
        print(sig.round(4).to_string(index=False) if len(sig) else "nessuno")

    params = pd.DataFrame([dict(parameter=k, value=str(v)) for k, v in cfg.params().items()])
    tables = {"subject_band_metrics": df, "group_tests": tests, "group_summary": summary,
              "threshold_divergence": divergence, "branching_ratio_bins": br_long,
              "power_law_cutoff_scan": scan, "shared_repertoire": shared, "parameters": params}
    save_tables_excel(tables, out_dir / "avalanche_results.xlsx", SHEET_ORDER)
    # per i grafici: tutto tranne i pattern (restano nelle cache per soggetto)
    details = {(s["group"], s["subject"], b): {k: v for k, v in det.items()
                                               if k not in ("unique_patterns", "pattern_counts")}
               for s in subjects for b, det in s["details"].items()}
    save_results(dict(metrics=df, tests=tests, summary=summary, details=details, br=br_long,
                      scan=scan, divergence=divergence, divergence_curves=div_curves,
                      shared=shared, params=cfg.params(), channel_set=channel_set,
                      bands=dict(cfg.BANDS)),
                 out_dir / "avalanche_results.pkl")
    print("\nFatto. Excel e avalanche_results.pkl in:", out_dir)


if __name__ == "__main__":
    main()
