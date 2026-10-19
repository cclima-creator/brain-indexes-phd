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

IEI robusto (avalanche_criticality/iei.py), per ora solo diagnostica (non cambia il time bin):
  stessi eventi -> intervalli tra eventi consecutivi -> log10 -> GMM / valle / fallback
  -> soglia intra/inter-valanga -> IEI medio intra. Cache in subjects_iei/, risultati in
  iei_results.xlsx / iei_results.pkl (grafici figA12-figA14 di plot_ips_avalan.py).
  RUN_AVALANCHES / RUN_IEI in config_ips_avalan.py scelgono quali analisi fare.
"""
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

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
from descriptive_analysis.summary_stats import compare_groups, fdr_bh
from descriptive_analysis.plots import amplitude_deviation_from_gaussian
from avalanche_criticality.binarization import (zscore_channels, zscore_histogram, binarize,
                                                avalanches_from_epochs, divergence_threshold)
# from avalanche_criticality.iei import robust_iei    # analisi dell'IEI disattivata (config: RUN_IEI)
from avalanche_criticality.branching_ratio import branching_ratio, branching_ratio_sweep
from avalanche_criticality.power_law import avalanche_criticality, compare_all_distributions, MODELS
from avalanche_criticality.repertoire import repertoire, shared_repertoire
from avalanche_criticality.fano_factor import fano_factor, fano_curve

warnings.filterwarnings("ignore")

OUTPUT_NAMES = ["avalanche_results.xlsx", "avalanche_results.pkl"]   # (iei_results.*: analisi IEI disattivata)
SHEET_ORDER = ["subject_band_metrics", "group_tests", "group_summary", "distribution_summary",
               "distribution_fit", "threshold_divergence",
               "branching_ratio_bin_tests",
               "branching_ratio_bins", "power_law_cutoff_scan", "shared_repertoire", "parameters"]

# colonne della tabella per soggetto x banda che vengono confrontate tra gruppi
TEST_METRICS = ["n_avalanches",                    # (mean/median_iei_ms tolti: analisi IEI disattivata)
                 "avalanche_rate_hz", "mean_size", "mean_duration_bins",
                "tau", "alpha", "third", "dcc", "dcc_clauset", "dcc_range", "tau_range", "alpha_range",
                "size_xmax_cutoff", "size_decades", "duration_xmax_cutoff", "duration_decades",
                "size_p_gof", "duration_p_gof", "size_mse", "duration_mse",
                "sigma_bins", "sigma_geometric", "sigma_ancestors",
                "repertoire_rate", "repertoire_per_avalanche", "diversity_mean", "ff", "ff_beta"]

IEI_SHEET_ORDER = ["iei_subject_band", "iei_group_tests", "iei_group_summary", "iei_bimodality",
                   "parameters"]
IEI_TEST_METRICS = ["iei_mean_ms", "threshold_ms", "intra_median_ms", "inter_median_ms",
                    "gmm_intra_peak_ms", "gmm_inter_peak_ms", "gmm_intra_sd_dex", "gmm_inter_sd_dex",
                    "gmm_intra_weight", "ashman_d", "frac_intra", "frac_zero", "median_all_ms",
                    "mean_all_ms", "iei_mean_with_zero_ms"]


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
    if cfg.IEI_MODE == "per_band":
        iei_s = cfg.BIN_SAMPLES[band_name] / sfreq    # bin per banda, uguale per tutti i soggetti
    av = avalanches_from_epochs(x, sfreq, thr, iei_s, cfg.EVENT_MODE, cfg.THRESH_TYPE, z=z)
    if cfg.IEI_MODE == "mean_iei" and np.isfinite(av["mean_iei_s"]):
        iei_s = av["mean_iei_s"]                      # l'IEI medio non dipende dal bin: secondo passaggio
        av = avalanches_from_epochs(x, sfreq, thr, iei_s, cfg.EVENT_MODE, cfg.THRESH_TYPE, z=z)
    sizes, durs = av["sizes"], av["durations"]
    row = dict(group=group, subject=sid, band=band_name, threshold_sd=thr, time_bin_ms=1000 * iei_s,
               n_channels=av["n_channels"], duration_s=av["duration_s"],
               n_events=av["n_events"], n_avalanches=len(sizes),
               avalanche_rate_hz=len(sizes) / av["duration_s"],
               # mean_iei_ms / median_iei_ms: analisi dell'IEI disattivata
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


# ================================================================== IEI ROBUSTO (solo diagnostica)
def iei_band(x, sfreq, threshold):
    """Eventi come in avalanches_from_epochs (z-score su tutte le epoche, canali piatti esclusi) -> robust_iei."""
    z = zscore_channels(x)
    z = z[:, np.isfinite(z).all(axis=(0, 2))]
    events = [binarize(zk, threshold, cfg.EVENT_MODE, cfg.THRESH_TYPE).sum(0) for zk in z]
    from avalanche_criticality.iei import robust_iei    # solo se RUN_IEI viene riattivato
    return robust_iei(events, sfreq, cfg.IEI_METHOD, cfg.IEI_FALLBACK_FACTOR, cfg.IEI_MIN_ASHMAN_D,
                      cfg.IEI_DEQUANTIZE, smooth_bins=cfg.IEI_SMOOTH_BINS)


def _iei_row(r, group, sid, band, thr):
    """Riga della tabella per soggetto x banda (tempi in ms; SD delle gaussiane in decadi)."""
    g, intra, inter = r["gmm"], r["intra_s"], r["inter_s"]
    n, n_all = intra.size + inter.size, r["n_intervals"] + r["n_zero"]
    med = lambda v: 1000 * float(np.median(v)) if v.size else np.nan
    return dict(group=group, subject=sid, band=band, threshold_sd=thr, method=r["method"],
                iei_mean_ms=1000 * r["iei_mean_s"], threshold_ms=1000 * r["threshold_s"],
                threshold_gmm_ms=1000 * r["threshold_gmm_s"],
                threshold_valley_ms=1000 * r["threshold_valley_s"],
                threshold_fallback_ms=1000 * r["threshold_fallback_s"],
                intra_median_ms=med(intra), inter_median_ms=med(inter),
                inter_mean_ms=1000 * float(inter.mean()) if inter.size else np.nan,
                n_intervals=r["n_intervals"], n_intra=int(intra.size), n_inter=int(inter.size),
                frac_intra=intra.size / n if n else np.nan,
                frac_zero=r["n_zero"] / n_all if n_all else np.nan,
                iei_mean_with_zero_ms=1000 * r["iei_mean_with_zero_s"],
                median_all_ms=1000 * r["median_s"], mean_all_ms=1000 * r["mean_all_s"],
                gmm_intra_peak_ms=1000 * 10 ** g["means"][0], gmm_inter_peak_ms=1000 * 10 ** g["means"][1],
                gmm_intra_sd_dex=g["sds"][0], gmm_inter_sd_dex=g["sds"][1],
                gmm_intra_weight=g["weights"][0], ashman_d=g["ashman_d"], delta_bic=g["bic1"] - g["bic2"],
                bimodal=bool(g["ashman_d"] >= cfg.IEI_MIN_ASHMAN_D),
                valley_found=bool(np.isfinite(r["threshold_valley_s"])))


def analyze_subject_iei(fpath, group, channels):
    """IEI robusto di un soggetto in ogni banda. Ritorna dict(rows, details, params, sfreq)."""
    ep = load_epochs(fpath, base.FIXED_LENGTH_S, base.ANALYSIS_WINDOW)
    if channels is not None:
        ep = pick_channels(ep, channels, base.OCCIPITAL_CHANNELS)
    sfreq = ep.info["sfreq"]
    data = get_data_uv(ep)
    rows, details = [], {}
    for band_name, band in cfg.BANDS.items():
        thr = cfg.threshold_for(group, band_name)
        r = iei_band(band_filter(data, sfreq, band), sfreq, thr)
        row = _iei_row(r, group, fpath.stem, band_name, thr)
        rows.append(row)
        # per i grafici: istogramma log, curva smussata, GMM e soglie (non gli intervalli, troppo grandi)
        details[band_name] = dict(
            centers=r["valley"]["centers"], density=r["valley"]["density"], smooth=r["valley"]["smooth"],
            gmm={k: r["gmm"][k] for k in ("weights", "means", "sds", "ashman_d", "bic1", "bic2")},
            **{k: r[k] for k in ("method", "iei_mean_s", "threshold_s", "threshold_gmm_s",
                                 "threshold_valley_s", "threshold_fallback_s")})
        print(f"    IEI {band_name:>10}: {row['method']:>8} | soglia {row['threshold_ms']:7.1f} ms | "
              f"IEI intra {row['iei_mean_ms']:6.1f} ms | D {row['ashman_d']:.2f} | "
              f"intra {100 * row['frac_intra']:.0f}%")
    return dict(rows=rows, details=details, params=cfg.iei_params(), sfreq=sfreq)


def load_or_run_iei(fpath, group, channels, cache_dir):
    """Come load_or_run, per l'IEI (cache separata: non tocca quella delle valanghe)."""
    cache = cache_dir / f"{fpath.stem}.pkl"
    if cache.exists() and not cfg.OVERWRITE:
        res = load_results(cache)
        if res.get("params") == cfg.iei_params():
            print("    IEI: letto dalla cache")
            return res
    res = analyze_subject_iei(fpath, group, channels)
    with open(cache, "wb") as f:
        pickle.dump(res, f)
    return res


def iei_tables(df):
    """
    Test tra gruppi per banda (Mann-Whitney, FDR dentro ogni banda), statistiche per gruppo e
    bimodalita': soggetti con GMM separata (D >= IEI_MIN_ASHMAN_D), con valle, metodo usato;
    test esatto di Fisher sulla proporzione di soggetti bimodali tra i due gruppi.
    """
    tests, summ, bim = [], [], []
    metrics = [m for m in IEI_TEST_METRICS if m in df.columns]
    groups = [g for g in cfg.GROUPS if g in set(df.group)]
    for b in cfg.BANDS:
        d = df[df.band == b]
        t = compare_groups(d, metrics)
        if len(t):
            t.insert(0, "band", b)
            tests.append(t)
        s = d.groupby("group")[metrics].agg(["median", "mean", "std", "count"])
        s.columns = [f"{m}_{a}" for m, a in s.columns]
        summ.append(s.reset_index().assign(band=b))
        row = dict(band=b)
        for g in groups:
            dg = d[d.group == g]
            row[f"n_{g}"] = len(dg)
            row[f"n_bimodal_{g}"] = int(dg.bimodal.sum())
            row[f"n_valley_{g}"] = int(dg.valley_found.sum())
            for m in ("gmm", "valley", "fallback"):
                row[f"method_{m}_{g}"] = int((dg.method == m).sum())
        if len(groups) == 2:
            table = [[row[f"n_bimodal_{g}"], row[f"n_{g}"] - row[f"n_bimodal_{g}"]] for g in groups]
            row["fisher_p_bimodal"] = stats.fisher_exact(table).pvalue
        bim.append(row)
    tests = pd.concat(tests, ignore_index=True) if tests else pd.DataFrame()
    summ = pd.concat(summ, ignore_index=True) if summ else pd.DataFrame()
    return tests, summ, pd.DataFrame(bim)


def save_iei_results(iei_subjects, out_dir, channel_set):
    """Tabelle dell'IEI robusto: iei_results.xlsx e iei_results.pkl (per plot_ips_avalan.py)."""
    df = pd.DataFrame([r for s in iei_subjects for r in s["rows"]])
    tests, summary, bimodality = iei_tables(df)
    print(f"\n=== IEI robusto: soggetti bimodali (GMM, D >= {cfg.IEI_MIN_ASHMAN_D:g}) e metodo usato ===")
    print(bimodality.round(4).to_string(index=False))
    if len(tests):
        print("\n=== IEI: test tra gruppi (Mann-Whitney, p_fdr dentro ogni banda), solo p < 0.05 ===")
        sig = tests[tests.p < 0.05]
        print(sig.round(4).to_string(index=False) if len(sig) else "nessuno")
    params = pd.DataFrame([dict(parameter=k, value=str(v)) for k, v in cfg.iei_params().items()])
    save_tables_excel({"iei_subject_band": df, "iei_group_tests": tests, "iei_group_summary": summary,
                       "iei_bimodality": bimodality, "parameters": params},
                      out_dir / "iei_results.xlsx", IEI_SHEET_ORDER)
    details = {(s["group"], s["subject"], b): det for s in iei_subjects for b, det in s["details"].items()}
    save_results(dict(metrics=df, tests=tests, summary=summary, bimodality=bimodality, details=details,
                      params=cfg.iei_params(), channel_set=channel_set, bands=dict(cfg.BANDS)),
                 out_dir / "iei_results.pkl")
    print("IEI: iei_results.xlsx e iei_results.pkl in:", out_dir)


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


def branching_bin_tests(br):
    """
    sigma tra gruppi per ogni banda, stimatore e time bin: Mann-Whitney U.
    p_fdr = FDR (Benjamini-Hochberg) sui time bin di quella banda e quello stimatore.
    """
    groups = [g for g in cfg.GROUPS if g in set(br.group)]
    if len(groups) != 2:
        return pd.DataFrame()
    rows = []
    for b in cfg.BANDS:
        for est in ("sigma_bins", "sigma_geometric", "sigma_ancestors"):
            part = []
            for ms, d in br[br.band == b].groupby("bin_ms"):
                x = [d.loc[d.group == g, est].dropna().values for g in groups]
                if min(len(v) for v in x) < 2:
                    continue
                part.append(dict(band=b, estimator=est, bin_ms=ms, p=stats.mannwhitneyu(*x).pvalue,
                                 **{f"median_{g}": float(np.median(v)) for g, v in zip(groups, x)}))
            if part:
                for r, q in zip(part, fdr_bh(np.array([r["p"] for r in part]))):
                    r["p_fdr"] = q
                rows += part
    return pd.DataFrame(rows)


def distribution_tables(subjects, out_dir):
    """
    Per ogni soggetto, banda e variabile (S, T):
      - percentuale di valanghe nel range in cui la power law regge (tratto arancio della fig. A3:
        n_range / n; 0 se nessun range regge) e decadi del range
      - power law plausibile (p di Clauset >= 0.1)
      - miglior modello tra power law, troncata, lognormale, esponenziale, esponenziale stirata e
        confidenza (power_law.compare_all_distributions)
    I confronti sono salvati in distribution_cache.pkl (chiave: soggetto, banda, variabile, parametri):
    rilanciando si ricalcolano solo quelli nuovi.
    Ritorna (tabella per soggetto, riassunto per banda x gruppo x variabile).
    """
    cache_path = out_dir / "distribution_cache.pkl"
    cache = load_results(cache_path) if cache_path.exists() else {}
    key_par = repr(sorted(cfg.params().items()))
    rows = []
    for s in subjects:
        for b, det in s["details"].items():
            for var, xs in (("size", det["sizes"]), ("duration", det["durations"])):
                f = det[f"{var}_fit"]
                key = (s["subject"], b, var, key_par)
                if key not in cache:
                    cache[key] = compare_all_distributions(xs, cfg.XMIN)
                c = cache[key]
                n = len(xs)
                rows.append(dict(
                    group=s["group"], subject=s["subject"], band=b, variable=var, n_avalanches=n,
                    pct_in_pl_range=100 * f.get("n_range", 0) / n if n else np.nan,
                    range_xmin=f.get("range_xmin"), xmax_cutoff=f.get("xmax_cutoff"),
                    decades=f.get("decades"), p_gof=f.get("p_gof"), pl_plausible=bool(f.get("plausible")),
                    best_model=c["best"], confidence=c["confidence"], tpl_cutoff=c.get("tpl_cutoff"),
                    **{k: v for k, v in c.items() if k.startswith(("R_", "p_"))}))
    save_results(cache, cache_path)
    fit = pd.DataFrame(rows)
    summ = []
    for (b, g, var), d in fit.groupby(["band", "group", "variable"], sort=False):
        pct = d.pct_in_pl_range
        row = dict(band=b, group=g, variable=var, n_subjects=len(d),
                   pct_in_pl_range_median=pct.median(), pct_in_pl_range_q25=pct.quantile(.25),
                   pct_in_pl_range_q75=pct.quantile(.75), decades_median=d.decades.median(),
                   n_pl_plausible=int(d.pl_plausible.sum()), n_strong=int((d.confidence == "strong").sum()))
        for m in MODELS:
            row[f"n_best_{m}"] = int((d.best_model == m).sum())
        row["n_ambiguous"] = int((d.confidence == "ambiguous").sum())
        summ.append(row)
    return fit, pd.DataFrame(summ)


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
    out_dir = check_output_dir(cfg.results_dir(channel_set, out_root), OUTPUT_NAMES, fallback / cfg.OUT_SUBDIR)
    cache_dir = out_dir / "subjects"
    cache_dir.mkdir(exist_ok=True)
    iei_cache_dir = out_dir / "subjects_iei"
    if cfg.RUN_IEI:                                   # analisi dell'IEI disattivata (config)
        iei_cache_dir.mkdir(exist_ok=True)
    print("Cartella di output:", out_dir)
    print(f"Soglie (SD): {cfg.THRESHOLD_SD} | IEI {cfg.BIN_SAMPLES if cfg.IEI_MODE == 'per_band' else str(cfg.MAX_IEI_MS) + ' ms'}{' campioni' if cfg.IEI_MODE == 'per_band' else ''} | eventi '{cfg.EVENT_MODE}' | "
          f"bootstrap {cfg.N_BOOT}")

    subjects, iei_subjects, t_start = [], [], time.time()
    for group, folder in cfg.GROUPS.items():
        files = base.subject_files(group, discover_files(folder))
        print(f"{group}: {len(files)} file")
        for fpath in files:
            print("  ", fpath.name)
            try:
                if cfg.RUN_AVALANCHES:
                    res = load_or_run(fpath, group, channels, cache_dir)
                    subjects.append(dict(group=group, subject=fpath.stem, **res))
                if cfg.RUN_IEI:
                    res = load_or_run_iei(fpath, group, channels, iei_cache_dir)
                    iei_subjects.append(dict(group=group, subject=fpath.stem, **res))
            except Exception as e:
                print("    ERRORE ->", e)
    if not subjects and not iei_subjects:
        raise SystemExit("Nessun file analizzato: controlla i percorsi in config_ips.py")
    print(f"\nAnalisi: {(time.time() - t_start) / 60:.1f} min")
    if iei_subjects:
        save_iei_results(iei_subjects, out_dir, channel_set)
    if subjects:
        save_avalanche_results(subjects, out_dir, out_root, channel_set)


def save_avalanche_results(subjects, out_dir, out_root, channel_set):
    """Tabelle di gruppo, soglie, Excel e avalanche_results.pkl delle valanghe."""

    df = pd.DataFrame([r for s in subjects for r in s["rows"]])
    df = df.drop(columns=["mean_iei_ms", "median_iei_ms"], errors="ignore")   # IEI disattivata (cache vecchie)
    tests, summary = group_tables(df)
    dist_fit, dist_summary = distribution_tables(subjects, out_dir)
    br_long, scan = long_tables(subjects)
    br_tests = branching_bin_tests(br_long)
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
              "threshold_divergence": divergence, "branching_ratio_bins": br_long, "branching_ratio_bin_tests": br_tests,
              "power_law_cutoff_scan": scan, "shared_repertoire": shared,
              "distribution_summary": dist_summary, "distribution_fit": dist_fit, "parameters": params}
    save_tables_excel(tables, out_dir / "avalanche_results.xlsx", SHEET_ORDER)
    # per i grafici: tutto tranne i pattern (restano nelle cache per soggetto)
    details = {(s["group"], s["subject"], b): {k: v for k, v in det.items()
                                               if k not in ("unique_patterns", "pattern_counts")}
               for s in subjects for b, det in s["details"].items()}
    save_results(dict(metrics=df, tests=tests, summary=summary, details=details, br=br_long,
                      br_tests=br_tests, dist_fit=dist_fit, dist_summary=dist_summary,
                      scan=scan, divergence=divergence, divergence_curves=div_curves,
                      shared=shared, params=cfg.params(), channel_set=channel_set,
                      bands=dict(cfg.BANDS)),
                 out_dir / "avalanche_results.pkl")
    print("\nFatto. Excel e avalanche_results.pkl in:", out_dir)


if __name__ == "__main__":
    main()
