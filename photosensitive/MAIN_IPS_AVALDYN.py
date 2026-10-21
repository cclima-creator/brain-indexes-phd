# -*- coding: utf-8 -*-
"""
MAIN dinamica delle valanghe, database IPS. Separato da MAIN_IPS_AVALAN.py ma con le stesse valanghe
(stessa binarizzazione, soglie, bande e time bin: config_ips_avalan.py). Moduli in avalanches_dynamics/.

Per soggetto e banda (cache in criticality_results/avalanches_dynamics/subjects/<file>.pkl):
  valanghe con l'evoluzione canale per canale (traiettorie) e i pattern spaziali.
Per banda, su tutti i soggetti (canali unificati sui nomi 10-10: T3=T7, T4=T8, T5=P7, T6=P8):
  1. repertorio: pattern piu' frequenti per gruppo (D1) e mappa 2D della diversita' (MDS, D2)
  2. PCA comune dei pattern: modi di co-attivazione (D3); per soggetto varianza spiegata da ogni
     componente e participation ratio (dimensionalita' del repertorio, D4)
  3. dinamica: PCA comune degli stati per time bin, campo di deriva e punti fissi per gruppo (D5);
     per soggetto autovalore del fit lineare della deriva (lam_max)
  4. sequenza: stati k-means comuni, matrice di transizione tra valanghe consecutive e
     informazione mutua contro il null rimescolato (D6)
I grafici si fanno dopo con plot_ips_avaldyn.py (senza rifare l'analisi).
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
from photosensitive import config_ips_avaldyn as cfg
from photosensitive.io_eeg import discover_files, load_epochs, get_data_uv
from photosensitive.report import (check_output_dir, save_tables_excel, save_results, load_results,
                                   FALLBACK_DIR)
from photosensitive.MAIN_IPS import pick_channels
from photosensitive.MAIN_IPS_AVALAN import band_filter
from descriptive_analysis.summary_stats import compare_groups
from avalanches_dynamics import patterns as pt
from avalanches_dynamics import state_space as ss
from avalanches_dynamics import transitions as tr
from avalanches_dynamics.topo import canonical_names

warnings.filterwarnings("ignore")

OUTPUT_NAMES = ["avalanche_dynamics.xlsx", "avalanche_dynamics.pkl"]
SUBJECT_METRICS = ["n_avalanches", "n_unique_patterns", "participation_ratio",
                   *[f"pc{i + 1}_explained" for i in range(cfg.N_COMPONENTS)],
                   "lam_max", "mi_bits", "mi_z", "self_transition"]


# ================================================================== UN SOGGETTO
def analyze_subject(fpath, group, channels):
    """Traiettorie e pattern delle valanghe in ogni banda (stesse valanghe di MAIN_IPS_AVALAN)."""
    ep = load_epochs(fpath, base.FIXED_LENGTH_S, base.ANALYSIS_WINDOW)
    if channels is not None:
        ep = pick_channels(ep, channels, base.OCCIPITAL_CHANNELS)
    sfreq = ep.info["sfreq"]
    data = get_data_uv(ep)
    bands = {}
    for b, rng in cfg.BANDS.items():
        x = band_filter(data, sfreq, rng)
        r = ss.avalanche_trajectories(x, sfreq, av.threshold_for(group, b), cfg.bin_s(b, sfreq),
                                      av.EVENT_MODE, av.THRESH_TYPE)
        bands[b] = r
        print(f"    {b:>10}: {len(r['trajectories']):6d} valanghe")
    return dict(bands=bands, ch_names=canonical_names(ep.ch_names), sfreq=sfreq, params=cfg.params())


def load_or_run(fpath, group, channels, cache_dir):
    cache = cache_dir / f"{fpath.stem}.pkl"
    if cache.exists() and not cfg.OVERWRITE:
        res = load_results(cache)
        if res.get("params") == cfg.params():
            print("    (letto dalla cache)")
            return res
    res = analyze_subject(fpath, group, channels)
    with open(cache, "wb") as f:
        pickle.dump(res, f)
    return res


# ================================================================== CANALI COMUNI
def align_channels(subjects):
    """Canali presenti in tutti i soggetti, nell'ordine del primo; per soggetto gli indici."""
    ref = subjects[0]["ch_names"]
    common = [c for c in ref if all(c in s["ch_names"] for s in subjects)]
    for s in subjects:
        s["idx"] = [s["ch_names"].index(c) for c in common]
    dropped = sorted({c for s in subjects for c in s["ch_names"]} - set(common))
    if dropped:
        print("Canali non presenti in tutti i soggetti, esclusi:", dropped)
    return common


# ================================================================== UNA BANDA
def analyze_band(subjects, band, ch_names):
    """Tutte le analisi di gruppo di una banda. Ritorna (righe per soggetto, risultati per i grafici)."""
    groups = [g for g in cfg.GROUPS if any(s["group"] == g for s in subjects)]
    pats = {s["subject"]: s["bands"][band]["patterns"][:, s["idx"]] for s in subjects}
    trajs = {s["subject"]: [t[:, s["idx"]] for t in s["bands"][band]["trajectories"]] for s in subjects}
    by_group = {g: [s["subject"] for s in subjects if s["group"] == g] for g in groups}
    out = dict(band=band, ch_names=ch_names, groups=groups)

    # 1. repertorio: pattern frequenti e mappa della diversita'
    top, union = {}, []
    for g in groups:
        u, f, n_sub = pt.group_frequencies([pats[sid] for sid in by_group[g]])
        um, fm, nm = pt.group_frequencies([pats[sid][pats[sid].sum(1) >= cfg.MIN_MULTI_CHANNELS]
                                           for sid in by_group[g]])
        top[g] = dict(patterns=u[:cfg.N_TOP_PATTERNS], freq=f[:cfg.N_TOP_PATTERNS],
                      n_subjects=n_sub[:cfg.N_TOP_PATTERNS], n_groups_subjects=len(by_group[g]),
                      n_unique=len(u), multi_patterns=um[:cfg.N_TOP_PATTERNS],
                      multi_freq=fm[:cfg.N_TOP_PATTERNS], multi_n_subjects=nm[:cfg.N_TOP_PATTERNS],
                      multi_n_unique=len(um))
        union.append((g, u[:cfg.MDS_MAX_PATTERNS], f[:cfg.MDS_MAX_PATTERNS]))
    keys = {}
    for g, u, f in union:
        for row, fr in zip(u, f):
            keys.setdefault(row.tobytes(), [row, {gg: 0.0 for gg in groups}])[1][g] = fr
    mds_pat = np.array([v[0] for v in keys.values()])
    mds_freq = {g: np.array([v[1][g] for v in keys.values()]) for g in groups}
    mds_xy, mds_expl = pt.hamming_mds(mds_pat)
    out.update(top=top, mds=dict(xy=mds_xy, freq=mds_freq, n_active=mds_pat.sum(1), explained=mds_expl))

    # 2. PCA comune dei pattern
    pca = pt.pattern_pca(pt.balanced_sample(list(pats.values()), cfg.MAX_PER_SUBJECT), cfg.N_COMPONENTS)
    group_evr = {g: pt.explained_by(np.vstack([pats[sid] for sid in by_group[g]]), pca) for g in groups}
    out.update(pca=pca, group_evr=group_evr)

    # 3. dinamica: PCA degli stati per bin, deriva, punti fissi
    spca = pt.pattern_pca(pt.balanced_sample([ss.state_vectors(trajs[sid]) for sid in trajs],
                                             cfg.MAX_PER_SUBJECT), 2)
    trans = {sid: ss.transitions(trajs[sid], spca["components"], spca["mean"]) for sid in trajs}
    allx = np.vstack([trans[sid][0] for sid in trans if len(trans[sid][0])])
    lo, hi = np.percentile(allx, [1, 99], axis=0)
    pad = 0.05 * (hi - lo)
    ex = np.linspace(lo[0] - pad[0], hi[0] + pad[0], cfg.GRID + 1)
    ey = np.linspace(lo[1] - pad[1], hi[1] + pad[1], cfg.GRID + 1)
    dyn = {}
    for g in groups:
        rng = np.random.default_rng(0)
        xs, dxs = [], []
        for sid in by_group[g]:                       # ogni soggetto pesa uguale
            x, dx = trans[sid]
            if len(x) > cfg.MAX_PER_SUBJECT:
                k = rng.choice(len(x), cfg.MAX_PER_SUBJECT, replace=False)
                x, dx = x[k], dx[k]
            xs.append(x)
            dxs.append(dx)
        x, dx = np.vstack(xs), np.vstack(dxs)
        fit = ss.fit_drift_polynomial(x, dx, cfg.POLY_DEGREE)
        dyn[g] = dict(field=ss.drift_field(x, dx, ex, ey, cfg.MIN_COUNT), fit=fit,
                      fixed=ss.fixed_points(fit, (ex[0], ex[-1]), (ey[0], ey[-1])),
                      linear=ss.linear_drift(x, dx), n_transitions=len(x))
    out.update(state_pca=spca, dynamics=dyn, edges=(ex, ey))

    # 4. sequenza delle valanghe
    km = tr.fit_states(pt.balanced_sample(list(pats.values()), cfg.MAX_PER_SUBJECT), cfg.N_STATES)
    seq = {}
    for s in subjects:
        sid = s["subject"]
        order = np.lexsort((s["bands"][band]["start"], s["bands"][band]["epoch"]))
        states = km.predict(pats[sid][order].astype(float))
        seq[sid] = tr.sequence_memory(states, s["bands"][band]["epoch"][order], cfg.N_STATES, cfg.N_PERM)
    out.update(state_centroids=km.cluster_centers_,
               group_P={g: np.nanmean([seq[sid]["P"] for sid in by_group[g]], axis=0) for g in groups},
               state_occupancy={g: np.mean([seq[sid]["counts"].sum(1) / max(seq[sid]["counts"].sum(), 1)
                                            for sid in by_group[g]], axis=0) for g in groups})

    # righe per soggetto
    rows = []
    for s in subjects:
        sid = s["subject"]
        evr = pt.explained_by(pats[sid], pca)
        lin = ss.linear_drift(*trans[sid])
        u, _, _ = pt.pattern_frequencies(pats[sid])
        rows.append(dict(group=s["group"], subject=sid, band=band, n_avalanches=len(pats[sid]),
                         n_unique_patterns=len(u), participation_ratio=pt.participation_ratio(pats[sid]),
                         **{f"pc{i + 1}_explained": v for i, v in enumerate(evr)},
                         lam_max=lin["lam_max"], mi_bits=seq[sid]["mi_bits"], mi_z=seq[sid]["mi_z"],
                         self_transition=seq[sid]["self_transition"],
                         n_transitions_bins=len(trans[sid][0])))
    return rows, out


# ================================================================== MAIN
def main():
    channel_set, channels, out_root = base.choose_channel_set()
    fallback = (FALLBACK_DIR if channels is None else FALLBACK_DIR / base.OCCIPITAL_SUBDIR) / cfg.OUT_NAME
    out_dir = check_output_dir(cfg.results_dir(channel_set, out_root), OUTPUT_NAMES, fallback)
    cache_dir = out_dir / "subjects"
    cache_dir.mkdir(exist_ok=True)
    print("Cartella di output:", out_dir)
    print(f"Soglie (SD): {av.THRESHOLD_SD} | time bin: {av.BIN_SAMPLES} campioni | eventi '{av.EVENT_MODE}'")

    subjects, t0 = [], time.time()
    for group, folder in cfg.GROUPS.items():
        files = base.subject_files(group, discover_files(folder))
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
    print(f"\nTraiettorie: {(time.time() - t0) / 60:.1f} min")
    ch_names = align_channels(subjects)

    rows, results = [], {}
    for b in cfg.BANDS:
        t1 = time.time()
        r, results[b] = analyze_band(subjects, b, ch_names)
        rows += r
        fps = {g: [f["kind"] for f in results[b]["dynamics"][g]["fixed"]] for g in results[b]["groups"]}
        print(f"  {b:>10}: punti fissi {fps}  ({time.time() - t1:.0f} s)")

    df = pd.DataFrame(rows)
    tests = []
    for b in cfg.BANDS:
        t = compare_groups(df[df.band == b], [m for m in SUBJECT_METRICS if m in df])
        if len(t):
            t.insert(0, "band", b)
            tests.append(t)
    tests = pd.concat(tests, ignore_index=True) if tests else pd.DataFrame()
    fixed = pd.DataFrame([dict(band=b, group=g, pc1=f["x"], pc2=f["y"], kind=f["kind"],
                               mu1=abs(f["mu"][0]), mu2=abs(f["mu"][1]))
                          for b, res in results.items() for g in res["groups"]
                          for f in res["dynamics"][g]["fixed"]])
    top = pd.DataFrame([dict(band=b, group=g, rank=i + 1, freq_pct=100 * fr, n_subjects=int(n),
                             n_active=int(p.sum()), channels=" ".join(c for c, a in zip(ch_names, p) if a))
                        for b, res in results.items() for g in res["groups"]
                        for i, (p, fr, n) in enumerate(zip(res["top"][g]["patterns"], res["top"][g]["freq"],
                                                           res["top"][g]["n_subjects"]))])
    if len(tests):
        print("\n=== Test tra gruppi (Mann-Whitney, p_fdr dentro ogni banda), solo p < 0.05 ===")
        sig = tests[tests.p < 0.05]
        print(sig.round(4).to_string(index=False) if len(sig) else "nessuno")
    params = pd.DataFrame([dict(parameter=k, value=str(v)) for k, v in
                           {**cfg.params(), "N_COMPONENTS": cfg.N_COMPONENTS, "N_STATES": cfg.N_STATES,
                            "GRID": cfg.GRID, "POLY_DEGREE": cfg.POLY_DEGREE,
                            "MAX_PER_SUBJECT": cfg.MAX_PER_SUBJECT}.items()])
    save_tables_excel({"subject_band": df, "group_tests": tests, "top_patterns": top,
                       "fixed_points": fixed, "parameters": params}, out_dir / "avalanche_dynamics.xlsx",
                      ["subject_band", "group_tests", "top_patterns", "fixed_points", "parameters"])
    save_results(dict(metrics=df, tests=tests, results=results, ch_names=ch_names, bands=dict(cfg.BANDS),
                      channel_set=channel_set, params=cfg.params()), out_dir / "avalanche_dynamics.pkl")
    print("\nFatto. avalanche_dynamics.xlsx e .pkl in:", out_dir)


if __name__ == "__main__":
    main()
