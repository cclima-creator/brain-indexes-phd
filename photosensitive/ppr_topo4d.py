# -*- coding: utf-8 -*-
"""
Spazio 4D topografico di una PPR: i 4 assi dello spazio di criticita' calcolati per ogni elettrodo
in finestre scorrevoli attorno all'esordio di una PPR, e una pagina HTML con 4 topografie
(una per asse) che evolvono nel tempo.

Assi (uno per topografia), per elettrodo e per finestra:
  1. Power        log10( P(slow) / P(fast) )   >0 ritmi lenti, <0 ritmi veloci       (Welch)
  2. Synchrony    node strength della PLV: media della PLV con tutti gli altri elettrodi
                  nella finestra (fase da Hilbert nella banda SYNC_BAND)            (Lachaux 1999)
  3. Avalanches   branching ratio locale: sum e_i(t)N(t+1) / sum e_i(t)N(t) sui time bin da 8 ms
                  in cui l'elettrodo ha un evento (eventi = picchi oltre +-k SD, stessa binarizzazione del modulo
                  avalanche_criticality).  <1 absorbing, >1 spreading               (Beggs & Plenz 2003)
  4. Chaos        Lempel-Ziv (LZ76 normalizzata) del segnale a banda larga:
                  basso = ordinato/regolare, alto = complesso                        (Lempel & Ziv 1976)

Ogni indice e' salvato grezzo e come z rispetto alla finestra di riferimento (default: pre 2),
elettrodo per elettrodo: la topografia mostra quanto ogni zona si sposta dal suo stato iniziale.

Uso (dati preprocessati a rango pieno, NON il dataset a 5 componenti):
  python -m photosensitive.ppr_topo4d --file sogg.set --onset 132.4 --end 135.1 \
      --stim-on 128.0 --stim-off 138.0 --stim-freq 18 --patient 9119460 --ppr-label "PPR 1" \
      --out ppr_topo4d_9119460_ppr1.html
  Con un file a epoche aggiungere --epoch k (tempi in secondi dentro l'epoca, come ep.times).

Avvertenze
  - Su finestre di 1-2 s il branching ratio e la LZC sono stime rumorose: servono a vedere dove e
    quando, non come valori assoluti di criticita'.
  - Dopo l'esordio la punta-onda gonfia potenza, sincronia e valanghe: l'informazione nuova sta
    prima dell'esordio e nei primi centinaia di ms.
  - Con --csd (consigliato) si usa la densita' di corrente di superficie: riduce la conduzione di
    volume e rende la topografia piu' locale.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import welch

from avalanche_criticality.binarization import binarize
from edge_of_chaos.chaos_indexes import compute_lempel_ziv_complexity
from synchrony_criticality.phase import bandpass_filter, analytic_signal

TEMPLATE = Path(__file__).with_name("ppr_topo4d_template.html")

# ---- parametri di default
WINDOW_S = 1.0            # finestra per power, synchrony, chaos
AVAL_WINDOW_S = 2.0       # finestra piu' lunga per il branching (servono abbastanza eventi)
STEP_S = 0.1              # passo tra finestre
SLOW_BAND = (1.0, 8.0)
FAST_BAND = (13.0, 45.0)
SYNC_BAND = (3.0, 30.0)
BROAD_BAND = (1.0, 45.0)
EVENT_SD = 3.0            # soglia eventi delle valanghe
AVAL_BIN_S = 0.008        # time bin delle valanghe
MIN_EVENTS = 3            # eventi minimi dell'elettrodo nella finestra per stimare il branching
PAD_S = 2.0               # margine per i filtri, tolto prima del calcolo

AXES = [
    dict(key="power", name="Power", lo="fast rhythms", hi="slow rhythms",
         unit="log10 P(slow)/P(fast)"),
    dict(key="sync", name="Synchrony", lo="asynchrony", hi="synchrony", unit="PLV node strength"),
    dict(key="aval", name="Avalanches", lo="absorbing", hi="spreading", unit="local branching ratio"),
    dict(key="chaos", name="Chaos", lo="order", hi="complexity", unit="LZ76 normalized"),
]


# ================================================================== INDICI
def window_centers(t0, t1, win_s, step_s):
    """Centri delle finestre tra t0 e t1 (s) con passo step_s."""
    return np.arange(t0 + win_s / 2, t1 - win_s / 2 + 1e-9, step_s)


def power_ratio(x, sfreq, slow=SLOW_BAND, fast=FAST_BAND):
    """log10(P slow / P fast) per canale. x: (n_ch, n_t)."""
    f, p = welch(x, fs=sfreq, nperseg=min(x.shape[-1], int(sfreq)), axis=-1)
    ps = p[:, (f >= slow[0]) & (f < slow[1])].sum(-1)
    pf = p[:, (f >= fast[0]) & (f < fast[1])].sum(-1)
    return np.log10(ps / np.maximum(pf, 1e-30))


def plv_strength(u):
    """u: fasori unitari (n_ch, n_t). Media della PLV di ogni canale con gli altri."""
    plv = np.abs(u @ u.conj().T) / u.shape[1]
    np.fill_diagonal(plv, np.nan)
    return np.nanmean(plv, axis=1)


def event_bins(x, sfreq, k=EVENT_SD, bin_s=AVAL_BIN_S):
    """Eventi (picchi oltre +-k SD, z per canale sul segmento) raggruppati in time bin: (n_ch, n_bin) 0/1."""
    z = (x - x.mean(-1, keepdims=True)) / np.where(x.std(-1, keepdims=True) == 0, np.nan,
                                                    x.std(-1, keepdims=True))
    ev = binarize(np.nan_to_num(z), k, mode="peak", thresh_type="both")
    bl = max(1, int(round(bin_s * sfreq)))
    nb = ev.shape[1] // bl
    return (ev[:, :nb * bl].reshape(ev.shape[0], nb, bl).sum(-1) > 0).astype(np.int16), bl


def local_branching(eb, min_events=MIN_EVENTS):
    """
    Branching locale per canale (rapporto delle somme, piu' stabile della media dei rapporti):
      sigma_i = sum_t e_i(t) N(t+1) / sum_t e_i(t) N(t)
    cioe' eventi nel bin successivo rispetto agli eventi del bin in cui l'elettrodo e' attivo.
    NaN se l'elettrodo ha meno di min_events eventi nella finestra.
    """
    n = eb.sum(0).astype(float)
    e = eb[:, :-1].astype(float)
    num, den = e @ n[1:], e @ n[:-1]
    return np.where(e.sum(1) >= min_events, num / np.maximum(den, 1), np.nan)


def compute_axes(data, sfreq, t_start, t_before, t_after, win_s=WINDOW_S, aval_win_s=AVAL_WINDOW_S,
                 step_s=STEP_S, sync_band=SYNC_BAND):
    """
    data: (n_ch, n_t) in uV che inizia al tempo t_start (s, relativo all'esordio) e copre almeno
    [-t_before - PAD_S, t_after + PAD_S]. Ritorna (tempi dei centri, dict asse -> (n_win, n_ch)).
    """
    def idx(t):
        return int(round((t - t_start) * sfreq))

    broad = bandpass_filter(data, sfreq, BROAD_BAND)
    u = analytic_signal(bandpass_filter(data, sfreq, sync_band))
    u = u / np.maximum(np.abs(u), 1e-30)
    a, b = idx(-t_before), idx(t_after)
    eb, bl = event_bins(broad[:, a:b], sfreq)

    tc = window_centers(-t_before, t_after, max(win_s, aval_win_s), step_s)
    out = {ax["key"]: np.full((tc.size, data.shape[0]), np.nan) for ax in AXES}
    for w, t in enumerate(tc):
        i0, i1 = idx(t - win_s / 2), idx(t + win_s / 2)
        out["power"][w] = power_ratio(data[:, i0:i1], sfreq)
        out["sync"][w] = plv_strength(u[:, i0:i1])
        out["chaos"][w] = [compute_lempel_ziv_complexity(c) for c in broad[:, i0:i1]]
        j0 = max(0, (idx(t - aval_win_s / 2) - a) // bl)
        j1 = (idx(t + aval_win_s / 2) - a) // bl
        out["aval"][w] = local_branching(eb[:, j0:j1])
    return tc, out


def baseline_z(values, tc, ref):
    """z per canale rispetto alle finestre con centro in ref = (t0, t1)."""
    m = (tc >= ref[0]) & (tc < ref[1])
    mu = np.nanmean(values[m], axis=0)
    sd = np.nanstd(values[m], axis=0)
    floor = 0.25 * np.nanstd(values, axis=0)        # evita z enormi se pre 2 e' quasi costante
    sd = np.fmax(sd, floor)
    sd = np.where(~np.isfinite(sd) | (sd < 1e-12), np.nan, sd)
    return (values - mu) / sd


# ================================================================== POSIZIONI 2D
def topo_coords(info):
    """
    Proiezione azimutale equidistante delle posizioni degli elettrodi: sfera fittata ai minimi
    quadrati, angolo dal vertice. Scala: l'elettrodo piu' esterno a r = 0.47 (testa r = 0.5).
    """
    pos = np.array([ch["loc"][:3] for ch in info["chs"]], float)
    if not np.isfinite(pos).all() or np.allclose(pos, 0):
        raise ValueError("posizioni degli elettrodi mancanti: impostare un montaggio (--montage)")
    A = np.c_[2 * pos, np.ones(len(pos))]                       # sfera: |p|^2 = 2 c.p + k
    c = np.linalg.lstsq(A, (pos ** 2).sum(1), rcond=None)[0][:3]
    p = pos - c
    theta = np.arccos(np.clip(p[:, 2] / np.linalg.norm(p, axis=1), -1, 1))
    phi = np.arctan2(p[:, 1], p[:, 0])
    rho = theta / theta.max() * 0.47
    return np.c_[rho * np.cos(phi), rho * np.sin(phi)]


# ================================================================== I/O
def load_segment(path, onset, t_before, t_after, epoch=None, montage="standard_1020", csd=False):
    """Carica il segmento [onset - t_before - PAD, onset + t_after + PAD] (s). Ritorna (data uV, sfreq, info, t_start)."""
    import mne
    path = str(path)
    if epoch is not None:
        ep = mne.io.read_epochs_eeglab(path, verbose="ERROR") if path.endswith(".set") \
            else mne.read_epochs(path, verbose="ERROR")
        ep = ep[epoch].pick("eeg")
        raw = mne.io.RawArray(ep.get_data()[0], ep.info, first_samp=0, verbose="ERROR")
        t0 = ep.times[0]
    else:
        raw = mne.io.read_raw(path, preload=True, verbose="ERROR").pick("eeg")
        t0 = 0.0
    locs = np.array([c["loc"][:3] for c in raw.info["chs"]], float)
    if montage and (not np.isfinite(locs).all() or np.allclose(locs, 0)):
        raw.set_montage(montage, on_missing="ignore", match_case=False)
    if csd:
        raw = mne.preprocessing.compute_current_source_density(raw, verbose="ERROR")
    sf = raw.info["sfreq"]
    a = int(round((onset - t_before - PAD_S - t0) * sf))
    b = int(round((onset + t_after + PAD_S - t0) * sf))
    if a < 0 or b > raw.n_times:
        raise ValueError("il segmento richiesto esce dal file: ridurre --before/--after")
    data = raw.get_data()[:, a:b] * (1.0 if csd else 1e6)
    return data, sf, raw.info, -t_before - PAD_S


def build_payload(data, sfreq, info, t_start, meta, phases, ref, t_before, t_after, **kw):
    tc, raw_vals = compute_axes(data, sfreq, t_start, t_before, t_after, **kw)
    names = [c["ch_name"] for c in info["chs"]]
    xy = topo_coords(info)

    def r(a):
        return [[None if not np.isfinite(v) else round(float(v), 4) for v in row] for row in a]

    return dict(meta=meta, channels=[dict(name=n, x=round(float(p[0]), 4), y=round(float(p[1]), 4))
                                     for n, p in zip(names, xy)],
                times=[round(float(t), 3) for t in tc], phases=phases, reference=list(ref),
                axes=[dict(ax, raw=r(raw_vals[ax["key"]]),
                           z=r(baseline_z(raw_vals[ax["key"]], tc, ref))) for ax in AXES])


def write_html(payload, out):
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "/*__DATA__*/null", json.dumps(payload, separators=(",", ":")))
    Path(out).write_text(html, encoding="utf-8")
    return out


def default_phases(end_rel, t_before, t_after, stim=None, eyes=None):
    """pre 2 / pre 1 / PPR / post relativi all'esordio, come nella figura dello spazio di criticita'."""
    ph = [dict(name="pre 2", t0=-min(4.0, t_before), t1=-2.0, color="#4f86f7"),
          dict(name="pre 1", t0=-2.0, t1=0.0, color="#9db7f5"),
          dict(name="PPR", t0=0.0, t1=end_rel, color="#ff5a5f"),
          dict(name="post", t0=end_rel, t1=min(end_rel + 2.0, t_after), color="#ffb347")]
    return dict(segments=ph, stim=stim, eyes_closed=eyes)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", required=True)
    p.add_argument("--epoch", type=int, default=None)
    p.add_argument("--onset", type=float, required=True, help="esordio PPR (s, tempo del file/epoca)")
    p.add_argument("--end", type=float, required=True, help="fine PPR (s)")
    p.add_argument("--stim-on", type=float, default=None)
    p.add_argument("--stim-off", type=float, default=None)
    p.add_argument("--stim-freq", type=float, default=None)
    p.add_argument("--eyes-closed", type=float, default=None, help="istante di chiusura occhi (s)")
    p.add_argument("--before", type=float, default=6.0)
    p.add_argument("--after", type=float, default=None, help="default: fine PPR + 3 s")
    p.add_argument("--patient", default="")
    p.add_argument("--ppr-label", default="PPR")
    p.add_argument("--montage", default="standard_1020")
    p.add_argument("--csd", action="store_true")
    p.add_argument("--simulated", action="store_true", help="marca la pagina come dati simulati")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    end_rel = a.end - a.onset
    after = a.after if a.after is not None else end_rel + 3.0
    data, sf, info, t_start = load_segment(a.file, a.onset, a.before, after, a.epoch, a.montage, a.csd)
    rel = (lambda t: None if t is None else round(t - a.onset, 3))
    stim = None if a.stim_on is None else dict(t0=rel(a.stim_on), t1=rel(a.stim_off), freq=a.stim_freq)
    phases = default_phases(end_rel, a.before, after, stim, rel(a.eyes_closed))
    meta = dict(patient=a.patient, ppr=a.ppr_label, file=Path(a.file).name, simulated=a.simulated,
                csd=a.csd, sfreq=sf, window_s=WINDOW_S, aval_window_s=AVAL_WINDOW_S, step_s=STEP_S,
                sync_band=SYNC_BAND, event_sd=EVENT_SD, aval_bin_ms=AVAL_BIN_S * 1000)
    ref = (phases["segments"][0]["t0"], phases["segments"][0]["t1"])
    write_html(build_payload(data, sf, info, t_start, meta, phases, ref, a.before, after), a.out)
    print("scritto", a.out)


if __name__ == "__main__":
    main()
