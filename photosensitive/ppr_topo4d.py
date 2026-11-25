# -*- coding: utf-8 -*-
"""
Spazio 4D topografico di una PPR: i 4 assi dello spazio di criticita' come misure LOCALI
(per elettrodo o per coppia di elettrodi) in finestre scorrevoli attorno all'esordio di una PPR,
piu' l'EEG di tutti i canali e la propagazione degli eventi sopra soglia. Scrive una pagina HTML.

Gli indici globali (branching ratio, repertorio dei pattern, esponenti delle power law, DCC,
metastabilita', K del test 0-1, lambda_max) descrivono il sistema intero e non hanno una
topografia: qui non ci sono. Per ogni asse si tengono misure locali non ridondanti.

  Power (quali zone hanno frequenze alte o basse)
    mf        frequenza mediana (SEF50) dello spettro 1-45 Hz                                  [Hz]
              Il rapporto lento/veloce, il centroide spettrale e le potenze relative dicono la stessa cosa
              (lo spettro e' composizionale); l'esponente aperiodico su 1 s e' instabile.
  Synchrony (coerenza: quali parti sono piu' connesse)
    coupling  per coppia: wPLI nella banda scelta (linee); per elettrodo: node strength = media
              del wPLI con gli altri elettrodi (colore)                                       (Vinck 2011)
              wPLI e non PLV: ignora l'accoppiamento a lag zero prodotto dalla conduzione di volume.
              SYNC_METHOD = "plv" per la PLV (utile su CSD, include la sincronia a lag zero).
    driving   aggancio allo stimolo: SNR (dB) alla frequenza dei flash e alla 2a armonica.
              E' sincronizzazione con l'esterno, diversa dalla coerenza tra elettrodi.
  Chaos (zone piu' caotiche)
    lzc_ns    Lempel-Ziv (LZ76) / media della LZ di surrogati a fase casuale (stesso spettro).
              =1: complessita' spiegata dallo spettro; <1: struttura non lineare piu' regolare
              (es. punta-onda); >1: piu' irregolare. Normalizzare sui surrogati toglie la parte gia'
              vista nell'asse Power; LZ, entropia di permutazione e sample entropy sarebbero ridondanti tra loro.
  Avalanches (ruolo locale nelle valanghe; la propagazione si vede nel pannello eventi)
    rate      eventi al secondo (picchi oltre +-k SD)
    lead      frazione delle valanghe (>= 2 elettrodi) nella finestra che partono da questo elettrodo
    reach     numero medio di elettrodi reclutati dalle valanghe che partono da questo elettrodo

Il colore di ogni testa e' l'ASSE: media degli z (rispetto a pre 2, per elettrodo) delle sue misure;
la pagina permette di vedere anche le singole misure.

Uso (dati preprocessati a rango pieno, NON il dataset a 5 componenti):
  python -m photosensitive.ppr_topo4d --file sogg.set --onset 132.4 --end 135.1 \
      --stim-on 128.0 --stim-off 138.0 --stim-freq 18 --patient 9119460 --ppr-label "PPR 1" \
      --csd --out ppr_topo4d_9119460_ppr1.html
  Con un file a epoche aggiungere --epoch k (tempi in secondi dentro l'epoca, come ep.times).

Avvertenze
  - Su finestre di 1-2 s le stime sono rumorose: servono a vedere dove e quando, non come valori assoluti.
  - Dopo l'esordio la punta-onda gonfia quasi tutte le misure: l'informazione nuova sta prima
    dell'esordio e nei primi centinaia di ms.
  - Con --csd (consigliato) le misure sono calcolate sulla densita' di corrente di superficie
    (meno conduzione di volume); l'EEG mostrato resta quello col riferimento originale.
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
AVAL_WINDOW_S = 2.0       # finestra per le misure delle valanghe (servono abbastanza eventi)
STEP_S = 0.1              # passo tra finestre
BROAD_BAND = (1.0, 45.0)
SYNC_BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
              "beta": (13.0, 30.0), "gamma": (30.0, 45.0)}
SYNC_DEFAULT = "beta"
SYNC_METHOD = "wpli"      # "wpli" o "plv"
SNR_NEIGHBORS = (1.5, 3.0)  # driving: bin vicini tra 1.5 e 3 Hz da entrambi i lati
N_SURROGATES = 4          # surrogati a fase casuale per la LZ normalizzata
EVENT_SD = 3.0            # soglia eventi
AVAL_BIN_S = 0.008        # time bin delle valanghe
EEG_DISPLAY_HZ = 250.0    # frequenza massima dell'EEG salvato nella pagina
PAD_S = 2.0               # margine per i filtri, tolto prima del calcolo


def axes_spec(sync_method=SYNC_METHOD):
    sm = "wPLI" if sync_method == "wpli" else "PLV"
    return [
        dict(key="power", name="Power", lo="low frequencies", hi="high frequencies",
             measures=[dict(key="mf", name="median frequency", unit="Hz")]),
        dict(key="sync", name="Synchrony", lo="asynchrony", hi="synchrony",
             measures=[dict(key="coupling", name=f"coherence ({sm} node strength)", unit=sm, banded=True),
                       dict(key="driving", name="photic driving (SNR f, 2f)", unit="dB")]),
        dict(key="chaos", name="Chaos", lo="order", hi="chaos",
             measures=[dict(key="lzc_ns", name="LZ / phase surrogates", unit="ratio")]),
        dict(key="aval", name="Avalanches", lo="quiet / absorbing", hi="active / spreading",
             measures=[dict(key="rate", name="event rate", unit="events/s"),
                       dict(key="lead", name="leader (starts avalanches)", unit="fraction"),
                       dict(key="reach", name="reach of its avalanches", unit="electrodes")]),
    ]


# ================================================================== MISURE LOCALI
def window_centers(t0, t1, win_s, step_s):
    """Centri delle finestre tra t0 e t1 (s) con passo step_s."""
    return np.arange(t0 + win_s / 2, t1 - win_s / 2 + 1e-9, step_s)


def median_frequency(x, sfreq, band=BROAD_BAND):
    """SEF50: frequenza sotto cui sta meta' della potenza in band. x: (n_ch, n_t)."""
    f, p = welch(x, fs=sfreq, nperseg=min(x.shape[-1], int(sfreq)), axis=-1)
    m = (f >= band[0]) & (f <= band[1])
    c = np.cumsum(p[:, m], axis=1)
    k = (c >= c[:, -1:] / 2).argmax(1)
    return f[m][k]


def pair_coupling(z, method=SYNC_METHOD):
    """
    z: segnale analitico in banda (n_ch, n_t). Ritorna la matrice (n_ch, n_ch).
    wPLI (Vinck 2011) sui campioni della finestra: |sum Im(z_i z_j*)| / sum |Im(z_i z_j*)|.
    PLV: |mean exp(i(phi_i - phi_j))|.
    """
    if method == "plv":
        u = z / np.maximum(np.abs(z), 1e-30)
        return np.abs(u @ u.conj().T) / z.shape[1]
    im = np.imag(z[:, None, :] * z.conj()[None, :, :])
    return np.abs(im.sum(-1)) / np.maximum(np.abs(im).sum(-1), 1e-30)


def photic_snr(x, sfreq, f0, neighbors=SNR_NEIGHBORS, fmax=BROAD_BAND[1]):
    """SNR medio (dB) a f0 e 2 f0 (se <= fmax) rispetto ai bin vicini. x: (n_ch, n_t)."""
    n = x.shape[-1]
    f, p = welch(x, fs=sfreq, nperseg=n, nfft=4 * n, axis=-1)
    out = []
    for h in (1, 2):
        fh = h * f0
        if fh + neighbors[1] > fmax:
            continue
        k = np.argmin(np.abs(f - fh))
        d = np.abs(f - fh)
        nb = (d >= neighbors[0]) & (d <= neighbors[1])
        out.append(10 * np.log10(p[:, k] / np.maximum(p[:, nb].mean(1), 1e-30)))
    return np.mean(out, axis=0) if out else np.full(x.shape[0], np.nan)


def phase_surrogates(x, n, rng):
    """n surrogati a fase casuale di x (1D): stesso spettro di ampiezza."""
    X = np.fft.rfft(x)
    ph = rng.uniform(0, 2 * np.pi, (n, X.size))
    ph[:, 0] = 0
    if x.size % 2 == 0:
        ph[:, -1] = 0
    return np.fft.irfft(np.abs(X) * np.exp(1j * ph), x.size)


def lzc_surrogate_ratio(x, n_surr=N_SURROGATES, rng=None):
    """LZ76 di x divisa per la media della LZ76 dei suoi surrogati a fase casuale."""
    rng = rng or np.random.default_rng(0)
    lz = compute_lempel_ziv_complexity(x)
    s = np.mean([compute_lempel_ziv_complexity(y) for y in phase_surrogates(x, n_surr, rng)])
    return lz / s if s > 0 else np.nan


def event_bins(x, sfreq, k=EVENT_SD, bin_s=AVAL_BIN_S):
    """Eventi (picchi oltre +-k SD, z per canale sul segmento) in time bin: (n_ch, n_bin) 0/1."""
    sd = x.std(-1, keepdims=True)
    z = (x - x.mean(-1, keepdims=True)) / np.where(sd == 0, np.nan, sd)
    ev = binarize(np.nan_to_num(z), k, mode="peak", thresh_type="both")
    bl = max(1, int(round(bin_s * sfreq)))
    nb = ev.shape[1] // bl
    return (ev[:, :nb * bl].reshape(ev.shape[0], nb, bl).sum(-1) > 0).astype(np.int8)


def avalanches_from_bins(eb):
    """Valanghe = tratti di bin consecutivi con almeno un evento. Lista di (start, end incl., iniziatori, n elettrodi)."""
    act = eb.any(0)
    d = np.diff(np.r_[0, act.astype(np.int8), 0])
    out = []
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1):
        out.append((int(s), int(e), np.flatnonzero(eb[:, s]), int(eb[:, s:e + 1].any(1).sum())))
    return out


def avalanche_measures(eb, avs, j0, j1, win_s):
    """rate, lead, reach per elettrodo nei bin [j0, j1)."""
    n_ch = eb.shape[0]
    rate = eb[:, j0:j1].sum(1) / win_s
    inside = [a for a in avs if j0 <= a[0] < j1]
    multi = [a for a in inside if a[3] >= 2]
    lead = np.full(n_ch, np.nan)
    if multi:
        lead = np.zeros(n_ch)
        for a in multi:
            lead[a[2]] += 1
        lead /= len(multi)
    reach_s, reach_n = np.zeros(n_ch), np.zeros(n_ch)
    for a in inside:
        reach_s[a[2]] += a[3]
        reach_n[a[2]] += 1
    reach = np.where(reach_n > 0, reach_s / np.maximum(reach_n, 1), np.nan)
    return rate, lead, reach


def compute_measures(data, sfreq, t_start, t_before, t_after, stim_freq=None, win_s=WINDOW_S,
                     aval_win_s=AVAL_WINDOW_S, step_s=STEP_S, sync_method=SYNC_METHOD, seed=0):
    """
    data: (n_ch, n_t) che inizia al tempo t_start (s, relativo all'esordio) e copre almeno
    [-t_before - PAD_S, t_after + PAD_S].
    Ritorna dict con times, misure (n_win, n_ch), coppie (n_win, n_pairs) per banda, eventi.
    """
    rng = np.random.default_rng(seed)

    def idx(t):
        return int(round((t - t_start) * sfreq))

    n_ch = data.shape[0]
    broad = bandpass_filter(data, sfreq, BROAD_BAND)
    zb = {b: analytic_signal(bandpass_filter(data, sfreq, rg)) for b, rg in SYNC_BANDS.items()}
    a, b = idx(-t_before), idx(t_after)
    eb = event_bins(broad[:, a:b], sfreq)
    avs = avalanches_from_bins(eb)
    bin_n = max(1, int(round(AVAL_BIN_S * sfreq)))

    tc = window_centers(-t_before, t_after, max(win_s, aval_win_s), step_s)
    iu = np.triu_indices(n_ch, 1)
    nw = tc.size
    M = {k: np.full((nw, n_ch), np.nan) for k in ("mf", "driving", "lzc_ns", "rate", "lead", "reach")}
    node = {bd: np.full((nw, n_ch), np.nan) for bd in SYNC_BANDS}
    pairs = {bd: np.zeros((nw, iu[0].size)) for bd in SYNC_BANDS}
    for w, t in enumerate(tc):
        i0, i1 = idx(t - win_s / 2), idx(t + win_s / 2)
        M["mf"][w] = median_frequency(broad[:, i0:i1], sfreq)
        if stim_freq:
            M["driving"][w] = photic_snr(data[:, i0:i1], sfreq, stim_freq)
        M["lzc_ns"][w] = [lzc_surrogate_ratio(c, rng=rng) for c in broad[:, i0:i1]]
        for bd, z in zb.items():
            C = pair_coupling(z[:, i0:i1], sync_method)
            pairs[bd][w] = C[iu]
            np.fill_diagonal(C, np.nan)
            node[bd][w] = np.nanmean(C, 1)
        j0 = max(0, (idx(t - aval_win_s / 2) - a) // bin_n)
        j1 = (idx(t + aval_win_s / 2) - a) // bin_n
        M["rate"][w], M["lead"][w], M["reach"][w] = avalanche_measures(eb, avs, j0, j1, aval_win_s)
    events = [[int(k), np.flatnonzero(eb[:, k]).tolist()] for k in np.flatnonzero(eb.any(0))]
    return dict(times=tc, measures=M, node=node, pairs=pairs, pair_index=np.c_[iu].tolist(),
                events=dict(bin_s=AVAL_BIN_S, t0=-t_before, n_bins=int(eb.shape[1]), bins=events))


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
    """
    Carica il segmento [onset - t_before - PAD, onset + t_after + PAD] (s).
    Ritorna (data per le misure, eeg per la pagina in uV col riferimento originale, sfreq, info, t_start).
    """
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
    sf = raw.info["sfreq"]
    a = int(round((onset - t_before - PAD_S - t0) * sf))
    b = int(round((onset + t_after + PAD_S - t0) * sf))
    if a < 0 or b > raw.n_times:
        raise ValueError("il segmento richiesto esce dal file: ridurre --before/--after")
    eeg = raw.get_data()[:, a:b] * 1e6
    data = eeg
    if csd:
        data = mne.preprocessing.compute_current_source_density(raw, verbose="ERROR").get_data()[:, a:b]
    return data, eeg, sf, raw.info, -t_before - PAD_S


def eeg_for_page(eeg, sfreq, t_start, t_before, t_after):
    """EEG 1-45 Hz di tutti i canali su [-t_before, t_after], al massimo EEG_DISPLAY_HZ."""
    x = bandpass_filter(eeg, sfreq, BROAD_BAND)
    a = int(round((-t_before - t_start) * sfreq))
    b = int(round((t_after - t_start) * sfreq))
    step = max(1, int(np.floor(sfreq / EEG_DISPLAY_HZ)))
    x = x[:, a:b:step]
    return dict(sfreq=sfreq / step, t0=-t_before, unit="µV",
                data=[[round(float(v), 1) for v in ch] for ch in x])


def _r(a, nd=4):
    return [[None if not np.isfinite(v) else round(float(v), nd) for v in row] for row in np.atleast_2d(a)]


def build_payload(data, eeg, sfreq, info, t_start, meta, phases, ref, t_before, t_after,
                  stim_freq=None, sync_method=SYNC_METHOD, **kw):
    res = compute_measures(data, sfreq, t_start, t_before, t_after, stim_freq,
                           sync_method=sync_method, **kw)
    tc = res["times"]
    axes = axes_spec(sync_method)
    for ax in axes:
        keep = []
        for ms in ax["measures"]:
            if ms.get("banded"):
                ms["bands"] = {bd: dict(range=list(SYNC_BANDS[bd]), raw=_r(v), z=_r(baseline_z(v, tc, ref)))
                               for bd, v in res["node"].items()}
                ms["default_band"] = SYNC_DEFAULT
            else:
                v = res["measures"][ms["key"]]
                if not np.isfinite(v).any():
                    continue                      # es. driving senza frequenza di stimolo
                ms["raw"], ms["z"] = _r(v), _r(baseline_z(v, tc, ref))
            keep.append(ms)
        ax["measures"] = keep
        if ax["key"] == "sync":
            ax["pair_index"] = res["pair_index"]
            ax["pairs"] = {bd: _r(v, 3) for bd, v in res["pairs"].items()}
    names = [c["ch_name"] for c in info["chs"]]
    xy = topo_coords(info)
    return dict(meta=meta, channels=[dict(name=n, x=round(float(p[0]), 4), y=round(float(p[1]), 4))
                                     for n, p in zip(names, xy)],
                times=[round(float(t), 3) for t in tc], phases=phases, reference=list(ref),
                axes=axes, events=res["events"],
                eeg=eeg_for_page(eeg, sfreq, t_start, t_before, t_after))


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
    p.add_argument("--sync-method", choices=("wpli", "plv"), default=SYNC_METHOD)
    p.add_argument("--simulated", action="store_true", help="marca la pagina come dati simulati")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    end_rel = a.end - a.onset
    after = a.after if a.after is not None else end_rel + 3.0
    data, eeg, sf, info, t_start = load_segment(a.file, a.onset, a.before, after, a.epoch,
                                                a.montage, a.csd)
    rel = (lambda t: None if t is None else round(t - a.onset, 3))
    stim = None if a.stim_on is None else dict(t0=rel(a.stim_on), t1=rel(a.stim_off), freq=a.stim_freq)
    phases = default_phases(end_rel, a.before, after, stim, rel(a.eyes_closed))
    meta = dict(patient=a.patient, ppr=a.ppr_label, file=Path(a.file).name, simulated=a.simulated,
                csd=a.csd, sfreq=sf, window_s=WINDOW_S, aval_window_s=AVAL_WINDOW_S, step_s=STEP_S,
                sync_method=a.sync_method, event_sd=EVENT_SD, aval_bin_ms=AVAL_BIN_S * 1000)
    ref = (phases["segments"][0]["t0"], phases["segments"][0]["t1"])
    write_html(build_payload(data, eeg, sf, info, t_start, meta, phases, ref, a.before, after,
                             stim_freq=a.stim_freq, sync_method=a.sync_method), a.out)
    print("scritto", a.out)


if __name__ == "__main__":
    main()
