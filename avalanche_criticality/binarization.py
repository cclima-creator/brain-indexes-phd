# -*- coding: utf-8 -*-
"""
Binarizzazione del segnale EEG e rilevamento delle neuronal avalanches.
Funzioni generali: non sanno nulla di gruppi o database (soglie e parametri arrivano da fuori).

Metodo (Maschke et al. 2024, Methods "Avalanche criticality analysis"; Shriki et al. 2013):
  1. z-score per canale: z = (x - media) / SD
  2. soglia = punto in cui la distribuzione degli z-score si stacca dalla gaussiana di fit
     (per IPS dall'analisi preliminare: 1.5 SD fotosensibili, 2.0 SD non fotosensibili;
     divergence_threshold() la ricalcola dagli istogrammi)
  3. evento = picco di ogni escursione oltre +soglia o sotto -soglia: 1 nel campione del
     massimo |z| dell'escursione, 0 altrove  (mode="peak", come edgeofpy / Maschke)
     In alternativa mode="excursion": ogni campione oltre soglia e' attivo (Sorrentino et al. 2021)
  4. valanga = sequenza di eventi (su qualsiasi canale) separati da un intervallo <= max_iei
     (8 ms in Maschke 2024). Dimensione S = numero di eventi, durata T = numero di time bin
     di lunghezza max_iei, pattern = canali con almeno un evento (Sorrentino et al. 2021).

Riferimenti
  Beggs & Plenz (2003) J Neurosci 23(35):11167-77. doi:10.1523/JNEUROSCI.23-35-11167.2003
  Shriki et al. (2013) J Neurosci 33(16):7079-90. doi:10.1523/JNEUROSCI.4286-12.2013
  Sorrentino et al. (2021) Sci Rep 11:4051. doi:10.1038/s41598-021-83425-4
  Maschke et al. (2024) Commun Biol 7:946. doi:10.1038/s42003-024-06613-8
  edgeofpy (O'Byrne, Tholke): https://github.com/jnobyrne/edgeofpy  (detect_avalanches)
"""
import numpy as np


# ------------------------------------------------------------------ SOGLIA
def zscore_channels(data, axis=-1):
    """
    z-score per canale lungo il tempo. data: (..., n_channels, n_times) oppure
    (n_epochs, n_channels, n_times): in questo caso media e SD di ogni canale sono calcolate
    su tutte le epoche insieme (stesso z-score dell'analisi preliminare delle ampiezze).
    Canali piatti (SD = 0) -> NaN.
    """
    x = np.asarray(data, dtype=float)
    if x.ndim == 3:                                   # (n_ep, n_ch, n_t): statistiche per canale su tutte le epoche
        mu = x.mean(axis=(0, 2), keepdims=True)
        sd = x.std(axis=(0, 2), keepdims=True)
    else:
        mu = x.mean(axis=axis, keepdims=True)
        sd = x.std(axis=axis, keepdims=True)
    sd = np.where(sd == 0, np.nan, sd)
    return (x - mu) / sd


def zscore_histogram(z, n_sigma=10.0, n_bins=400):
    """Densita' degli z-score (tutti i canali insieme) tra -n_sigma e +n_sigma (Maschke: -10..10 SD)."""
    edges = np.linspace(-n_sigma, n_sigma, n_bins + 1)
    zf = np.asarray(z, dtype=float).ravel()
    dens, _ = np.histogram(zf[np.isfinite(zf)], bins=edges, density=True)
    return (edges[:-1] + edges[1:]) / 2, dens


def divergence_threshold(centers, heavier, min_sd=1.0, mu=0.0):
    """
    Punto di divergenza dalla gaussiana (Maschke 2024, Fig. 3; Shriki 2013): il |z| da cui in poi
    la distribuzione dei dati sta sopra la gaussiana di fit (coda piu' pesante).

    centers : centri dei bin degli z-score.
    heavier : bool, stessa lunghezza; True dove i dati superano la gaussiana in modo significativo
              (es. 'heavier' di descriptive_analysis.plots.amplitude_deviation_from_gaussian,
              cioe' gaussiana sotto l'IC 95% bootstrap del gruppo).
    min_sd  : si cerca solo oltre min_sd: vicino al centro una distribuzione leptocurtica
              e' anch'essa sopra la gaussiana, ma quello e' il picco, non la coda.

    Per ogni lato prende il primo bin 'heavier' oltre min_sd andando dal centro verso la coda;
    ritorna (soglia media dei due lati, lato positivo, lato negativo). NaN se un lato non diverge.
    """
    c = np.asarray(centers, dtype=float) - mu
    h = np.asarray(heavier, dtype=bool)
    out = []
    for side in (1, -1):
        m = (side * c) >= min_sd
        cs, hs = side * c[m], h[m]
        out.append(float(np.min(cs[hs])) if hs.any() else np.nan)
    pos, neg = out
    return float(np.nanmean([pos, neg])) if np.isfinite([pos, neg]).any() else np.nan, pos, neg


# ------------------------------------------------------------------ EVENTI
def _runs(mask):
    """(inizio, fine esclusa) dei tratti consecutivi True in un vettore booleano."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def binarize(z, threshold, mode="peak", thresh_type="both"):
    """
    Segnale binario degli eventi. z: (n_channels, n_times) gia' z-scored.

    threshold   : soglia in SD.
    mode        : "peak" = un evento per escursione, nel campione con |z| massimo
                  (Shriki 2013, edgeofpy.binarized_events usato da Maschke 2024);
                  "excursion" = tutti i campioni oltre soglia (Sorrentino 2021).
    thresh_type : "both" (|z| > soglia), "above" (z > soglia), "below" (z < -soglia).
    Escursione positiva e negativa sono separate: un passaggio diretto da +soglia a -soglia
    senza rientrare nella banda produce due eventi distinti.
    Ritorna array int8 (n_channels, n_times) con 0/1.
    """
    z = np.asarray(z, dtype=float)
    up = z > threshold if thresh_type in ("both", "above") else np.zeros(z.shape, bool)
    lo = z < -threshold if thresh_type in ("both", "below") else np.zeros(z.shape, bool)
    if mode == "excursion":
        return (up | lo).astype(np.int8)
    if mode != "peak":
        raise ValueError("mode deve essere 'peak' o 'excursion'")
    ev = np.zeros(z.shape, dtype=np.int8)
    az = np.abs(np.nan_to_num(z))
    for ch in range(z.shape[0]):
        for mask in (up[ch], lo[ch]):
            starts, ends = _runs(mask)
            for s, e in zip(starts, ends):
                ev[ch, s + np.argmax(az[ch, s:e])] = 1
    return ev


# ------------------------------------------------------------------ VALANGHE
def bin_counts(counts, bin_len):
    """Somma una serie di conteggi (n_times,) in bin consecutivi di bin_len campioni (l'ultimo incompleto resta)."""
    counts = np.asarray(counts)
    n = int(np.ceil(len(counts) / bin_len)) * bin_len
    padded = np.zeros(n, dtype=counts.dtype)
    padded[:len(counts)] = counts
    return padded.reshape(-1, bin_len).sum(1)


def detect_avalanches(events, sfreq, max_iei_s=0.008, drop_edges=True):
    """
    Valanghe da una matrice di eventi (n_channels, n_times), metodo dell'inter-event interval
    (Beggs & Plenz 2003; Maschke 2024; edgeofpy._det_avls):
    gli eventi di tutti i canali sono messi in fila nel tempo; due eventi consecutivi stanno nella
    stessa valanga se distano <= max_iei campioni, altrimenti inizia una nuova valanga.

    drop_edges : scarta le valanghe che toccano il primo o l'ultimo campione (tagliate dall'inizio
                 o dalla fine dell'epoca, quindi di dimensione e durata sconosciute).

    Ritorna (avalanches, events_per_sample, info):
      avalanches : lista di dict con
          start, end    : primo e ultimo campione con un evento (end incluso)
          size          : S = numero di eventi su tutti i canali
          dur_bin       : T = durata in time bin di lunghezza max_iei, ceil((end - start + 1) / bin)
          dur_s         : durata in secondi (end - start + 1) / sfreq
          n_chan        : numero di canali coinvolti
          profile       : eventi per time bin (n eventi in ogni bin successivo, lunghezza dur_bin)
          pattern       : bool (n_channels,), canali con almeno un evento (Sorrentino 2021)
      events_per_sample : eventi totali per campione (somma sui canali)
      info : dict con bin_len (campioni), mean_iei_s (IEI medio tra eventi successivi)
    """
    events = np.asarray(events)
    n_ch, n_t = events.shape
    bin_len = max(1, int(round(max_iei_s * sfreq)))
    per_sample = events.sum(0)
    idx = np.flatnonzero(per_sample)
    info = dict(bin_len=bin_len, mean_iei_s=np.nan, n_events=int(per_sample.sum()))
    if idx.size == 0:
        return [], per_sample, info
    iei = np.diff(idx)
    if iei.size:
        info["mean_iei_s"] = float(iei.mean() / sfreq)
    breaks = np.flatnonzero(iei > bin_len)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    avalanches = []
    for s, e in zip(starts, ends):
        if drop_edges and (s == 0 or e == n_t - 1):
            continue
        prof = bin_counts(per_sample[s:e + 1], bin_len)
        pattern = events[:, s:e + 1].any(1)
        avalanches.append(dict(start=int(s), end=int(e), size=int(prof.sum()),
                               dur_bin=int(len(prof)), dur_s=(e - s + 1) / sfreq,
                               n_chan=int(pattern.sum()), profile=prof, pattern=pattern))
    return avalanches, per_sample, info


def avalanche_arrays(avalanches):
    """(sizes, durations in bin, patterns (n_av, n_ch) bool) da una lista di valanghe."""
    if not avalanches:
        return np.array([], int), np.array([], int), np.zeros((0, 0), bool)
    sizes = np.array([a["size"] for a in avalanches], dtype=int)
    durs = np.array([a["dur_bin"] for a in avalanches], dtype=int)
    patterns = np.vstack([a["pattern"] for a in avalanches]).astype(bool)
    return sizes, durs, patterns


def avalanches_from_epochs(data, sfreq, threshold, max_iei_s=0.008, mode="peak",
                           thresh_type="both", z=None):
    """
    Pipeline completa per un soggetto con dati a epoche (n_epochs, n_channels, n_times):
    z-score per canale su tutte le epoche, binarizzazione, valanghe dentro ogni epoca
    (una valanga non attraversa il confine tra due epoche, che non sono contigue nel tempo).

    Ritorna dict con
      avalanches : tutte le valanghe (con chiave 'epoch' in piu')
      sizes, durations, patterns : array (vedi avalanche_arrays)
      events     : lista, per epoca, degli eventi per campione (somma sui canali), per BR e Fano
      n_events, mean_iei_s, bin_len, n_samples, duration_s, threshold
    """
    data = np.asarray(data, dtype=float)
    if data.ndim == 2:
        data = data[None]
    if z is None:
        z = zscore_channels(data)
    keep = np.isfinite(z).all(axis=(0, 2))           # scarta canali piatti
    z = z[:, keep]
    avs, ev_list, ieis, n_events = [], [], [], 0
    for k, zk in enumerate(z):
        ev = binarize(zk, threshold, mode, thresh_type)
        a, per_sample, info = detect_avalanches(ev, sfreq, max_iei_s)
        for x in a:
            x["epoch"] = k
            # pattern sui canali originali (i canali piatti restano False)
            full = np.zeros(len(keep), bool)
            full[keep] = x["pattern"]
            x["pattern"] = full
        avs += a
        ev_list.append(per_sample)
        n_events += info["n_events"]
        if np.isfinite(info["mean_iei_s"]):
            ieis.append(info["mean_iei_s"])
    sizes, durs, patterns = avalanche_arrays(avs)
    n_samples = int(data.shape[0] * data.shape[2])
    return dict(avalanches=avs, sizes=sizes, durations=durs, patterns=patterns, events=ev_list,
                n_events=n_events, mean_iei_s=float(np.mean(ieis)) if ieis else np.nan,
                bin_len=max(1, int(round(max_iei_s * sfreq))), n_samples=n_samples,
                duration_s=n_samples / sfreq, threshold=threshold, n_channels=int(keep.sum()))
