# -*- coding: utf-8 -*-
"""
Inter-event interval (IEI) robusto: separa gli intervalli intra-valanga da quelli inter-valanga
e ne ricava l'IEI medio da usare come time bin. Funzioni generali: non sanno nulla di gruppi o database.

Metodo
  1. timestamp di tutti gli eventi threshold-crossing di tutti i canali, ordinati, e intervalli
     consecutivi (diff). Gli intervalli si calcolano dentro ogni epoca (le epoche non sono contigue).
     Intervalli = 0 (eventi nello stesso campione su canali diversi) non hanno log: sono esclusi
     dalla distribuzione e contati a parte (n_zero).
  2. log10 degli intervalli e istogramma. Attesa una distribuzione bimodale: intervalli brevi
     (intra-valanga, ~4-20 ms) e lunghi (inter-valanga, ~100 ms - s).
     Gli intervalli sono multipli del periodo di campionamento: per GMM e istogramma ogni intervallo
     di k campioni e' spostato a caso in [k - 0.5, k + 0.5) (dequantizzazione), altrimenti a scale
     brevi il log ha picchi isolati (1, 2, 3 campioni) che la GMM puo' prendere per una componente.
     Media e popolazioni restano sugli intervalli veri.
  3. soglia tra le due popolazioni, in ordine di preferenza:
       a. GMM a 2 componenti nello spazio log: soglia dove le due gaussiane pesate si incrociano
          (tra le due medie). Accettata solo se le componenti sono separate (Ashman's D >= min_ashman_d;
          D > 2 ~ mistura bimodale, Ashman et al. 1994).
       b. valle dell'istogramma log smussato (gaussian_filter1d + find_peaks sulla curva negata):
          minimo piu' profondo tra i due picchi piu' prominenti.
       c. fallback: mediana x fallback_factor (distribuzione unimodale o fit fallito).
  4. intervalli sotto soglia = intra-valanga; la loro media e' l'IEI robusto (iei_mean_s).

Riferimenti
  Beggs & Plenz (2003) J Neurosci 23(35):11167-77 (IEI medio come time bin)
  Ashman, Bird & Zepf (1994) J Mol Evol / AJ 108:2348 (D di separazione tra due gaussiane)
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import brentq
from scipy.signal import find_peaks
from scipy.stats import norm

# griglia comune del log10 IEI (s): 0.3 ms - 30 s, bin di 0.05 decadi (stessa per tutti i
# soggetti, cosi' gli istogrammi si possono mediare per gruppo)
LOG_EDGES = np.arange(-3.5, 1.5 + 1e-9, 0.05)


def event_intervals(events):
    """
    Intervalli (in campioni) tra eventi consecutivi, tutti i canali insieme, dentro ogni epoca.
    events: lista (una voce per epoca) di array (n_times,) con il numero di eventi per campione
            (somma sui canali), oppure (n_channels, n_times) binari; anche un solo array.
    Un campione con n eventi conta n timestamp: tra loro gli intervalli sono 0.
    """
    if isinstance(events, np.ndarray) and events.ndim <= 2:
        events = [events]
    out = []
    for ev in events:
        ev = np.asarray(ev)
        counts = ev.sum(0) if ev.ndim == 2 else ev
        ts = np.repeat(np.arange(counts.size), counts.astype(int))
        if ts.size > 1:
            out.append(np.diff(ts))
    return np.concatenate(out) if out else np.array([], dtype=int)


def _gmm_crossing(w, mu, sd):
    """Punto tra le due medie dove w1 N(mu1, sd1) = w2 N(mu2, sd2). NaN se non c'e'."""
    f = lambda x: w[0] * norm.pdf(x, mu[0], sd[0]) - w[1] * norm.pdf(x, mu[1], sd[1])
    xs = np.linspace(mu[0], mu[1], 2001)
    s = np.sign(f(xs))
    k = np.flatnonzero(s[:-1] * s[1:] < 0)
    if k.size == 0:
        return np.nan
    # se ce n'e' piu' di uno (varianze molto diverse) prende quello dove la mistura e' piu' bassa
    mix = lambda x: w[0] * norm.pdf(x, mu[0], sd[0]) + w[1] * norm.pdf(x, mu[1], sd[1])
    roots = [brentq(f, xs[i], xs[i + 1]) for i in k]
    return float(min(roots, key=mix))


def gmm_threshold(log_iei, max_fit=200_000, seed=0):
    """
    GMM a 2 componenti sul log10 IEI. Ritorna dict con pesi, medie, SD (log10 s, componente
    intra prima), soglia all'incrocio (log10 s), BIC a 1 e 2 componenti, Ashman's D, converged.
    """
    from sklearn.mixture import GaussianMixture
    x = np.asarray(log_iei, dtype=float)
    out = dict(weights=np.full(2, np.nan), means=np.full(2, np.nan), sds=np.full(2, np.nan),
               threshold_log=np.nan, bic1=np.nan, bic2=np.nan, ashman_d=np.nan, converged=False)
    if x.size < 20:
        return out
    if x.size > max_fit:
        x = np.random.default_rng(seed).choice(x, max_fit, replace=False)
    X = x[:, None]
    try:
        g2 = GaussianMixture(2, n_init=3, random_state=seed).fit(X)
        g1 = GaussianMixture(1, random_state=seed).fit(X)
    except Exception:
        return out
    order = np.argsort(g2.means_.ravel())
    w = g2.weights_[order]
    mu = g2.means_.ravel()[order]
    sd = np.sqrt(g2.covariances_.ravel()[order])
    out.update(weights=w, means=mu, sds=sd, bic1=float(g1.bic(X)), bic2=float(g2.bic(X)),
               ashman_d=float(np.sqrt(2) * abs(mu[1] - mu[0]) / np.sqrt(sd[0] ** 2 + sd[1] ** 2)),
               converged=bool(g2.converged_), threshold_log=_gmm_crossing(w, mu, sd))
    return out


def valley_threshold(log_iei, edges=LOG_EDGES, smooth_bins=2.0, min_prominence=0.05):
    """
    Valle dell'istogramma log smussato. min_prominence: frazione del massimo della curva smussata.
    Ritorna dict con la soglia (log10 s, NaN se non ci sono due picchi), i due picchi,
    l'istogramma (densita') e la curva smussata sui centri dei bin.
    """
    x = np.asarray(log_iei, dtype=float)
    counts, _ = np.histogram(x, bins=edges)
    dens = counts / (counts.sum() * np.diff(edges)) if counts.sum() else counts.astype(float)
    centers = (edges[:-1] + edges[1:]) / 2
    smooth = gaussian_filter1d(dens.astype(float), smooth_bins)
    out = dict(threshold_log=np.nan, peaks_log=np.full(2, np.nan), centers=centers, density=dens,
               smooth=smooth)
    if not np.any(smooth > 0):
        return out
    pk, prop = find_peaks(smooth, prominence=min_prominence * smooth.max())
    if pk.size < 2:
        return out
    two = np.sort(pk[np.argsort(prop["prominences"])[-2:]])
    mins, _ = find_peaks(-smooth[two[0]:two[1] + 1])
    if mins.size == 0:
        return out
    v = two[0] + mins[np.argmin(smooth[two[0] + mins])]
    out.update(threshold_log=float(centers[v]), peaks_log=centers[two])
    return out


def robust_iei(events, sfreq, method="auto", fallback_factor=5.0, min_ashman_d=2.0,
               dequantize=True, edges=LOG_EDGES, smooth_bins=2.0, seed=0):
    """
    IEI robusto da una lista di epoche di eventi (vedi event_intervals).

    method          : "auto" (GMM -> valle -> fallback), oppure "gmm", "valley", "fallback" per
                      forzarne uno (se fallisce si passa comunque al fallback).
    fallback_factor : soglia di fallback = mediana degli IEI x fallback_factor.
    min_ashman_d    : separazione minima delle due gaussiane per accettare la soglia della GMM.

    Ritorna dict con
      iei_mean_s    : media degli intervalli intra-valanga (sotto soglia) -> time bin
      threshold_s   : soglia tra intra e inter (s);  method: "gmm" / "valley" / "fallback"
      intra_s, inter_s : le due sotto-popolazioni (s)
      threshold_gmm_s, threshold_valley_s, threshold_fallback_s : le tre soglie (NaN se non trovate)
      gmm (pesi, medie, SD in log10 s, BIC, Ashman's D), valley (picchi, istogramma, curva smussata)
      n_intervals (> 0), n_zero (intervalli = 0, esclusi), median_s, mean_all_s,
      iei_mean_with_zero_s (media intra contando anche gli intervalli = 0)
    """
    iv = event_intervals(events)
    n_zero = int(np.sum(iv == 0))
    iv = iv[iv > 0]
    iei = iv / sfreq
    out = dict(iei_mean_s=np.nan, threshold_s=np.nan, method="none", intra_s=np.array([]),
               inter_s=np.array([]), threshold_gmm_s=np.nan, threshold_valley_s=np.nan,
               threshold_fallback_s=np.nan, n_intervals=int(iv.size), n_zero=n_zero,
               median_s=np.nan, mean_all_s=np.nan, iei_mean_with_zero_s=np.nan,
               gmm=gmm_threshold([]), valley=valley_threshold([], edges, smooth_bins))
    if iv.size == 0:
        return out
    rng = np.random.default_rng(seed)
    lx = np.log10((iv + (rng.uniform(-0.5, 0.5, iv.size) if dequantize else 0)) / sfreq)

    gmm = gmm_threshold(lx, seed=seed)
    val = valley_threshold(lx, edges, smooth_bins)
    thr_g = 10 ** gmm["threshold_log"] if np.isfinite(gmm["threshold_log"]) else np.nan
    thr_v = 10 ** val["threshold_log"] if np.isfinite(val["threshold_log"]) else np.nan
    thr_f = float(np.median(iei) * fallback_factor)
    gmm_ok = gmm["converged"] and np.isfinite(thr_g) and gmm["ashman_d"] >= min_ashman_d

    if method in ("auto", "gmm") and gmm_ok:
        thr, used = thr_g, "gmm"
    elif method in ("auto", "valley") and np.isfinite(thr_v):
        thr, used = thr_v, "valley"
    else:
        thr, used = thr_f, "fallback"

    intra, inter = iei[iei <= thr], iei[iei > thr]
    out.update(iei_mean_s=float(intra.mean()) if intra.size else np.nan, threshold_s=float(thr),
               method=used, intra_s=intra, inter_s=inter, threshold_gmm_s=thr_g,
               threshold_valley_s=thr_v, threshold_fallback_s=thr_f, median_s=float(np.median(iei)),
               mean_all_s=float(iei.mean()), gmm=gmm, valley=val,
               iei_mean_with_zero_s=(float(intra.sum() / (intra.size + n_zero))
                                     if intra.size + n_zero else np.nan))
    return out
