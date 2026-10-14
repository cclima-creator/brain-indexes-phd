# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 14:11:18 2026

@author: Chiara Clima
"""
"""
PSD
 ↓
integrazione nelle bande
 ↓
potenza assoluta
 ↓
potenza relativa
 ↓
log-potenza
 ↓
tabella pandas
"""


"""Spettro di potenza, potenza per banda ed esponente aperiodico (1/f). Funzioni pure su array numpy."""
import numpy as np                      ##array e matematica
import pandas as pd                     ##tabelle e risultati
from scipy.signal import welch          ## welch = metodo per stimare power spectral density
from scipy.integrate import trapezoid    ## integrali


def compute_psd(data_uv, sfreq, fmin=1.0, fmax=45.0, segment_s=None, overlap=0.5,
                window="hann"):
    """
    PSD di Welch per epoca e canale.

    data_uv : (n_epochs, n_channels, n_times) in uV               ##sono i dati in microvolt del codice di importazione
    segment_s : None = un solo segmento = intera epoca (risoluzione 1/T Hz, le frequenze
                intere cadono su un bin se T e' intero); altrimenti durata del segmento
                in secondi (piu' media, meno risoluzione).
    Ritorna freqs (n_f,), psd (n_epochs, n_channels, n_f) in uV^2/Hz.
    """
    n_t = data_uv.shape[-1]
    if segment_s is None:                             ##segment none= uso intera epoca come segmento
        nperseg, noverlap = n_t, 0
    else:
        nperseg = min(int(round(segment_s * sfreq)), n_t)
        noverlap = int(nperseg * overlap)
    freqs, psd = welch(data_uv, fs=sfreq, window=window, nperseg=nperseg,
                       noverlap=noverlap, axis=-1)
    m = (freqs >= fmin) & (freqs <= min(fmax, sfreq / 2))
    return freqs[m], psd[..., m]                 ##ho stima discreta!!


def integrate_band(psd, freqs, lo, hi):
    """Integrale della PSD tra lo (incluso) e hi (escluso). psd: (..., n_f)."""
    m = (freqs >= lo) & (freqs < hi)
    return trapezoid(psd[..., m], freqs[m], axis=-1)        ##rende continua integrando con trapezoid


def band_power_table(psd, freqs, ch_names, bands, total_range=(1.0, 45.0)):
    """
    Potenza per banda e canale in formato 'long'.   ##formato long solo una formalita per dire che le metto in riga e non in colonna nella tabella
    psd : (n_channels, n_f). Colonne: channel, band, abs_uV2, rel, log10_abs.
    """
    tot = integrate_band(psd, freqs, total_range[0], total_range[1] + 1e-6)
    rows = []                           ##lista vuota per accumulare le righe della futura tabella.
    for band, (lo, hi) in bands.items():        ##per ogni banda di freq
        bp = integrate_band(psd, freqs, lo, hi)     ##integra psd
        for i, ch in enumerate(ch_names):
            rows.append(dict(channel=ch, band=band, abs_uV2=bp[i],
                             rel=bp[i] / tot[i], log10_abs=np.log10(bp[i])))
    return pd.DataFrame(rows)


def aperiodic_fit(psd, freqs, frange=(2.0, 40.0)):
    """
    Componente aperiodica (1/f) dello spettro, per canale: log10 P(f) = offset - exponent * log10 f.
    psd: (n_channels, n_f). Fit robusto in 2 passi, come la modalita' 'fixed' di FOOOF/specparam:
      1. retta ai minimi quadrati su tutti i punti in frange (scala log-log);
      2. si tengono solo i punti sotto o sulla retta (i picchi oscillatori, es. alpha o la
         risposta alla stimolazione, stanno sopra) e si rifitta.
    exponent > 0: spettro che scende; piu' grande = piu' ripido (piu' potenza lenta rispetto a
    quella veloce; spesso letto come bilancio eccitazione/inibizione).
    Ritorna dict di array per canale: exponent, offset (log10 uV^2/Hz a 1 Hz), r2 (bonta' del
    fit sui punti tenuti), n_points.
    """
    m = (freqs >= frange[0]) & (freqs <= frange[1])
    x = np.log10(freqs[m])
    n_ch = psd.shape[0]
    out = {k: np.full(n_ch, np.nan) for k in ("exponent", "offset", "r2", "n_points")}
    for c in range(n_ch):
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.log10(psd[c, m])
        ok = np.isfinite(y)
        if ok.sum() < 5:
            continue
        b1, b0 = np.polyfit(x[ok], y[ok], 1)                    # passo 1
        keep = ok & (y - (b0 + b1 * x) <= 0)                     # passo 2: niente picchi
        if keep.sum() < 5:
            keep = ok
        b1, b0 = np.polyfit(x[keep], y[keep], 1)
        res = y[keep] - (b0 + b1 * x[keep])
        ss = ((y[keep] - y[keep].mean()) ** 2).sum()
        out["exponent"][c], out["offset"][c] = -b1, b0
        out["r2"][c] = 1 - (res ** 2).sum() / ss if ss > 0 else np.nan
        out["n_points"][c] = keep.sum()
    return out



"""
long:
O1 delta
O1 theta
O1 alpha
O1 beta



wide:
O1 | delta | theta | alpha | beta
"""