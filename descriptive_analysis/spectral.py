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


"""Spettro di potenza e potenza per banda. Funzioni pure su array numpy."""
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



"""
long:
O1 delta
O1 theta
O1 alpha
O1 beta



wide:
O1 | delta | theta | alpha | beta
"""