# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 14:10:44 2026

@author: Chiara Clima
"""

"""Rapporto segnale/rumore (dB). Funzioni pure su PSD (n_channels, n_f)."""
import numpy as np


def snr_broadband(psd, freqs, signal_range=(1.0, 45.0), noise_range=(55.0, 70.0)):     ##noise postato tra 55 e 70 ma capire bene
    """
    SNR a banda larga: potenza media nel range di segnale / potenza media nel range di
    rumore (proxy: alta frequenza). Ritorna dB per canale (NaN se un range e' vuoto).
    """
    s = (freqs >= signal_range[0]) & (freqs <= signal_range[1])
    n = (freqs >= noise_range[0]) & (freqs <= noise_range[1])
    if s.sum() == 0 or n.sum() == 0:
        return np.full(psd.shape[0], np.nan)
    return 10 * np.log10(psd[:, s].mean(1) / psd[:, n].mean(1))


def snr_at_frequency(psd, freqs, f0, gap=0.3, width=1.5, fmax=None):       ##quanto e' forte una freq rispetto a quelle vicine
                                                                ##non mi convince per niente
                                                                ##quanto una specifica frequenza emerge rispetto al proprio background locale.
    """
    SNR a banda stretta in f0: potenza al bin di f0 / media dei bin vicini
    (distanti da `gap` a `width` Hz, entrambi i lati). Ritorna dB per canale.
    fmax: se f0 (o i bin vicini) supera fmax ritorna NaN (es. 45 Hz: sopra e' rumore).
    """
    n_ch = psd.shape[0]
    if fmax is not None and np.isfinite(f0) and f0 + width > fmax:
        return np.full(n_ch, np.nan)
    if not np.isfinite(f0) or f0 > freqs.max():
        return np.full(n_ch, np.nan)
    df = freqs[1] - freqs[0]
    i0 = np.argmin(np.abs(freqs - f0))
    if abs(freqs[i0] - f0) > df:
        return np.full(n_ch, np.nan)
    d = np.abs(freqs - f0)
    nb = (d >= gap) & (d <= width)
    if nb.sum() < 4:
        return np.full(n_ch, np.nan)
    return 10 * np.log10(psd[:, i0] / psd[:, nb].mean(1))


def snr_narrowband(psd, freqs, f0, n_harmonics=3, gap=0.3, width=1.5, fmax=None):    ##ripete snr_at_frequency() per più armoniche della stessa frequenza fondamentale
    """Dizionario {armonica: SNR dB per canale} per f0, 2*f0, ..."""
    return {h: snr_at_frequency(psd, freqs, f0 * h, gap, width, fmax)
            for h in range(1, n_harmonics + 1)}

##la risposta può manifestarsi alla frequenza di stimolazione e alle sue armoniche.