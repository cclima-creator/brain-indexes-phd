# -*- coding: utf-8 -*-
"""
codice generico per calcolar efase usando trasformata di hilbert per i dati grezzi. la fase viene poi usata da molti 
indici di sincronizzazione.
ci sono degli smoothing e filtri necessari per pulizia matematica

Instantaneous phase of band-limited EEG, common to every synchrony index.

    raw EEG (epochs x channels x samples)
        -> Butterworth band-pass, zero phase (scipy.signal.butter + sosfiltfilt)
        -> Hilbert transform -> analytic signal z(t) (scipy.signal.hilbert)
        -> instantaneous phase phi_j(t) = angle(z)
        -> edge trimming (filter and Hilbert edge effects)

The filter is built in second-order sections (sos): with the ba form a 4th-order band-pass at
1-4 Hz and 1024 Hz has poles too close to the unit circle and becomes numerically unstable.
sosfiltfilt is the same forward-backward filtering as filtfilt (zero phase, doubled order).

References: Le Van Quyen et al. 2001 (J Neurosci Methods), Lachaux et al. 1999 (Hum Brain Mapp).
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert


def bandpass_filter(data, sfreq, band, order=4):
    """
    Zero-phase Butterworth band-pass along the last axis.
    data: (..., n_times); band: (low, high) Hz. The upper edge is clipped below Nyquist.
    """
    lo, hi = band
    hi = min(hi, 0.99 * sfreq / 2)
    sos = butter(order, [lo, hi], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def analytic_signal(x):
    """Analytic signal z(t) = x(t) + i H[x](t) along the last axis."""
    return hilbert(x, axis=-1)


def instantaneous_phase(z):
    """Instantaneous phase in (-pi, pi] of an analytic signal."""
    return np.angle(z)


def trim_edges(x, sfreq, trim_s):
    """Removes trim_s seconds at both ends of the last axis (filter / Hilbert transients)."""
    n = int(round(trim_s * sfreq))
    if n == 0:
        return x
    if 2 * n >= x.shape[-1]:
        raise ValueError(f"edge trimming of {trim_s} s leaves no samples")
    return x[..., n:-n]


def band_analytic(data, sfreq, band, order=4, trim_s=1.0):
    """
    Band-passed analytic signal with trimmed edges. data: (n_epochs, n_channels, n_times),
    filtered epoch by epoch. Returns z: (n_epochs, n_channels, n_times_trimmed), complex.
    """
    z = analytic_signal(bandpass_filter(data, sfreq, band, order))
    return trim_edges(z, sfreq, trim_s)


def band_phases(data, sfreq, band, order=4, trim_s=1.0):
    """
    Full pipeline for one band. data: (n_epochs, n_channels, n_times), filtered epoch by epoch.
    Returns (phases, amplitude envelope), both (n_epochs, n_channels, n_times_trimmed).
    """
    z = band_analytic(data, sfreq, band, order, trim_s)
    return instantaneous_phase(z), np.abs(z)


def drop_flat_channels(data, tol=1e-12):
    """Indices of channels with non-zero variance in every epoch (flat channels have no phase)."""
    ok = (data.std(axis=-1) > tol).all(axis=0)
    return np.flatnonzero(ok)
