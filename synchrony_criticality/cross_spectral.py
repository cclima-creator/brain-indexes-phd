# -*- coding: utf-8 -*-
"""
Lag-insensitive coupling from cross-spectra averaged ACROSS EPOCHS, as in the original papers
(and in mne_connectivity.spectral_connectivity_epochs), instead of over the Hilbert samples of one
epoch (lag_insensitive.py, kept for comparison).

    X_k(f)   multitaper Fourier coefficients of every epoch (DPSS tapers), channel i
    S_ij(f)  cross-spectrum of one epoch = mean over tapers of X_i X_j*
    sums over the E epochs:
      iCOH  = Im( <S_ij> / sqrt(<S_ii> <S_jj>) )                                     Nolte et al. 2004
      PLI   = | <sign Im S_ij> |                                                     Stam et al. 2007
      wPLI  = | <Im S_ij> | / <|Im S_ij|>                                             Vinck et al. 2011
      dwPLI = ((sum Im)^2 - sum Im^2) / ((sum |Im|)^2 - sum Im^2)                     Vinck et al. 2011
then the mean over the frequency bins of the band (iCOH: the complex coherency is averaged over
the bins first, then Im and |.|, so the sign of the lag does not cancel across pairs).
Epochs are the independent "trials" of the estimators: the bias of PLI, wPLI and |iCOH| falls with
the number of epochs (dwPLI is unbiased), so the number of epochs must be the same in the compared
conditions (descriptive_analysis.segments.equal_subsets).
"""
import numpy as np
from scipy.signal.windows import dpss


def epoch_spectra(data, sfreq, nw=2.0, fmax=None):
    """
    Multitaper Fourier coefficients. data: (n_epochs, n_channels, n_times).
    nw = time-half-bandwidth product (2 -> 3 tapers, frequency smoothing +-nw / T Hz).
    Returns (X (n_epochs, n_tapers, n_channels, n_freqs) complex64, freqs).
    """
    n_ep, n_ch, n_t = data.shape
    k = max(1, int(2 * nw - 1))
    tapers = dpss(n_t, nw, k)                                     # (k, n_t)
    x = data - data.mean(-1, keepdims=True)
    freqs = np.fft.rfftfreq(n_t, 1 / sfreq)
    keep = freqs <= (fmax if fmax else sfreq / 2)
    X = np.empty((n_ep, k, n_ch, keep.sum()), np.complex64)
    for t in range(k):
        X[:, t] = np.fft.rfft(x * tapers[t], axis=-1)[..., keep]
    return X, freqs[keep]


def lag_insensitive_across_epochs(X, freqs, band, idx=None):
    """
    PLI, wPLI, dwPLI, iCOH (n_channels, n_channels, diagonal 0) of one band from epoch_spectra.
    idx = epochs to use (default all).
    """
    sel = (freqs >= band[0]) & (freqs <= band[1])
    Xb = X[:, :, :, sel] if idx is None else X[np.asarray(idx)][:, :, :, sel]
    n_ep, n_tap, n_ch, n_f = Xb.shape
    s_im = np.zeros((n_ch, n_ch, n_f))
    s_abs = np.zeros_like(s_im)
    s_sq = np.zeros_like(s_im)
    s_sign = np.zeros_like(s_im)
    s_cross = np.zeros((n_ch, n_ch, n_f), complex)
    for e in range(n_ep):
        S = np.einsum("tif,tjf->ijf", Xb[e], np.conj(Xb[e])) / n_tap      # (n_ch, n_ch, n_f)
        im = S.imag
        s_im += im
        s_abs += np.abs(im)
        s_sq += im ** 2
        s_sign += np.sign(im)
        s_cross += S
    with np.errstate(invalid="ignore", divide="ignore"):
        pli = np.abs(s_sign / n_ep)
        wpli = np.where(s_abs > 0, np.abs(s_im) / s_abs, 0.0)
        den = s_abs ** 2 - s_sq
        dwpli = np.where(den > 0, (s_im ** 2 - s_sq) / den, 0.0)
        auto = np.real(np.einsum("iif->if", s_cross))                       # (n_ch, n_f)
        coh = (s_cross / np.sqrt(auto[:, None, :] * auto[None, :, :])).mean(-1)
    out = dict(pli=pli.mean(-1), wpli=wpli.mean(-1), dwpli=dwpli.mean(-1), icoh=np.abs(coh.imag))
    for m in out.values():
        np.fill_diagonal(m, 0.0)
    return out
