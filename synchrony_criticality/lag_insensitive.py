# -*- coding: utf-8 -*-
"""
Phase synchrony indices insensitive to zero-lag coupling.

Any instantaneous mixing of sources (volume conduction, common reference, channels rebuilt from a
few ICA components) adds the same signal to two channels with phase difference 0 or pi. It only
changes the REAL part of the cross-spectrum S_ij(t) = z_i(t) conj(z_j(t)); a true interaction with
a time lag also produces an IMAGINARY part. The indices below use only Im S_ij, so they are not
inflated by zero-lag mixing (they are, in turn, blind to true zero-lag coupling).

    PLI_ij   = | < sign(Im S_ij) > |                                   (Stam et al. 2007)
    wPLI_ij  = | < Im S_ij > | / < |Im S_ij| >                          (Vinck et al. 2011)
    dwPLI_ij = ((sum Im S)^2 - sum (Im S)^2) / ((sum |Im S|)^2 - sum (Im S)^2)
                                                    debiased wPLI^2 (Vinck et al. 2011)
    iCOH_ij  = | Im < S_ij > | / sqrt(< |z_i|^2 > < |z_j|^2 >)          (Nolte et al. 2004)

z is the analytic signal of the band-passed EEG (phase.py); < . > is the average over the samples
of one epoch (Hilbert, time-domain version). Values per epoch, then averaged over epochs.
wPLI weights each sample by |Im S|, so samples with phase difference near 0 / pi (dominated by noise
and mixing) count less than in the PLI; the debiased version removes the positive bias of finite
samples (expected value 0 without lagged coupling) and can be slightly negative.

References: Nolte et al. 2004 (Clin Neurophysiol); Stam et al. 2007 (Hum Brain Mapp);
Vinck et al. 2011 (NeuroImage); mne-connectivity methods "pli", "wpli", "wpli2_debiased", "imcoh".
"""
import numpy as np


def _imag_cross(z):
    """Im S_ij(t) for every pair: z (n_channels, n_times) complex -> (n_ch, n_ch, n_times)."""
    x, y = z.real, z.imag
    # Im(z_i conj(z_j)) = y_i x_j - x_i y_j
    return y[:, None, :] * x[None, :, :] - x[:, None, :] * y[None, :, :]


def lag_insensitive_matrices(z):
    """
    PLI, wPLI, debiased wPLI and imaginary coherence of one epoch.
    z: analytic signal (n_channels, n_times). Returns dict of (n_ch, n_ch) matrices (diagonal 0).
    """
    im = _imag_cross(z)
    s, s_abs, s_sq = im.sum(-1), np.abs(im).sum(-1), (im ** 2).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        wpli = np.where(s_abs > 0, np.abs(s) / s_abs, 0.0)
        den = s_abs ** 2 - s_sq
        dwpli = np.where(den > 0, (s ** 2 - s_sq) / den, 0.0)
    pli = np.abs(np.sign(im).mean(-1))
    power = (np.abs(z) ** 2).mean(-1)
    icoh = np.abs(s / z.shape[-1]) / np.sqrt(np.outer(power, power))
    out = dict(pli=pli, wpli=wpli, dwpli=dwpli, icoh=icoh)
    for m in out.values():
        np.fill_diagonal(m, 0.0)
    return out


def lag_insensitive_epochs(z):
    """
    Same indices for every epoch, averaged over epochs. z: (n_epochs, n_channels, n_times).
    Returns (dict of mean matrices, dict of per-epoch mean over pairs).
    """
    per = [lag_insensitive_matrices(zk) for zk in z]
    keys = per[0].keys()
    iu = np.triu_indices(z.shape[1], k=1)
    mats = {k: np.mean([p[k] for p in per], axis=0) for k in keys}
    ep = {k: np.array([p[k][iu].mean() for p in per]) for k in keys}
    return mats, ep
