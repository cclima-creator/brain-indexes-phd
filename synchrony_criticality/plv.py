# -*- coding: utf-8 -*-
"""
Phase Locking Value across epochs (Lachaux et al. 1999).

    PLV_ij(t) = | 1/M sum_k exp(i dphi_ij,k(t)) |            (eq. 3.14, M = number of epochs)

At every time t the phase difference is averaged over the M epochs, then |.| is taken; the
subject value is the mean of PLV_ij(t) over t. It measures how reproducible the phase
relation between i and j is across epochs at the same latency from the epoch onset, so it makes
sense when the epochs are time-locked to an event (here: start of the photic stimulation).
Averaging over epochs AND time together before |.| (as in a single np.mean over both axes)
would give a different quantity: the phase coherence of the epochs concatenated.

Bias: with M independent epochs E[PLV^2] = 1/M, so subjects with fewer epochs get a higher PLV
for the same coupling. The pairwise phase consistency (PPC, Vinck et al. 2010) is unbiased:
    PPC = (M |mean_k exp(i dphi_k)|^2 - 1) / (M - 1)   (expected value 0 without locking)
Use the PPC for group comparisons when the number of epochs differs between subjects.

References: Lachaux et al. 1999 (Hum Brain Mapp); Vinck et al. 2010 (NeuroImage);
mne-connectivity spectral_connectivity_epochs(method="plv" / "ppc") for the frequency-domain version.
"""
import numpy as np


def _locking(phases_epochs, chunk=1024):
    """|mean over epochs of exp(i dphi_ij(t))| for every pair and time: (n_ch, n_ch, n_times)."""
    n_ep, n_ch, n_t = phases_epochs.shape
    u = np.exp(1j * phases_epochs)
    out = np.empty((n_ch, n_ch, n_t))
    for s in range(0, n_t, chunk):                      # in time chunks: bounded memory
        sl = slice(s, s + chunk)
        out[:, :, sl] = np.abs(np.einsum("eit,ejt->ijt", u[:, :, sl], u[:, :, sl].conj())) / n_ep
    return out


def plv_time(phases_epochs):
    """Time-resolved PLV_ij(t). phases_epochs: (n_epochs, n_channels, n_times) -> (n_ch, n_ch, n_times)."""
    return _locking(phases_epochs)


def plv_matrix(phases_epochs):
    """PLV_ij averaged over time: (n_channels, n_channels), diagonal = 1."""
    return _locking(phases_epochs).mean(-1)


def plv_loop(phases_epochs):
    """Reference double-loop version of plv_matrix (slow, used in the tests)."""
    n_ep, n_ch, n_t = phases_epochs.shape
    PLV = np.eye(n_ch)
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            d_phi = phases_epochs[:, i, :] - phases_epochs[:, j, :]
            PLV[i, j] = PLV[j, i] = np.abs(np.mean(np.exp(1j * d_phi), axis=0)).mean()
    return PLV


def ppc_from_plv(plv_t, n_epochs):
    """Pairwise phase consistency from PLV(t) (same shape), averaged over time if plv_t has a time axis."""
    return (n_epochs * plv_t ** 2 - 1) / (n_epochs - 1)


def plv_summary(phases_epochs):
    """
    PLV and PPC of one subject in one band.
    Returns dict(plv=(n_ch, n_ch), ppc=(n_ch, n_ch), plv_t=mean over pairs of PLV(t), n_epochs).
    """
    n_ep = phases_epochs.shape[0]
    if n_ep < 2:
        raise ValueError("PLV across epochs needs at least 2 epochs")
    L = _locking(phases_epochs)
    i, j = np.triu_indices(L.shape[0], k=1)
    return dict(plv=L.mean(-1), ppc=ppc_from_plv(L, n_ep).mean(-1),
                plv_t=L[i, j].mean(0), n_epochs=n_ep)
