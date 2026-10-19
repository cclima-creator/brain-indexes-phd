# -*- coding: utf-8 -*-
"""
Phase coherence between pairs of channels, averaged over TIME (one value per epoch).

    dphi_ij(t) = phi_i(t) - phi_j(t)                         (eq. 3.12)
    PC_ij      = | < exp(i dphi_ij(t)) >_t |                 (eq. 3.13)

PC_ij = 1: constant phase difference during the whole epoch; ~0: phase difference drifting
uniformly. Also called mean phase coherence (Mormann et al. 2000) or "time PLV". It differs from
the PLV of plv.py, where the average is taken over epochs at fixed time (Lachaux et al. 1999).

Vectorized: with u_j(t) = exp(i phi_j(t)), exp(i dphi_ij) = u_i conj(u_j), so the whole matrix
is |U U^H| / T (U: channels x times). Same values as the double loop over pairs.

Zero-lag coupling (volume conduction, common reference, few back-projected ICA components) also
gives PC close to 1: compare groups on the same montage and processing, and keep in mind that
PC alone cannot separate true coupling from shared sources.

References: Mormann et al. 2000 (Physica D); Lachaux et al. 1999; Rosenblum et al. 2001.
"""
import numpy as np


def phase_coherence_matrix(phases):
    """
    phases: (n_channels, n_times). Returns PC: (n_channels, n_channels), symmetric, diagonal = 1.
    """
    u = np.exp(1j * phases)
    return np.abs(u @ u.conj().T) / phases.shape[-1]


def phase_coherence_loop(phases):
    """Reference double-loop version of phase_coherence_matrix (slow, used in the tests)."""
    n_ch = phases.shape[0]
    PC = np.eye(n_ch)
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            d_phi = phases[i] - phases[j]
            PC[i, j] = PC[j, i] = np.abs(np.mean(np.exp(1j * d_phi)))
    return PC


def phase_coherence_epochs(phases):
    """
    PC of every epoch. phases: (n_epochs, n_channels, n_times).
    Returns (n_epochs, n_channels, n_channels).
    """
    u = np.exp(1j * phases)
    return np.abs(np.einsum("eit,ejt->eij", u, u.conj())) / phases.shape[-1]


def upper(mat):
    """Upper-triangle values (i < j) of a square matrix or of a stack (..., n, n)."""
    i, j = np.triu_indices(mat.shape[-1], k=1)
    return mat[..., i, j]


def node_strength(mat):
    """Mean coupling of each channel with all the others (diagonal excluded)."""
    n = mat.shape[-1]
    return (mat.sum(-1) - np.diagonal(mat, axis1=-2, axis2=-1)) / (n - 1)
