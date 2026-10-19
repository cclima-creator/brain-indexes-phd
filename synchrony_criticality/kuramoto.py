# -*- coding: utf-8 -*-
"""
Kuramoto order parameter, global synchrony and metastability.

    R(t) = | 1/N sum_j exp(i phi_j(t)) |                     (eq. 3.15)
    GS   = mean_t R(t)                                       (eq. 3.16)
    M    = std_t  R(t)                                       (eq. 3.17)

R(t) = 1: all N channels share the same phase; R(t) ~ 0: phases spread on the circle.
GS measures how synchronized the network is on average, M how much this synchronization
fluctuates in time (metastable dynamics: neither stuck in synchrony nor in disorder).

With epochs, R(t) is computed inside each epoch (no jumps across epoch boundaries):
  - GS is the mean over all samples of all epochs (= mean of the per-epoch GS, equal lengths)
  - M is the std within each epoch, averaged over epochs; metastability_pooled is the std over
    all samples together (it also contains the differences of GS between epochs/conditions).

Finite-size bias: with N independent uniform phases R is Rayleigh distributed with
E[R] = sqrt(pi / (4N)) (0.19 for N = 21), so GS is never 0 for real data. Circular-shift
surrogates (each channel shifted by a random lag inside the epoch) keep the spectrum of every
channel and destroy only the phase relations between channels: GS / GS_surrogate > 1 means
synchrony above what the channels' own dynamics would give.

References: Kuramoto 1984; Acebron et al. 2005 (Rev Mod Phys); Arenas et al. 2008 (Phys Rep);
Shanahan 2010 (Chaos); Cabral et al. 2011; Deco et al. 2017 (Sci Rep).
"""
import numpy as np


def kuramoto_order_parameter(phases):
    """
    phases: (n_channels, n_times), or (n_epochs, n_channels, n_times).
    Returns R(t): (n_times,) or (n_epochs, n_times). Mean over the channel axis.
    """
    return np.abs(np.mean(np.exp(1j * phases), axis=-2))


def global_synchrony(R_t):
    """GS = time average of R(t) (over every axis)."""
    return float(np.mean(R_t))


def metastability(R_t):
    """M = standard deviation of R(t) over time. R_t: (n_times,) -> float, (n_epochs, n_times) -> per epoch."""
    return np.std(R_t, axis=-1)


def random_phase_R(n_channels):
    """Expected R for n_channels independent uniform phases (Rayleigh mean, sqrt(pi / 4N))."""
    return float(np.sqrt(np.pi / (4 * n_channels)))


def circular_shift_surrogate(phases, rng):
    """Each channel of each epoch circularly shifted by an independent random lag (phases or phasors)."""
    n_ep, n_ch, n_t = phases.shape
    shifts = rng.integers(0, n_t, size=(n_ep, n_ch))
    out = np.empty_like(phases)
    for e in range(n_ep):
        for c in range(n_ch):
            out[e, c] = np.roll(phases[e, c], -shifts[e, c])
    return out


def kuramoto_epochs(phases, n_surrogates=0, seed=0, hist_bins=50):
    """
    All Kuramoto measures for one subject in one band. phases: (n_epochs, n_channels, n_times).
    Returns (summary dict, per-epoch dict, R(t) of every epoch, histogram of R).
    """
    u = np.exp(1j * phases).astype(np.complex64)              # unit phasors, computed once
    R = np.abs(u.mean(axis=-2)).astype(float)                 # R(t), (n_epochs, n_times)
    gs_ep, m_ep = R.mean(axis=-1), metastability(R)
    n_ch = phases.shape[1]
    out = dict(gs=global_synchrony(R), metastability=float(m_ep.mean()),
               metastability_pooled=float(np.std(R)), gs_random=random_phase_R(n_ch),
               R_median=float(np.median(R)), R_q05=float(np.quantile(R, 0.05)),
               R_q95=float(np.quantile(R, 0.95)),
               # coefficient of variation: metastability relative to the synchrony level
               metastability_cv=float(m_ep.mean() / R.mean()) if R.mean() > 0 else np.nan)
    if n_surrogates:
        rng = np.random.default_rng(seed)
        gs_s, m_s = [], []
        for _ in range(n_surrogates):
            Rs = np.abs(circular_shift_surrogate(u, rng).mean(axis=-2))   # shifting u = shifting phi
            gs_s.append(Rs.mean())
            m_s.append(metastability(Rs).mean())
        gs_s, m_s = np.array(gs_s), np.array(m_s)
        out.update(gs_surrogate=float(gs_s.mean()), metastability_surrogate=float(m_s.mean()),
                   gs_norm=out["gs"] / gs_s.mean(), metastability_norm=out["metastability"] / m_s.mean(),
                   # empirical p: fraction of surrogates at least as synchronized as the data
                   gs_p_surrogate=float((1 + (gs_s >= out["gs"]).sum()) / (1 + n_surrogates)))
    edges = np.linspace(0, 1, hist_bins + 1)
    hist = np.histogram(R, bins=edges, density=True)[0]
    return out, dict(gs=gs_ep, metastability=m_ep), R, dict(edges=edges, density=hist)
