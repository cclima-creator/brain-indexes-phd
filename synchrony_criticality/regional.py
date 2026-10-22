# -*- coding: utf-8 -*-
"""
Regional synchrony: where on the scalp a difference in synchrony is located.

Channels are grouped into regions (e.g. frontal, central, occipital). For every pair-wise index
(PC, PLV, PPC, PLI, wPLI, dwPLI, iCOH: an N x N matrix per subject) two kinds of value are taken:
    within-region   r      mean over the pairs (i < j) with both channels in r
    between-region  (a, b) mean over the pairs with one channel in a and one in b
Global indices (Kuramoto GS and metastability) are not pair-wise, so their regional versions are:
    within-region   r      Kuramoto order parameter of the channels of r only:
                           R_r(t) = | mean_{j in r} exp(i phi_j(t)) |
    between-region  (a, b) synchrony of the two regional phases psi_a(t) = angle(Z_a(t)),
                           Z_a = mean_{j in a} exp(i phi_j(t)):
                           R_ab(t) = | (exp(i psi_a) + exp(i psi_b)) / 2 | = | cos((psi_a - psi_b) / 2) |
and GS = mean of R(t), M = std of R(t) inside each epoch, averaged over epochs (as kuramoto.py).

Caution: the random-phase level of R depends on the number of oscillators (sqrt(pi / 4N) for
N channels; 2/pi = 0.64 for two regional phases), so regional GS values are compared between
groups for the same region, never between regions of different size.
"""
import numpy as np

# old 10-20 name -> 10-10 name for the same electrode
ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


def canonical(name):
    """Upper-case 10-10 name (T3 -> T7, T4 -> T8, T5 -> P7, T6 -> P8)."""
    up = name.upper()
    return {k.upper(): v.upper() for k, v in ALIASES.items()}.get(up, up)


def region_indices(ch_names, regions):
    """
    regions: {region: [channel names]} (any 10-20 / 10-10 spelling, case-insensitive).
    Returns {region: [indices into ch_names]}; regions without channels are left out.
    """
    pos = {canonical(c): i for i, c in enumerate(ch_names)}
    out = {}
    for r, chans in regions.items():
        idx = sorted({pos[canonical(c)] for c in chans if canonical(c) in pos})
        if idx:
            out[r] = idx
    return out


def region_pairs(names):
    """Region pairs in a fixed order: (r, r) for within, then (a, b) with a before b."""
    return [(r, r) for r in names] + [(a, b) for k, a in enumerate(names) for b in names[k + 1:]]


def within_between(mat, idx):
    """
    Regional means of a symmetric pair-wise matrix (diagonal ignored).
    idx: output of region_indices. Returns {(a, b): value} for region_pairs(list(idx)); a within
    region with a single channel has no pairs and gives NaN.
    """
    mat = np.asarray(mat, float)
    out = {}
    for a, b in region_pairs(list(idx)):
        ia, ib = idx[a], idx[b]
        if a == b:
            i, j = np.triu_indices(len(ia), k=1)
            vals = mat[np.ix_(ia, ia)][i, j]
        else:
            vals = mat[np.ix_(ia, ib)].ravel()
        out[(a, b)] = float(np.nanmean(vals)) if vals.size else np.nan
    return out


def _gs_m(R):
    """GS and M of R(t) shaped (n_epochs, n_times): mean, and std within epoch averaged."""
    return float(R.mean()), float(R.std(axis=-1).mean())


def regional_kuramoto(phases, idx):
    """
    Within- and between-region Kuramoto. phases: (n_epochs, n_channels, n_times).
    Returns {(a, b): (gs, metastability)} for region_pairs(list(idx)).
    """
    u = np.exp(1j * phases)
    Z = {r: u[:, ix, :].mean(axis=1) for r, ix in idx.items()}           # (n_epochs, n_times)
    out = {}
    for a, b in region_pairs(list(idx)):
        if a == b:
            R = np.abs(Z[a]) if len(idx[a]) > 1 else None
        else:
            pa, pb = Z[a] / np.abs(Z[a]), Z[b] / np.abs(Z[b])           # regional phases
            R = np.abs((pa + pb) / 2)
        out[(a, b)] = _gs_m(R) if R is not None else (np.nan, np.nan)
    return out
