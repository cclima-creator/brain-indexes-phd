# -*- coding: utf-8 -*-
"""
Epoched data that were (partly) continuous recordings, and indices that depend on the amount of data.

1) Filtering and Hilbert transform on the continuous signal.
   An epoch of 5 s filtered on its own has edge transients (a 1 Hz FIR high-pass is ~3.3 s long,
   a 1-4 Hz Butterworth rings for ~1.3 s) and the Hilbert transform is circular on the epoch.
   apply_on_runs() concatenates every run of contiguous epochs into one signal, pads it by
   reflection, applies the operation (filter, filter + Hilbert, ...) and cuts it back into the
   epochs: only the two ends of a run can still have a transient, attenuated by the padding.
   Separate trials (e.g. one epoch per photic stimulation frequency) are runs of one epoch.

2) Same number of epochs for indices that depend on it.
   Counts, unique-pattern repertoires, PLV, estimates of an SD over time, entropy production and
   covariance widths are biased by the amount of data. equal_subsets() gives subsets of n_target
   epochs; the index is computed on each subset and averaged (mean_over_subsets()).
"""
import numpy as np


def runs_of_single_epochs(n_epochs):
    """Every epoch is a separate trial."""
    return [np.array([k]) for k in range(n_epochs)]


def apply_on_runs(data, runs, fn, sfreq, pad_s=2.0):
    """
    fn applied to every run of contiguous epochs as one signal.
    data: (n_epochs, n_channels, n_times); runs: list of epoch-index arrays (together all epochs);
    fn: (n_channels, n) -> (n_channels, n) along the last axis (real or complex).
    Each run is padded at both ends by pad_s seconds of odd reflection (at most the run length - 1).
    Returns an array shaped like data, with the dtype of fn's output.
    """
    data = np.asarray(data)
    n_ep, n_ch, n_t = data.shape
    out = None
    for r in runs:
        r = np.asarray(r)
        x = data[r].transpose(1, 0, 2).reshape(n_ch, -1)           # (n_channels, len(r) * n_t)
        p = int(min(round(pad_s * sfreq), x.shape[-1] - 1))
        xp = np.pad(x, ((0, 0), (p, p)), mode="reflect", reflect_type="odd") if p > 0 else x
        y = np.asarray(fn(xp))[..., p:p + x.shape[-1]]
        if out is None:
            out = np.empty(data.shape, dtype=y.dtype)
        out[r] = y.reshape(n_ch, len(r), n_t).transpose(1, 0, 2)
    return out


def run_signals(x, runs):
    """(n_epochs, n_channels, n_times) -> one (n_channels, n) signal per run (contiguous epochs joined)."""
    return [np.concatenate(list(x[np.asarray(r)]), axis=-1) for r in runs]


def subset_runs(runs, idx):
    """The runs restricted to the epochs in idx, renumbered as positions in idx."""
    pos = {e: k for k, e in enumerate(idx)}
    out = [np.array([pos[e] for e in r if e in pos]) for r in runs]
    return [r for r in out if r.size]


def equal_subsets(n_epochs, n_target, n_draws=5, mode="blocks", seed=0):
    """
    Subsets of n_target epochs (all epochs if n_epochs <= n_target).
    mode "blocks": n_draws blocks of n_target consecutive epochs with evenly spaced starts (keeps the
                   contiguity of continuous recordings);
         "random": n_draws random subsets (separate trials).
    """
    if n_target is None or n_epochs <= n_target:
        return [np.arange(n_epochs)]
    if mode == "blocks":
        starts = np.unique(np.round(np.linspace(0, n_epochs - n_target, n_draws)).astype(int))
        return [np.arange(s, s + n_target) for s in starts]
    rng = np.random.default_rng(seed)
    return [np.sort(rng.choice(n_epochs, n_target, replace=False)) for _ in range(n_draws)]


def mean_over_subsets(subsets, f):
    """f(idx) -> dict of numbers (or arrays) for every subset; returns the mean of each entry (NaN-aware)."""
    import warnings
    res = [f(idx) for idx in subsets]
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN entries stay NaN
        for k in res[0]:
            out[k] = np.nanmean(np.array([r[k] for r in res], dtype=float), axis=0)
    return out
