# -*- coding: utf-8 -*-
"""
Input of the Ayahuasca database: which subjects have which condition, and the epochs of one
subject in one condition (with the central 27-33 min window cut from the 20-40 slot).
"""
import os
import re
from pathlib import Path

import numpy as np
import mne

from psychedelics import config_aya as cfg


def subject_id(path):
    """'P01_AYA_REST_...' -> 'P01'."""
    return Path(path).name.split("_")[0]


def _real_file(cond, sid):
    folder, pattern, _ = cfg.CONDITIONS[cond]
    files = sorted(Path(folder).glob(pattern.format(sid=sid)))
    return files[0] if files else None


def all_subjects():
    """IDs found in any condition folder, minus EXCLUDED_SUBJECTS."""
    ids = set()
    for folder, pattern, _ in cfg.CONDITIONS.values():
        ids |= {subject_id(f) for f in Path(folder).glob(pattern.format(sid="*"))}
    ids -= set(cfg.EXCLUDED_SUBJECTS)
    return sorted(ids & set(cfg.ONLY_SUBJECTS) if cfg.ONLY_SUBJECTS else ids)


_ASSIGN = {}


def rest_assignment(virtual="rest_pre"):
    """
    {subject: real condition} of a virtual condition. Subjects with both real conditions are split
    at random (seed REST_SEED) into two halves (the extra one, if odd, to the first condition);
    subjects with only one use it.
    """
    if virtual in _ASSIGN:
        return _ASSIGN[virtual]
    a, b = cfg.VIRTUAL[virtual]
    subs = all_subjects()
    both = [s for s in subs if _real_file(a, s) and _real_file(b, s)]
    rng = np.random.default_rng(cfg.REST_SEED)
    order = list(rng.permutation(both))
    half = (len(both) + 1) // 2
    out = {s: (a if k < half else b) for k, s in enumerate(order)}
    for s in subs:
        if s not in out:
            out[s] = a if _real_file(a, s) else b if _real_file(b, s) else None
    _ASSIGN[virtual] = {s: c for s, c in sorted(out.items()) if c}
    return _ASSIGN[virtual]


def resolve(cond, sid):
    """Real condition of (cond, sid): cond itself, or the drawn one for a virtual condition."""
    return rest_assignment(cond).get(sid) if cond in cfg.VIRTUAL else cond


def condition_file(cond, sid):
    """The .set file of subject sid in condition cond (virtual conditions resolved), or None."""
    real = resolve(cond, sid)
    return _real_file(real, sid) if real else None


def analysis_subjects(analysis):
    """Subjects that have a file for every condition of the analysis (paired design)."""
    conds = cfg.ANALYSES[analysis]
    return [s for s in all_subjects() if all(condition_file(c, s) is not None for c in conds)]


def comparison_subjects(comp):
    """Subjects with every condition of both arms of a drug x time comparison."""
    aya, pla, _ = cfg.COMPARISONS[comp]
    return [s for s in all_subjects() if all(condition_file(c, s) is not None for c in aya + pla)]


def epoch_end_s(labels):
    """Labels '1205s' -> 1205.0 (end of the epoch in s from the start of the hour); NaN if not numeric."""
    out = []
    for l in labels:
        m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*s\s*", str(l))
        out.append(float(m.group(1)) if m else np.nan)
    return np.array(out)


def load_condition(cond, sid):
    """
    mne.Epochs of one subject in one condition (EEG channels only). For a condition with a window
    (t0, t1) only the epochs with end time in (t0, t1] are kept (labels '<t>s' of the DURING files).
    """
    cond = resolve(cond, sid)
    path = condition_file(cond, sid)
    if path is None:
        raise FileNotFoundError(f"no file for {sid} in {cond}")
    ep = mne.io.read_epochs_eeglab(str(path), verbose="ERROR")
    ep.load_data()
    ep.pick("eeg")
    window = cfg.CONDITIONS[cond][2]
    if window is not None:
        inv = {v: k for k, v in ep.event_id.items()}
        t = epoch_end_s([inv.get(c, str(c)) for c in ep.events[:, 2]])
        keep = np.where((t > window[0]) & (t <= window[1]))[0]
        if not keep.size:
            raise ValueError(f"{sid} {cond}: no epoch in {window} s")
        ep = ep[keep]
    if cfg.MAX_EPOCHS:
        ep = ep[:cfg.MAX_EPOCHS]
    return ep


def epoch_labels(ep):
    inv = {v: k for k, v in ep.event_id.items()}
    return [inv.get(c, str(c)) for c in ep.events[:, 2]]


def contiguous_runs(labels, epoch_s=5.0):
    """
    Runs of consecutive epochs (lists of epoch indices) that were contiguous in the recording.
    DURING files: label "<t>s" = end time, so a run breaks where the end times are not 5 s apart
    (a removed epoch). PRE / POST files (labels "EVT_5S") do not say where the removed epochs
    were (66-72 of the 72 epochs are left): they are one run, with the few junctions unknown.
    """
    t = epoch_end_s(labels)
    if not np.isfinite(t).all():
        return [np.arange(len(labels))]
    cut = np.flatnonzero(np.abs(np.diff(t) - epoch_s) > 1e-6) + 1
    return [r for r in np.split(np.arange(len(labels)), cut) if r.size]


def load_data(cond, sid):
    """
    dict(data (n_ep, n_ch, n_t) in uV, sfreq, ch_names, labels, file, runs, n_equal) for every module.
    runs = contiguous_runs (filtering and Hilbert transform are done on each run as one signal);
    n_equal = number of epochs the N-dependent indices are computed on (n_equal()).
    """
    ep = load_condition(cond, sid)
    labels = epoch_labels(ep)
    return dict(data=ep.get_data() * 1e6, sfreq=float(ep.info["sfreq"]), ch_names=list(ep.ch_names),
                labels=labels, file=str(condition_file(cond, sid)), runs=contiguous_runs(labels),
                n_equal=n_equal(cond, sid))


# ================================================================== EQUAL NUMBER OF EPOCHS
# Counts depend on the epochs available (6-min conditions 41-72 epochs, 20-min slots 110-240):
# indices that grow or shrink with the amount of data are computed on the same number of epochs
# in every condition of a subject (the minimum over the conditions of the same duration family).
FAMILIES = {"6min": ["pre", "during_27-33", "post", "pla_pre", "pla_during_27-33", "pla_post"],
            "20min": ["slot_0-20", "slot_20-40", "slot_40-60",
                      "pla_slot_0-20", "pla_slot_20-40", "pla_slot_40-60"]}
_COUNTS = None


def _counts_file():
    return cfg.OUT_DIR / "cache" / "epoch_counts.json"


def n_epochs(cond, sid):
    """Number of epochs of (cond, sid) after the time window and MAX_EPOCHS (cached on disk)."""
    global _COUNTS
    import json
    real = resolve(cond, sid)
    path = condition_file(real, sid)
    if path is None:
        return None
    st = path.stat()
    key = f"{real}|{path}|{st.st_size}|{round(st.st_mtime)}|{cfg.CONDITIONS[real][2]}|{cfg.MAX_EPOCHS}"
    if _COUNTS is None:
        try:
            _COUNTS = json.loads(_counts_file().read_text())
        except Exception:
            _COUNTS = {}
    if key not in _COUNTS:
        _COUNTS[key] = len(load_condition(real, sid))
        try:                                # runner.run_all fills the file before the parallel processes
            _counts_file().parent.mkdir(parents=True, exist_ok=True)
            tmp = _counts_file().with_name(f"epoch_counts.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(_COUNTS, indent=0))
            tmp.replace(_counts_file())
        except OSError:
            pass
    return _COUNTS[key]


def n_equal(cond, sid):
    """Minimum number of epochs of subject sid over the conditions of the family of cond."""
    real = resolve(cond, sid)
    fam = next(c for c in FAMILIES.values() if real in c)
    n = [n_epochs(c, sid) for c in fam if condition_file(c, sid) is not None]
    return int(min(n))
