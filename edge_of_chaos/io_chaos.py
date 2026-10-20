# -*- coding: utf-8 -*-
"""
Input/output of the edge-of-chaos analysis: EEG files -> array in uV, per-subject cache (pickle),
Excel tables. No computation here.

Readable formats (mne): EEGLAB .set (+ .fdt), .fif (epochs or raw), .edf/.bdf/.vhdr (raw).
Continuous recordings are cut in fixed-length epochs (fixed_length_s).
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import mne

FALLBACK_DIR = Path.home() / "eeg_chaos_output"


# ================================================================== INPUT
def discover_files(folder, pattern="*.set"):
    """Sorted list of the files matching pattern in folder."""
    return sorted(Path(folder).glob(pattern))


def _read_raw(path):
    readers = {".set": mne.io.read_raw_eeglab, ".fif": mne.io.read_raw_fif,
               ".edf": mne.io.read_raw_edf, ".bdf": mne.io.read_raw_bdf,
               ".vhdr": mne.io.read_raw_brainvision}
    return readers[path.suffix.lower()](str(path), preload=True, verbose="ERROR")


def load_epochs(path, fixed_length_s=None, window=None, picks="eeg", sfreq=None):
    """
    mne.Epochs from a file. Files that already contain epochs are used as they are; continuous
    files are cut in epochs of fixed_length_s seconds (error if fixed_length_s is None).
    window=(tmin, tmax) crops every epoch (tmax excluded); sfreq resamples (None = keep).
    """
    path = Path(path)
    ep = None
    try:
        if path.suffix.lower() == ".set":
            ep = mne.io.read_epochs_eeglab(str(path), verbose="ERROR")
        elif path.suffix.lower() == ".fif" and "epo" in path.stem:
            ep = mne.read_epochs(str(path), preload=True, verbose="ERROR")
    except Exception:
        ep = None
    if ep is None:
        if fixed_length_s is None:
            raise ValueError(f"{path.name} is continuous: give fixed_length_s to cut it in epochs")
        raw = _read_raw(path)
        ep = mne.make_fixed_length_epochs(raw, duration=fixed_length_s, preload=True, verbose="ERROR")
    ep.load_data()
    ep.pick(picks)
    if window is not None:
        ep.crop(window[0], window[1], include_tmax=False)
    if sfreq is not None and not np.isclose(ep.info["sfreq"], sfreq):
        ep.resample(sfreq, verbose="ERROR")
    return ep


def pick_channels(ep, channels):
    """Keeps the channels in `channels` that exist in ep (case-insensitive); None = all."""
    if channels is None:
        return ep
    want = {c.upper() for c in channels}
    keep = [c for c in ep.ch_names if c.upper() in want]
    if not keep:
        raise ValueError(f"none of {channels} in {ep.ch_names}")
    return ep.pick(keep)


def epoch_labels(ep):
    """Event label of every epoch (list of str)."""
    inv = {v: k for k, v in ep.event_id.items()}
    return [inv.get(c, str(c)) for c in ep.events[:, 2]]


def load_subject(path, fixed_length_s=None, window=None, channels=None, sfreq=None, max_epochs=None):
    """
    One subject ready for chaos_indexes: dict(data (n_epochs, n_channels, n_times) in uV,
    sfreq, ch_names, labels). Flat channels (zero variance in every epoch) are removed.
    """
    ep = pick_channels(load_epochs(path, fixed_length_s, window, sfreq=sfreq), channels)
    data = ep.get_data() * 1e6
    labels = epoch_labels(ep)
    if max_epochs is not None:
        data, labels = data[:max_epochs], labels[:max_epochs]
    flat = np.all(data.std(axis=-1) == 0, axis=0)
    if flat.any():
        print(f"    flat channels removed: {[c for c, f in zip(ep.ch_names, flat) if f]}")
    return dict(data=data[:, ~flat], sfreq=float(ep.info["sfreq"]),
                ch_names=[c for c, f in zip(ep.ch_names, flat) if not f], labels=labels)


# ================================================================== OUTPUT
def check_output_dir(folder, fallback=FALLBACK_DIR):
    """A writable folder (folder, otherwise fallback), checked BEFORE the analysis."""
    for f in (Path(folder), Path(fallback)):
        try:
            f.mkdir(parents=True, exist_ok=True)
            t = f / "_write_test.tmp"
            t.write_bytes(b"test")
            t.unlink()
            return f
        except OSError as e:
            print(f"WARNING: cannot write in {f}\n  -> {e}")
    raise SystemExit("No writable folder: set the output folder to a valid path.")


def save_results(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_results(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_cached_subject(path, params):
    """Cached subject if it exists and was computed with the same parameters, else None."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        res = load_results(path)
    except Exception:
        return None
    return res if res.get("params") == params else None


def save_tables_excel(tables, path, sheet_order=()):
    """One .xlsx, one sheet per table (dict name -> DataFrame), header row frozen."""
    from openpyxl.utils import get_column_letter
    names = [n for n in sheet_order if n in tables] + [n for n in tables if n not in sheet_order]
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name in names:
            df = tables[name]
            sheet = name[:31]
            df.to_excel(xw, sheet_name=sheet, index=False)
            ws = xw.sheets[sheet]
            ws.freeze_panes = "A2"
            for i, col in enumerate(df.columns, start=1):
                widths = [len(str(col))] + [len(str(v)) for v in df[col].head(200)]
                ws.column_dimensions[get_column_letter(i)].width = min(max(widths) + 2, 40)
