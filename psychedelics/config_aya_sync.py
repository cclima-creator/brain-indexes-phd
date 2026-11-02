# -*- coding: utf-8 -*-
"""
Synchrony settings for the Ayahuasca database. Same names as photosensitive/config_ips_sync.py,
because the per-band analysis is photosensitive.MAIN_IPS_SYNC.analyze_band, run with this module
as its config (MAIN_AYA_SYNC.py).
"""
from psychedelics import config_aya as base

BANDS = {"broadband": (1.0, 45.0), **base.BANDS}

FILTER_ORDER = 4
# filter + Hilbert on every run of contiguous epochs (reflection padding at the ends of the run):
# no trimming, the whole 5 s epochs are analysed (before 2026-11-02: each 5 s epoch filtered alone
# and 1 s trimmed at both ends, 3 s analysed)
EDGE_TRIM_S = 0.0
MT_NW = 2.0                       # multitaper cross-spectra (3 DPSS tapers, +-0.4 Hz on 5 s epochs)
NEW_METHOD_SUBDIR = "new_method"

N_SURROGATES = 20
SEED = 0
R_HIST_BINS = 50
R_EXAMPLE_S = 5.0                 # the whole epoch
DECIMATE_PLOT = 2                 # 125 Hz for the stored R(t) / PLV(t)

INDICES = ("kuramoto", "pc", "plv", "ppc", "pli", "wpli", "dwpli", "icoh")

ALPHA = 0.05
OVERWRITE = False


def params():
    return dict(INDICES=INDICES, BANDS=BANDS, FILTER_ORDER=FILTER_ORDER, EDGE_TRIM_S=EDGE_TRIM_S,
                N_SURROGATES=N_SURROGATES, SEED=SEED, R_HIST_BINS=R_HIST_BINS,
                R_EXAMPLE_S=R_EXAMPLE_S, DECIMATE_PLOT=DECIMATE_PLOT, CONDITIONS=repr(base.CONDITIONS),
                METHOD="new: filter/Hilbert per run, M over the whole signal, cross-spectra across epochs",
                MT_NW=MT_NW, PAD_S=base.PAD_S, EQUAL_MODE=base.EQUAL_MODE, EQUAL_DRAWS=base.EQUAL_DRAWS,
                EQUAL_SEED=base.EQUAL_SEED)
