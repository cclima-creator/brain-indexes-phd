# -*- coding: utf-8 -*-
"""
Avalanche settings for the Ayahuasca database. Same names as photosensitive/config_ips_avalan.py,
because the per-band analysis is photosensitive.MAIN_IPS_AVALAN.analyze_band, run with this
module as its config (MAIN_AYA_AVALAN.py). Only what differs from IPS is commented.
"""
from psychedelics import config_aya as base

# one threshold for every condition (paired design: the same rule in pre, during and post).
# Divergence point from the Gaussian of the broadband (1-40 Hz) z-scores, mean over subjects
# per condition: 0-20 1.85, 20-40 1.85, 40-60 1.88, pre 1.88, post 1.93, during 27-33 1.90 SD
THRESHOLD_SD = 1.9
EVENT_MODE = "peak"
THRESH_TYPE = "both"
DIVERGENCE_MIN_SD = 1.0

BANDS = {"broadband": (1.0, 40.0), **base.BANDS}

# 250 Hz: one sample = 4 ms. The per-band bins of IPS (1024 Hz, from the intra-avalanche IEI:
# broadband 1.7, delta 7.1, theta 3.3, alpha 2.2, beta 1.5, gamma 1.2 ms) rounded to whole samples
# at 250 Hz: delta 2 samples (8 ms), every other band 1 sample (4 ms, the shortest possible bin)
MAX_IEI_MS = 8.0
IEI_MODE = "per_band"
BIN_SAMPLES = {"broadband": 1, "delta": 2, "theta": 1, "alpha": 1, "beta": 1, "gamma": 1}
BR_BINS_MS = (4, 8, 12, 16, 20, 24, 28, 32)    # multiples of the 4 ms sample

XMIN = "min"
N_BOOT = 100
N_SIM = 100
MIN_TAIL = 30
MIN_PCT_IN_RANGE = 50.0
THIRD_METHOD = "ols"

REPERTOIRE_MIN_CHANNELS = 1
REPERTOIRE_MAX_PATTERNS = 4000

OVERWRITE = False


def threshold_for(group, band):
    """Same threshold for every condition and band."""
    return float(THRESHOLD_SD)


def params():
    """Parameters that change the results: if different from the cached ones, the subject is redone."""
    return dict(THRESHOLD_SD=THRESHOLD_SD, EVENT_MODE=EVENT_MODE, THRESH_TYPE=THRESH_TYPE,
                BANDS=BANDS, MAX_IEI_MS=MAX_IEI_MS, IEI_MODE=IEI_MODE, BIN_SAMPLES=BIN_SAMPLES,
                BR_BINS_MS=tuple(BR_BINS_MS), XMIN=XMIN, N_BOOT=N_BOOT, N_SIM=N_SIM,
                MIN_TAIL=MIN_TAIL, THIRD_METHOD=THIRD_METHOD,
                REPERTOIRE_MIN_CHANNELS=REPERTOIRE_MIN_CHANNELS, CONDITIONS=repr(base.CONDITIONS),
                FILTER="contiguous runs, reflection padding", PAD_S=base.PAD_S, EQUAL_MODE=base.EQUAL_MODE,
                EQUAL_DRAWS=base.EQUAL_DRAWS, EQUAL_SEED=base.EQUAL_SEED)
