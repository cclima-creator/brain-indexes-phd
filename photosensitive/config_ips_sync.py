# -*- coding: utf-8 -*-
"""
Settings of the synchronization analysis on the IPS database (MAIN_IPS_SYNC.py / plot_ips_sync.py).
Paths, groups, colors and channel choice come from config_ips.py (same data as the preliminary
and avalanche analyses); only the synchrony parameters are here. For another database: copy this file.
"""
from pathlib import Path

from photosensitive import config_ips as base

GROUPS = base.GROUPS
COLORS = base.COLORS
OUT_SUBDIR = "synchrony"         # used only if OUT_ROOT = None: results in OUT_DIR/synchrony
# results folder, next to criticality_results: OUT_ROOT (all channels) or OUT_ROOT/occipital
# (visual ROI). None = inside the preliminary-analysis folder (OUT_DIR/synchrony)
OUT_ROOT = Path(r"C:\Users\Chiara Clima\Desktop\eeg preprocess photosensitive\syncron_results")

# ---- bands: broadband 1-45 Hz (as the other analyses) and the narrow bands delta 1-4, theta 4-8,
# alpha 8-13, beta 13-30, gamma 30-45 Hz. CAUTION: the Hilbert phase is well defined only for
# narrow-band signals; on the broadband signal it follows the dominant rhythm (often alpha) and
# jumps between components, so the broadband synchrony values are a rough summary, not a phase
# of one oscillation.
BANDS = {"broadband": (1.0, 45.0), **base.BANDS}

# ---- phase extraction (synchrony_criticality/phase.py)
FILTER_ORDER = 4                  # Butterworth order (doubled by the forward-backward filtering)
EDGE_TRIM_S = 1.0                 # seconds removed at both ends of each epoch after Hilbert
                                  # (12 s epochs -> 10 s analysed; 1 s = 1 cycle at 1 Hz)

# ---- Kuramoto (synchrony_criticality/kuramoto.py)
N_SURROGATES = 20                 # circular-shift surrogates per subject and band (0 = none)
SEED = 0
R_HIST_BINS = 50                  # histogram of R(t) in [0, 1], kept for the figures
R_EXAMPLE_S = 4.0                 # seconds of R(t) of the first epoch kept for the figures
DECIMATE_PLOT = 8                 # R(t) and PLV(t) kept for the figures every 8 samples (128 Hz)

# ---- indices computed (changing this list re-analyses the cached subjects)
# Kuramoto (GS, M), PC, PLV / PPC are sensitive to zero-lag mixing: with channels rebuilt from 5 ICA
# components they mostly reflect the mixing, and GS / M are a relative comparison between groups.
# PLI, wPLI, dwPLI, iCOH use only the lagged part of the coupling (synchrony_criticality/lag_insensitive.py).
INDICES = ("kuramoto", "pc", "plv", "ppc", "pli", "wpli", "dwpli", "icoh")

# ---- scalp regions for the within- / between-region analysis (MAIN_IPS_SYNC_REGIONS.py);
# 10-20 names of this montage (T3 = T7, T4 = T8, T5 = P7, T6 = P8 are matched automatically)
REGIONS = {"frontopolar": ["Fp1", "Fp2", "AF7", "AF8"],
           "frontal": ["F7", "F3", "Fz", "F4", "F8"],
           "central": ["C3", "Cz", "C4"],
           "temporal": ["T3", "T4", "T5", "T6"],
           "parietal": ["P3", "Pz", "P4"],
           "occipital": ["O1", "O2"]}

# ---- group statistics
ALPHA = 0.05                      # significance for the edge-wise tests and the figures
MIN_SUBJECTS_PER_FREQ = 3         # stim-frequency tests only with >= 3 subjects per group

# ---- per-subject cache: if it exists with the same parameters it is not recomputed
OVERWRITE = False


def results_dir(channel_set, prelim_root):
    """Results folder: OUT_ROOT (or OUT_ROOT/occipital), or prelim_root/OUT_SUBDIR."""
    if OUT_ROOT is None:
        return Path(prelim_root) / OUT_SUBDIR
    return OUT_ROOT if channel_set == "all" else OUT_ROOT / base.OCCIPITAL_SUBDIR


def params():
    """Parameters that change the results: if different from the cached ones, the subject is redone."""
    return dict(INDICES=INDICES, BANDS=BANDS, FILTER_ORDER=FILTER_ORDER, EDGE_TRIM_S=EDGE_TRIM_S,
                N_SURROGATES=N_SURROGATES, SEED=SEED, R_HIST_BINS=R_HIST_BINS,
                R_EXAMPLE_S=R_EXAMPLE_S, DECIMATE_PLOT=DECIMATE_PLOT,
                FIXED_LENGTH_S=base.FIXED_LENGTH_S, ANALYSIS_WINDOW=base.ANALYSIS_WINDOW)
