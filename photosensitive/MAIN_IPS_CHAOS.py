# -*- coding: utf-8 -*-
"""
MAIN edge of chaos, IPS database: 0-1 chaos test, lambda_max and covariance width, LZC, Katz FD,
Renyi entropy, multiscale entropy, entropy production, DFA, fE/I, aperiodic exponent
(edge_of_chaos/MAIN_chaos.run, functions in edge_of_chaos/chaos_indexes.py).
Same files, groups, excluded subjects and channel choice as the other IPS analyses (config_ips.py).

Results (Excel, pickle, per-subject cache in subjects/, figures in figures/) in
<eeg preprocess photosensitive>/edge_of_chaos_results, next to criticality_results and
syncron_results (edge_of_chaos_results/occipital for the visual ROI).
Settings: defaults of edge_of_chaos/config_chaos.py, overridden below for this database.
To redo everything: OVERWRITE = True below.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from photosensitive import config_ips as base
from edge_of_chaos import config_chaos as cfg
from edge_of_chaos.MAIN_chaos import run

OUT_ROOT = Path(r"C:\Users\Chiara Clima\Desktop\eeg preprocess photosensitive\edge_of_chaos_results")

# ---- overrides of config_chaos for the IPS database
cfg.BANDS = dict(base.BANDS)          # same bands as the other IPS analyses
cfg.SIGNAL_BANDS = ["broadband", *base.BANDS]   # every measure on broadband and on each band
# 12 s epochs, one per stimulation frequency: separate trials. Each epoch is filtered alone (padded
# by reflection) and trimmed to the 10 s of stimulation (EDGE_TRIM_S = 1); K is computed on each
# 10 s trial (K_CHUNK_S = None) and averaged. Before 2026-11-02 the trimmed epochs (different
# stimulation frequencies) were concatenated into one long signal for K, DFA and fE/I.
# DFA and fE/I are left out: long-range temporal correlations need minutes of continuous signal
# (Hardstone et al. 2012), here there are separate 10 s trials.
cfg.EDGE_TRIM_S = 1.0
cfg.K_CHUNK_S = None
cfg.MEASURES = [m for m in cfg.MEASURES if m not in ("dfa", "fei_ratio")]
cfg.EQUAL_EPOCHS = True               # covariance width / lambda_max, entropy production on the
cfg.EQUAL_MODE = "random"             # minimum number of epochs over the subjects (random subsets)
cfg.EQUAL_DRAWS = base.EQUAL_DRAWS
cfg.PAD_S = base.PAD_S
cfg.OVERWRITE = False
N_JOBS = 7                            # subjects analysed in parallel (8 cores: one left free)


def main():
    channel_set, channels, _ = base.choose_channel_set()
    out = OUT_ROOT if channel_set == "all" else OUT_ROOT / base.OCCIPITAL_SUBDIR
    run(base.GROUPS, out, cfg, colors=base.COLORS, fixed_length_s=base.FIXED_LENGTH_S,
        window=base.ANALYSIS_WINDOW, channels=channels, file_filter=base.subject_files, n_jobs=N_JOBS)


if __name__ == "__main__":
    main()
