# -*- coding: utf-8 -*-
"""
MAIN edge of chaos, IPS database: 0-1 chaos test, lambda_max and covariance width, LZC, Katz FD,
Renyi entropy, multiscale entropy, entropy production, DFA, fE/I, aperiodic exponent
(edge_of_chaos/MAIN_chaos.run, functions in edge_of_chaos/chaos_indexes.py).
Same files, groups, excluded subjects and channel choice as the other IPS analyses (config_ips.py).

Results (Excel, pickle, per-subject cache in subjects/, figures in figures/) in
<eeg preprocess criticality>/edge_of_chaos_results, next to criticality_results and
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

OUT_ROOT = Path(r"C:\Users\Chiara Clima\Desktop\eeg preprocess criticality\edge_of_chaos_results")

# ---- overrides of config_chaos for the IPS database
cfg.BANDS = dict(base.BANDS)          # same bands as the other IPS analyses
cfg.SIGNAL_BANDS = ["broadband", *base.BANDS]   # every measure on broadband and on each band
# 12 s epochs, one per stimulation frequency: K, DFA and fE/I need long signals, so they are
# computed on the concatenation of the trimmed epochs (10 s each). With CONCATENATE_EPOCHS = False
# the DFA fit range would shrink to <= 1 s (1/10 of one epoch).
cfg.CONCATENATE_EPOCHS = True
cfg.OVERWRITE = False
N_JOBS = 7                            # subjects analysed in parallel (8 cores: one left free)


def main():
    channel_set, channels, _ = base.choose_channel_set()
    out = OUT_ROOT if channel_set == "all" else OUT_ROOT / base.OCCIPITAL_SUBDIR
    run(base.GROUPS, out, cfg, colors=base.COLORS, fixed_length_s=base.FIXED_LENGTH_S,
        window=base.ANALYSIS_WINDOW, channels=channels, file_filter=base.subject_files, n_jobs=N_JOBS)


if __name__ == "__main__":
    main()
