# -*- coding: utf-8 -*-
"""
Settings of the Ayahuasca (Santo Daime) database: eyes-closed resting state / meditation.

Recordings per subject (EEGLAB .set + .fdt, 16 channels, 250 Hz, 5 s epochs, bad epochs removed):
  PRE      6 min eyes-closed rest before the drug                      Daime_pre_post/AYA PRE
  DURING   60 min eyes-closed meditation after Ayahuasca, split in three 20 min slots
           (0-20, 20-40, 40-60)                                       Daime_during/split_20minutes_.../aya
  POST     6 min eyes-closed rest after the meditation                 Daime_pre_post/AYA POST

Every epoch of the DURING files has the label "<t>s" = end of the epoch in seconds from the start
of the hour (epoch = (t - 5, t] s), so the central 6 min (minutes 7-13 of the 20-40 slot = 27-33 of
the hour) are the epochs with 1620 < t <= 1980.

Every participant did the ceremony twice, once with Ayahuasca and once with placebo (same layout,
PLA instead of AYA in the file names; placebo DURING files in .../pla/pla).

Analyses with three conditions (repeated measures on the same subjects, ANALYSES):
  1 intra_aya / 3 intra_placebo                slot 0-20 vs 20-40 vs 40-60 min
  2 pre_during_post / 4 ..._placebo            PRE vs DURING central 6 min (27-33) vs POST
  7 rest_vs_aya_vs_placebo                     rest before the ceremony vs DURING 27-33 Ayahuasca
                                               vs DURING 27-33 placebo; the rest is the pre of one of
                                               the two ceremonies, drawn at random 50/50 (REST_PRE)
Drug x time comparisons (COMPARISONS): the Ayahuasca and the placebo trajectory of 1 vs 3 and 2 vs 4.
"""
import os
from pathlib import Path

ROOT = Path(r"C:\Users\Chiara Clima\Desktop\eeg preprocessed ayauasca")
PRE_DIR = ROOT / "Daime_pre_post" / "AYA PRE"
POST_DIR = ROOT / "Daime_pre_post" / "AYA POST"
DURING_DIR = ROOT / "Daime_during" / "split_20minutes_from60minutesepochedremoved_5s_numerated" / "aya"
PLA_PRE_DIR = ROOT / "Daime_pre_post" / "PLACEBO PRE"
PLA_POST_DIR = ROOT / "Daime_pre_post" / "PLACEBO POST"
PLA_DURING_DIR = ROOT / "Daime_during" / "split_20minutes_from60minutesepochedremoved_5s_numerated" / "pla" / "pla"
# everything is written here; AYA_OUT_DIR / AYA_ONLY_SUBJECTS / AYA_MAX_EPOCHS only for quick tests
OUT_DIR = Path(os.environ.get("AYA_OUT_DIR", ROOT / "results"))
ONLY_SUBJECTS = [s for s in os.environ.get("AYA_ONLY_SUBJECTS", "").split(",") if s]
MAX_EPOCHS = int(os.environ["AYA_MAX_EPOCHS"]) if os.environ.get("AYA_MAX_EPOCHS") else None

# ---- conditions: name -> (file pattern with {sid}, window on the hour in s or None)
# window = (t0, t1): keep the epochs whose end time t (label "<t>s") is in (t0, t1]
CENTRAL_WINDOW_S = (27 * 60, 33 * 60)
CONDITIONS = {
    "slot_0-20": (DURING_DIR, "{sid}_AYA_REST_DURING_*_0-20.set", None),
    "slot_20-40": (DURING_DIR, "{sid}_AYA_REST_DURING_*_20-40.set", None),
    "slot_40-60": (DURING_DIR, "{sid}_AYA_REST_DURING_*_40-60.set", None),
    "pre": (PRE_DIR, "{sid}_AYA_REST_PRE_*.set", None),
    "during_27-33": (DURING_DIR, "{sid}_AYA_REST_DURING_*_20-40.set", CENTRAL_WINDOW_S),
    "post": (POST_DIR, "{sid}_AYA_REST_POST_*.set", None),
    "pla_slot_0-20": (PLA_DURING_DIR, "{sid}_PLA_REST_DURING_*_0-20.set", None),
    "pla_slot_20-40": (PLA_DURING_DIR, "{sid}_PLA_REST_DURING_*_20-40.set", None),
    "pla_slot_40-60": (PLA_DURING_DIR, "{sid}_PLA_REST_DURING_*_40-60.set", None),
    "pla_pre": (PLA_PRE_DIR, "{sid}_PLA_REST_PRE_*.set", None),
    "pla_during_27-33": (PLA_DURING_DIR, "{sid}_PLA_REST_DURING_*_20-40.set", CENTRAL_WINDOW_S),
    "pla_post": (PLA_POST_DIR, "{sid}_PLA_REST_POST_*.set", None),
}
# virtual condition: for every subject one of the two real conditions (io_aya.rest_assignment):
# drawn at random 50/50 among the subjects that have both, the available one otherwise
REST_PRE = ("pre", "pla_pre")
REST_SEED = 2026
VIRTUAL = {"rest_pre": REST_PRE}

LABELS = {"slot_0-20": "0-20 min", "slot_20-40": "20-40 min", "slot_40-60": "40-60 min",
          "pre": "pre", "during_27-33": "during\n(27-33 min)", "post": "post",
          "pla_slot_0-20": "0-20 min", "pla_slot_20-40": "20-40 min", "pla_slot_40-60": "40-60 min",
          "pla_pre": "pre", "pla_during_27-33": "during\n(27-33 min)", "pla_post": "post",
          "rest_pre": "rest\n(pre-ceremony)"}
LABELS_7 = {"during_27-33": "Ayahuasca\n(27-33 min)", "pla_during_27-33": "placebo\n(27-33 min)"}


def labels_for(analysis):
    """Tick labels of an analysis (analysis 7 names the drug, the others the time)."""
    return {**LABELS, **LABELS_7} if analysis.startswith("7_") else LABELS

# ---- analyses: name -> ordered conditions (the order is the x axis of every figure)
ANALYSES = {
    "1_intra_aya": ["slot_0-20", "slot_20-40", "slot_40-60"],
    "2_pre_during_post": ["pre", "during_27-33", "post"],
    "3_intra_placebo": ["pla_slot_0-20", "pla_slot_20-40", "pla_slot_40-60"],
    "4_pre_during_post_placebo": ["pla_pre", "pla_during_27-33", "pla_post"],
    "7_rest_vs_aya_vs_placebo": ["rest_pre", "during_27-33", "pla_during_27-33"],
}
ANALYSIS_TITLES = {"1_intra_aya": "Intra-Ayahuasca: three 20-min slots of the meditation",
                   "2_pre_during_post": "Pre vs during (27-33 min) vs post Ayahuasca",
                   "3_intra_placebo": "Intra-placebo: three 20-min slots of the meditation",
                   "4_pre_during_post_placebo": "Pre vs during (27-33 min) vs post placebo",
                   "7_rest_vs_aya_vs_placebo": "Rest (pre-ceremony) vs Ayahuasca vs placebo (27-33 min)"}
# drug x time: name -> (Ayahuasca conditions, placebo conditions, time labels)
COMPARISONS = {
    "5_intra_aya_vs_placebo": (ANALYSES["1_intra_aya"], ANALYSES["3_intra_placebo"],
                               ["0-20 min", "20-40 min", "40-60 min"]),
    "6_pre_during_post_aya_vs_placebo": (ANALYSES["2_pre_during_post"], ANALYSES["4_pre_during_post_placebo"],
                                         ["pre", "during 27-33", "post"]),
}
COMPARISON_TITLES = {"5_intra_aya_vs_placebo": "Trajectory over the meditation: Ayahuasca vs placebo",
                     "6_pre_during_post_aya_vs_placebo": "Pre-during-post: Ayahuasca vs placebo"}
DRUG_COLORS = {"aya": "#d62728", "pla": "#1f77b4"}
COLORS = {"slot_0-20": "#9ecae1", "slot_20-40": "#3182bd", "slot_40-60": "#08519c",
          "pre": "#7f7f7f", "during_27-33": "#d62728", "post": "#2ca02c",
          "pla_slot_0-20": "#fdd0a2", "pla_slot_20-40": "#fd8d3c", "pla_slot_40-60": "#d94801",
          "pla_pre": "#7f7f7f", "pla_during_27-33": "#1f77b4", "pla_post": "#2ca02c",
          "rest_pre": "#7f7f7f"}

# module folders inside every analysis folder
MODULE_DIRS = {"stats": "0_statistical_analysis", "avalanches": "1_avalanches",
               "synchrony": "2_synchrony", "chaos": "3_edge_of_chaos"}

# ---- bands (same as the other databases); broadband 1-45 Hz (avalanches 1-40 Hz as Maschke 2024)
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
BAND_ORDER = ["broadband", *BANDS]

EXCLUDED_SUBJECTS = []                    # IDs (e.g. "P07") left out of every analysis

# ---- statistics (paired, same subjects in every condition)
ALPHA = 0.05
# omnibus: Friedman test across the 3 conditions (Kendall's W as effect size)
# post hoc: Wilcoxon signed-rank on each pair (matched-pairs rank-biserial r as effect size)
# FDR (Benjamini-Hochberg):
#   per-index figures: Friedman p across the bands of that index; post hoc p across the
#                      3 pairs x bands of that index
#   tables: also p_fdr_all = Friedman p across every index x band of the module

N_JOBS = 7                                # subject x condition analysed in parallel (8 cores)

# ---- continuous signal and equal amount of data (descriptive_analysis/segments.py)
# Filters and Hilbert transforms run on every run of contiguous epochs (io_aya.contiguous_runs),
# padded by PAD_S s of reflection at its two ends: no edge trimming inside the recording.
PAD_S = 2.0
# Indices that depend on the amount of data are computed on io_aya.n_equal epochs (the minimum of
# the subject over the conditions of the same duration), as the mean over EQUAL_DRAWS blocks of
# consecutive epochs with evenly spaced starts (blocks keep the contiguity of the recording).
EQUAL_MODE = "blocks"
EQUAL_DRAWS = 5
EQUAL_SEED = 0
