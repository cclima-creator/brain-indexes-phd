# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 16:14:11 2026

@author: Chiara Clima
"""

"""
Impostazioni SPECIFICHE del database IPS (stimolazione fotica, fotosensibili vs non).
Per un altro database: copia questo file (config_<nome>.py) e cambia solo qui
percorsi, gruppi, ROI e la funzione parse_stim_freq.
"""
import re
from pathlib import Path

import numpy as np

# ---- dati e output
# dataset con solo le 5 componenti brain principali per soggetto (<ID>_5compo.set)
PHOTOSENSITIVE_FOLDER = r"C:\Users\Chiara Clima\Desktop\eeg preprocess criticality\eeg photosensitive\photosensitive_5components"
NOT_PHOTOSENSITIVE_FOLDER = r"C:\Users\Chiara Clima\Desktop\eeg preprocess criticality\eeg not photosensitive\not_photosensitive_5components"
GROUPS = {"photosensitive": PHOTOSENSITIVE_FOLDER,
          "not_photosensitive": NOT_PHOTOSENSITIVE_FOLDER}
COLORS = {"photosensitive": "tab:red", "not_photosensitive": "tab:blue"}
# soggetti esclusi dall'analisi (ID = parte del nome file prima del primo "_")
EXCLUDED_SUBJECTS = {"not_photosensitive": ["9303987"]}


def subject_files(group, files):
    """Toglie dalla lista dei file i soggetti in EXCLUDED_SUBJECTS[group]."""
    excl = set(EXCLUDED_SUBJECTS.get(group, []))
    return [f for f in files if f.stem.split("_")[0] not in excl]
OUT_DIR = Path(r"C:\Users\Chiara Clima\Desktop\eeg preprocess criticality\analysis_output_ips_5compo")

# ---- quali canali analizzare
CHANNEL_SET = None          # None = lo script chiede all'avvio; "all" = tutti i canali EEG;
                            # "occipital" = solo la ROI visiva estesa qui sotto
OCCIPITAL_CHANNELS = ["O1", "O2", "P3", "P4", "Pz", "T5", "T6"]
CHANNEL_ALIASES = {"T5": "P7", "T6": "P8"}   # stesso elettrodo con il nome del sistema 10-10
OCCIPITAL_SUBDIR = "occipital"   # i risultati occipitali vanno in OUT_DIR / occipital


def choose_channel_set(choice=None):
    """
    Sceglie i canali da analizzare e la cartella dei risultati.
    choice (o CHANNEL_SET) = "all" / "occipital"; se None lo chiede in console.
    Ritorna (nome, lista dei canali da tenere o None = tutti, cartella di output).
    """
    choice = choice or CHANNEL_SET
    while choice not in ("all", "occipital"):
        ans = input("Analizzare tutti i canali [a] o solo la ROI visiva estesa "
                    f"{OCCIPITAL_CHANNELS} [o]? ").strip().lower()
        choice = {"a": "all", "all": "all", "o": "occipital", "occipital": "occipital"}.get(ans)
    if choice == "all":
        return "all", None, OUT_DIR
    chans = OCCIPITAL_CHANNELS + [CHANNEL_ALIASES[c] for c in OCCIPITAL_CHANNELS if c in CHANNEL_ALIASES]
    return "occipital", chans, OUT_DIR / OCCIPITAL_SUBDIR


# ---- caricamento
FIXED_LENGTH_S = None      # solo per registrati continui: durata delle epoche (s)
ANALYSIS_WINDOW = None     # es. (1.0, 11.0) per escludere un pre-stimolo; None = epoca intera

# ---- spettro e bande
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 45)}
BAND_TOTAL = (1.0, 45.0)    # range per la potenza totale (potenza relativa)
FMIN, FMAX_PSD = 1.0, 70.0
SEGMENT_S = None            # None = una PSD sull'intera epoca
FMAX_ANALYSIS = 45.0         # sopra i 45 Hz e' rumore: grafici e SNR a banda stretta si fermano qui
PSD_PLOT_FMAX = FMAX_ANALYSIS
                             # FMAX_PSD resta 70 SOLO perche' l'SNR a banda larga usa 55-70 Hz come stima del rumore
COMMON_F = np.arange(FMIN, PSD_PLOT_FMAX + 1e-9, 0.1)

# ---- SNR
SNR_SIGNAL = (1.0, 45.0)    # SNR a banda larga: range di segnale
SNR_NOISE = (55.0, 70.0)    #                    range di rumore
NB_GAP, NB_WIDTH = 0.3, 1.5 # SNR a banda stretta: bin vicini da 0.3 a 1.5 Hz
N_HARMONICS = 3
ROI_CHANNELS = ["Fp1", "AF7", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "O1", "AF8", "O2", "T6", "P3", "P4", "Pz"]

# ---- varianza, ACFW, stazionarieta', 1/f, CDF
STAT_N_SEGMENTS = 20        # stazionarieta': ogni epoca divisa in 20 segmenti (Reverse Arrangement Test)
STAT_ALPHA = 0.05           # p sotto cui un'epoca/canale e' considerata non stazionaria
ACF_MAX_LAG_S = 1.0         # ACFW: lag massimo cercato (s); oltre -> NaN
APERIODIC_RANGE = (2.0, 40.0)  # range del fit FOOOF (Hz)
APERIODIC_MODE = "knee"      # FOOOF: "knee" (fondo piatto sotto una frequenza, poi 1/f) o "fixed" (retta)
FOOOF_SETTINGS = dict(peak_width_limits=(1.0, 12.0), max_n_peaks=8, min_peak_height=0.05,
                      peak_threshold=2.0)
ALPHA_PEAK_BAND = (8.0, 13.0) # dove cercare il picco alfa individuale tra i picchi di FOOOF
KS_N_PERM = 5000            # permutazioni del test KS sulle CDF
LOO_Z_THRESH = 3.0          # fig1b: soggetto 'influente' se il suo contributo supera mediana + 3 MAD
LOO_MIN_DB = 0.5            #        e sposta la media del gruppo di almeno 0.5 dB in media

# ---- etichette eventi -> frequenza di stimolo
STIM_FREQ_MAP = {}          # es. {"S 11": 10.0}: ha la precedenza sul parsing automatico


def parse_stim_freq(label):
    """Estrae la frequenza (Hz) da etichette tipo '22 Hz/Olhos abertos/...' o 'Olhos Abertos/10 Hz/condition 16'."""
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*hz", str(label), re.I)
    if m:
        return float(m.group(1).replace(",", "."))
    nums = re.findall(r"\d+(?:\.\d+)?", str(label))
    return float(nums[-1]) if nums else np.nan


# ---- file prodotti (servono al controllo iniziale della cartella di output)
OUTPUT_NAMES = [
    "fig1_psd_all_subjects.png", "fig1_psd_all_subjects_interactive.html",
    "fig1b_psd_leave_one_out.png",
    "fig2_psd_each_subject.png", "fig3_epoch_maps.png",
    "fig4_snr_vs_stim_freq.png", "fig4_snr_vs_stim_freq_interactive.html",
    "fig5_hist_gauss_bands.png", "fig6_relative_power_boxplot.png",
    *[f"fig8_amplitude_zscore_{g}.png" for g in GROUPS],
    "fig9_epoch_counts.png", "fig10_stationarity.png", "fig11_variance_acfw.png",
    "fig12_aperiodic.png", "fig13_topo_band_power.png", "fig14_topo_signal.png",
    "risultati_eeg.xlsx", "results.pkl",
]
SHEET_ORDER = ["group_summary_bands", "group_tests", "subject_level_metrics", "psd_leave_one_out",
               "epoch_counts_group", "epoch_counts", "ks_permutation_tests", "channel_tests",
               "snr_by_stim_freq_group", "epoch_table_snr", "band_power_per_subject",
               "band_power_per_channel", "signal_per_channel", "signal_per_epoch_channel"]