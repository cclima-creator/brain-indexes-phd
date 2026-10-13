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
PHOTOSENSITIVE_FOLDER = r"C:\Users\Chiara Clima\Desktop\eeg preprocess criticality\eeg photosensitive\removed epo + ICA + removed bad IC"
NOT_PHOTOSENSITIVE_FOLDER = r"C:\Users\Chiara Clima\Desktop\eeg preprocess criticality\eeg not photosensitive\removed epo + ICA + removed bad IC"
GROUPS = {"photosensitive": PHOTOSENSITIVE_FOLDER,
          "not_photosensitive": NOT_PHOTOSENSITIVE_FOLDER}
COLORS = {"photosensitive": "tab:red", "not_photosensitive": "tab:blue"}
OUT_DIR = Path(r"C:\Users\Chiara Clima\Desktop\eeg preprocess criticality\analysis_output_ips")

# ---- caricamento
FIXED_LENGTH_S = None      # solo per registrati continui: durata delle epoche (s)
ANALYSIS_WINDOW = None     # es. (1.0, 11.0) per escludere un pre-stimolo; None = epoca intera

# ---- spettro e bande
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 45)}
BAND_TOTAL = (1.0, 45.0)    # range per la potenza totale (potenza relativa)
FMIN, FMAX_PSD = 1.0, 70.0
SEGMENT_S = None            # None = una PSD sull'intera epoca
COMMON_F = np.arange(FMIN, FMAX_PSD, 0.1)

# ---- SNR
SNR_SIGNAL = (1.0, 45.0)    # SNR a banda larga: range di segnale
SNR_NOISE = (55.0, 70.0)    #                    range di rumore
NB_GAP, NB_WIDTH = 0.3, 1.5 # SNR a banda stretta: bin vicini da 0.3 a 1.5 Hz
N_HARMONICS = 3
ROI_CHANNELS = ["Fp1", "AF7", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "O1", "AF8", "O2", "T6", "P3", "P4", "Pz"]

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
    "fig2_psd_each_subject.png", "fig3_epoch_maps.png",
    "fig4_snr_vs_stim_freq.png", "fig4_snr_vs_stim_freq_interactive.html",
    "fig5_hist_gauss_bands.png", "fig6_relative_power_boxplot.png",
    "fig7_amplitude_hist_subjects.png", "fig8_amplitude_zscore_overlay.png",
    "risultati_eeg.xlsx", "results.pkl",
]
SHEET_ORDER = ["group_summary_bands", "group_tests", "subject_level_metrics",
               "snr_by_stim_freq_group", "epoch_table_snr", "band_power_per_subject",
               "band_power_per_channel"]