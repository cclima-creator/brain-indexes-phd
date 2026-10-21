# -*- coding: utf-8 -*-
"""
Impostazioni dell'analisi della dinamica delle valanghe sul database IPS (MAIN_IPS_AVALDYN.py /
plot_ips_avaldyn.py). Binarizzazione, soglie, bande e time bin sono gli STESSI dell'analisi delle
valanghe (config_ips_avalan.py), cosi' le valanghe sono identiche; qui solo i parametri nuovi.
"""
from photosensitive import config_ips as base
from photosensitive import config_ips_avalan as av

GROUPS = av.GROUPS
COLORS = av.COLORS
BANDS = av.BANDS
OUT_NAME = "avalanches_dynamics"   # risultati in criticality_results/avalanches_dynamics

# ---- repertorio
MIN_MULTI_CHANNELS = 3            # seconda riga della fig. D1: pattern piu' frequenti con almeno 3 canali
N_TOP_PATTERNS = 12                # pattern piu' frequenti disegnati per gruppo e banda (fig. D1)
MDS_MAX_PATTERNS = 300             # pattern piu' frequenti per gruppo nella mappa della diversita' (D2)
N_COMPONENTS = 5                   # componenti della PCA dei pattern (D3)
MAX_PER_SUBJECT = 3000             # valanghe (o passaggi) per soggetto nei fit comuni: ogni soggetto pesa uguale

# ---- dinamica (campo di deriva e punti fissi)
GRID = 15                          # celle per lato della griglia sul piano PC1-PC2 (D5)
MIN_COUNT = 20                     # passaggi minimi per cella
POLY_DEGREE = 3                    # grado del polinomio del campo di deriva (punti fissi)

# ---- sequenza delle valanghe
N_STATES = 8                       # stati (k-means dei pattern) per la matrice di transizione (D6)
N_PERM = 200                       # rimescolamenti per il null dell'informazione mutua

OVERWRITE = False                  # True: ignora la cache per soggetto


def results_dir(channel_set, prelim_root):
    """criticality_results/avalanches_dynamics (o criticality_results/occipital/avalanches_dynamics)."""
    return av.results_dir(channel_set, prelim_root) / OUT_NAME


def bin_s(band, sfreq):
    """Time bin della banda in secondi (stesso di MAIN_IPS_AVALAN)."""
    if av.IEI_MODE == "per_band":
        return av.BIN_SAMPLES[band] / sfreq
    return av.MAX_IEI_MS / 1000


def params():
    """Parametri che cambiano le traiettorie salvate nella cache per soggetto."""
    return dict(THRESHOLD_SD=av.THRESHOLD_SD, EVENT_MODE=av.EVENT_MODE, THRESH_TYPE=av.THRESH_TYPE,
                BANDS=BANDS, IEI_MODE=av.IEI_MODE, BIN_SAMPLES=av.BIN_SAMPLES, MAX_IEI_MS=av.MAX_IEI_MS,
                FIXED_LENGTH_S=base.FIXED_LENGTH_S, ANALYSIS_WINDOW=base.ANALYSIS_WINDOW)
