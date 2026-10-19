# -*- coding: utf-8 -*-
"""
Impostazioni dell'analisi delle valanghe sul database IPS (MAIN_IPS_AVALAN.py / plot_ips_avalan.py).
Percorsi, gruppi, colori e scelta dei canali vengono da config_ips.py (stessi dati dell'analisi
preliminare); qui solo i parametri delle valanghe. Per un altro database: copia questo file.
"""
from pathlib import Path

from photosensitive import config_ips as base

GROUPS = base.GROUPS
COLORS = base.COLORS
OUT_SUBDIR = "avalanches"        # usato solo se OUT_ROOT = None: risultati in OUT_DIR/avalanches
# cartella dei risultati delle valanghe (tutti i canali; la ROI occipitale va in OUT_ROOT/occipital).
# None = dentro la cartella dell'analisi preliminare (OUT_DIR/avalanches)
OUT_ROOT = Path(r"C:\Users\Chiara Clima\Desktop\eeg preprocess criticality\criticality_results")

# ---- binarizzazione (Maschke et al. 2024; Shriki et al. 2013)
# soglia in SD dal punto in cui la distribuzione degli z-score si stacca dalla gaussiana
# (fig. A1, broadband 5 componenti: 1.72 fotosensibili, 1.85 non fotosensibili).
# Un numero per gruppo, oppure un dict {banda: soglia} per gruppo.
THRESHOLD_SD = {"photosensitive": 1.7, "not_photosensitive": 1.9}
EVENT_MODE = "peak"              # "peak": un evento al picco di ogni escursione (Maschke/edgeofpy)
                                 # "excursion": ogni campione oltre soglia (Sorrentino 2021)
THRESH_TYPE = "both"             # escursioni positive e negative (|z| > soglia)
DIVERGENCE_MIN_SD = 1.0          # il punto di divergenza si cerca oltre 1 SD (vedi binarization.divergence_threshold)

# ---- bande: "broadband" come Maschke (1-40 Hz), poi le bande di config_ips
BANDS = {"broadband": (1.0, 40.0), **base.BANDS}

# ---- valanghe
MAX_IEI_MS = 8.0                 # inter-event interval = time bin delle valanghe (Maschke 2024: 8 ms)
IEI_MODE = "per_band"            # "per_band": BIN_SAMPLES[banda], uguale per tutti i soggetti e gruppi
                                 # "fixed": MAX_IEI_MS per tutti (Maschke 2024)
                                 # "mean_iei": per ogni soggetto e banda, l'IEI medio tra eventi successivi
                                 # (Beggs & Plenz 2003; cambia molto tra le bande: ~4 ms gamma, ~70 ms delta)
# time bin per banda in campioni (a 1024 Hz 1 campione = 0.977 ms), dall'IEI robusto intra-valanga
# (iei.py) arrotondato al numero di campioni piu' vicino:
#   broadband 1.663 ms -> 2 (1.953 ms) | delta 7.095 -> 8 (7.813) | theta 3.336 -> 4 (3.906)
#   alpha 2.236 -> 2 (1.953)           | beta 1.527 -> 2 (1.953)  | gamma 1.153 -> 1 (0.977)
BIN_SAMPLES = {"broadband": 2, "delta": 8, "theta": 4, "alpha": 2, "beta": 2, "gamma": 1}
BR_BINS_MS = (4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 28, 30, 35)   # sensibilita' del branching ratio

# ---- power law (Clauset 2009)
XMIN = "min"                     # xmin di tau/alpha/DCC: "min" = minimo dei dati (edgeofpy, Maschke 2024);
                                 # None = Clauset (minimo di KS; sempre riportato anche come clauset_*)
N_BOOT = 100                     # p-value di Clauset: 100 per esplorare, 1000-2500 per i risultati finali
N_SIM = 100                      # p-value di ogni range nella ricerca del cutoff
MIN_TAIL = 30                    # minimo di valanghe nella coda / nel range per fittare
THIRD_METHOD = "ols"             # 1/(sigma nu z): "ols" (Friedman 2012) o "edgeofpy" (solo confronto)

# ---- repertorio
REPERTOIRE_MIN_CHANNELS = 1      # pattern con almeno 1 canale (= tutti i non vuoti)
REPERTOIRE_MAX_PATTERNS = 4000   # sopra, la diversita' si stima su un sottoinsieme casuale

# ---- IEI robusto (avalanche_criticality/iei.py): per ora solo diagnostica, NON cambia il time bin
# delle valanghe (che resta MAX_IEI_MS / IEI_MODE). Risultati in iei_results.xlsx / .pkl.
RUN_AVALANCHES = True            # False: MAIN_IPS_AVALAN fa solo l'analisi dell'IEI
RUN_IEI = True
IEI_METHOD = "auto"              # "auto" = GMM -> valle dell'istogramma -> fallback
IEI_MIN_ASHMAN_D = 2.0           # GMM accettata solo se le due gaussiane sono separate (D >= 2)
IEI_FALLBACK_FACTOR = 5.0        # fallback: soglia = mediana x 5
IEI_SMOOTH_BINS = 2.0            # smoothing dell'istogramma log (bin da 0.05 decadi)
IEI_DEQUANTIZE = True            # jitter di ±0.5 campioni prima del log (vedi iei.py)

# ---- cache per soggetto: se esiste con gli stessi parametri non si ricalcola
OVERWRITE = False


def results_dir(channel_set, prelim_root):
    """Cartella dei risultati: OUT_ROOT (o OUT_ROOT/occipital), oppure prelim_root/OUT_SUBDIR."""
    if OUT_ROOT is None:
        return Path(prelim_root) / OUT_SUBDIR
    return OUT_ROOT if channel_set == "all" else OUT_ROOT / base.OCCIPITAL_SUBDIR


def threshold_for(group, band):
    """Soglia (SD) per gruppo e banda da THRESHOLD_SD."""
    t = THRESHOLD_SD[group]
    return float(t[band]) if isinstance(t, dict) else float(t)


def iei_params():
    """Parametri della cache dell'IEI (subjects_iei/): eventi come le valanghe + parametri del metodo."""
    return dict(THRESHOLD_SD=THRESHOLD_SD, EVENT_MODE=EVENT_MODE, THRESH_TYPE=THRESH_TYPE, BANDS=BANDS,
                IEI_METHOD=IEI_METHOD, IEI_MIN_ASHMAN_D=IEI_MIN_ASHMAN_D,
                IEI_FALLBACK_FACTOR=IEI_FALLBACK_FACTOR, IEI_SMOOTH_BINS=IEI_SMOOTH_BINS,
                IEI_DEQUANTIZE=IEI_DEQUANTIZE,
                FIXED_LENGTH_S=base.FIXED_LENGTH_S, ANALYSIS_WINDOW=base.ANALYSIS_WINDOW)


def params():
    """Parametri che cambiano i risultati: se diversi da quelli della cache, il soggetto si rifa'."""
    return dict(THRESHOLD_SD=THRESHOLD_SD, EVENT_MODE=EVENT_MODE, THRESH_TYPE=THRESH_TYPE,
                BANDS=BANDS, MAX_IEI_MS=MAX_IEI_MS, IEI_MODE=IEI_MODE, BIN_SAMPLES=BIN_SAMPLES, BR_BINS_MS=tuple(BR_BINS_MS), XMIN=XMIN,
                N_BOOT=N_BOOT, N_SIM=N_SIM, MIN_TAIL=MIN_TAIL, THIRD_METHOD=THIRD_METHOD,
                REPERTOIRE_MIN_CHANNELS=REPERTOIRE_MIN_CHANNELS,
                FIXED_LENGTH_S=base.FIXED_LENGTH_S, ANALYSIS_WINDOW=base.ANALYSIS_WINDOW)
