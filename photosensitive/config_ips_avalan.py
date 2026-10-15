# -*- coding: utf-8 -*-
"""
Impostazioni dell'analisi delle valanghe sul database IPS (MAIN_IPS_AVALAN.py / plot_ips_avalan.py).
Percorsi, gruppi, colori e scelta dei canali vengono da config_ips.py (stessi dati dell'analisi
preliminare); qui solo i parametri delle valanghe. Per un altro database: copia questo file.
"""
from photosensitive import config_ips as base

GROUPS = base.GROUPS
COLORS = base.COLORS
OUT_SUBDIR = "avalanches"        # risultati in OUT_DIR/avalanches (o OUT_DIR/occipital/avalanches)

# ---- binarizzazione (Maschke et al. 2024; Shriki et al. 2013)
# soglia in SD dal punto in cui la distribuzione degli z-score si stacca dalla gaussiana
# (analisi preliminare, fig8). Un numero per gruppo, oppure un dict {banda: soglia} per gruppo.
THRESHOLD_SD = {"photosensitive": 1.5, "not_photosensitive": 2.0}
EVENT_MODE = "peak"              # "peak": un evento al picco di ogni escursione (Maschke/edgeofpy)
                                 # "excursion": ogni campione oltre soglia (Sorrentino 2021)
THRESH_TYPE = "both"             # escursioni positive e negative (|z| > soglia)
DIVERGENCE_MIN_SD = 1.0          # il punto di divergenza si cerca oltre 1 SD (vedi binarization.divergence_threshold)

# ---- bande: "broadband" come Maschke (1-40 Hz), poi le bande di config_ips
BANDS = {"broadband": (1.0, 40.0), **base.BANDS}

# ---- valanghe
MAX_IEI_MS = 8.0                 # inter-event interval = time bin delle valanghe (Maschke 2024: 8 ms)
IEI_MODE = "fixed"               # "fixed": MAX_IEI_MS per tutti (Maschke 2024)
                                 # "mean_iei": per ogni soggetto e banda, l'IEI medio tra eventi successivi
                                 # (Beggs & Plenz 2003; cambia molto tra le bande: ~4 ms gamma, ~70 ms delta)
BR_BINS_MS = (4, 6, 8, 10, 12, 14, 16, 18)   # sensibilita' del branching ratio (analisi preliminare)

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

# ---- cache per soggetto: se esiste con gli stessi parametri non si ricalcola
OVERWRITE = False


def threshold_for(group, band):
    """Soglia (SD) per gruppo e banda da THRESHOLD_SD."""
    t = THRESHOLD_SD[group]
    return float(t[band]) if isinstance(t, dict) else float(t)


def params():
    """Parametri che cambiano i risultati: se diversi da quelli della cache, il soggetto si rifa'."""
    return dict(THRESHOLD_SD=THRESHOLD_SD, EVENT_MODE=EVENT_MODE, THRESH_TYPE=THRESH_TYPE,
                BANDS=BANDS, MAX_IEI_MS=MAX_IEI_MS, IEI_MODE=IEI_MODE, BR_BINS_MS=tuple(BR_BINS_MS), XMIN=XMIN,
                N_BOOT=N_BOOT, N_SIM=N_SIM, MIN_TAIL=MIN_TAIL, THIRD_METHOD=THIRD_METHOD,
                REPERTOIRE_MIN_CHANNELS=REPERTOIRE_MIN_CHANNELS,
                FIXED_LENGTH_S=base.FIXED_LENGTH_S, ANALYSIS_WINDOW=base.ANALYSIS_WINDOW)
