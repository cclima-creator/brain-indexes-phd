# -*- coding: utf-8 -*-
"""
Branching ratio (branching parameter) sigma: quanti eventi, in media, segue un evento nel time bin successivo.
  sigma < 1 subcritico (l'attivita' si spegne), sigma = 1 critico, sigma > 1 supercritico.

Tre stimatori, tutti calcolati sugli eventi per campione prodotti da binarization.py:
  - "bins" (Maschke et al. 2024, edgeofpy.branching_ratio; Beggs & Plenz 2003):
        sigma = < n(t+1) / n(t) >  media su tutti i bin t con n(t) > 0
  - "ancestors" (Beggs & Plenz 2003, definizione originale):
        per ogni valanga lunga almeno 2 bin, discendenti / antenati = n(2° bin) / n(1° bin);
        sigma = media su tutte le valanghe
  - "geometric" (Sorrentino et al. 2021, Methods):
        sigma_i = prod_{j=1}^{N_bin-1} ( n_{j+1} / n_j )^(1 / (N_bin - 1))   per la valanga i
        sigma   = prod_{i=1}^{N_aval} sigma_i^(1 / N_aval)                  (media geometrica)
        (edgeofpy.avl_branching_ratio ha in piu' i fattori 1/(N_bin-1) e 1/N, segnalati come bug
        nel loro codice: qui si usa la formula dell'articolo)
sigma dipende molto dalla durata del time bin: branching_ratio_sweep() la ripete per piu' bin
(4-18 ms come nell'analisi preliminare; Maschke usa 1-12 ms).

Riferimenti
  Beggs & Plenz (2003) J Neurosci 23(35):11167-77. doi:10.1523/JNEUROSCI.23-35-11167.2003
  Sorrentino et al. (2021) Sci Rep 11:4051. doi:10.1038/s41598-021-83425-4
  Maschke et al. (2024) Commun Biol 7:946. doi:10.1038/s42003-024-06613-8
  Wilting & Priesemann (2018) Nat Commun 9:2325 (bias per sottocampionamento: sigma di
  questi stimatori classici e' sottostimato se si osserva solo una parte del sistema, come in EEG)
"""
import numpy as np

from .binarization import bin_counts


def _as_list(events):
    """Accetta un vettore (n_times,) o una lista di vettori (uno per epoca)."""
    if isinstance(events, np.ndarray) and events.ndim == 1:
        return [events]
    return list(events)


def avalanche_profiles(counts, bin_len):
    """
    Profili delle valanghe (eventi per bin) da una serie di eventi per campione, con lo stesso
    criterio di binarization.detect_avalanches (eventi a distanza <= bin_len nella stessa valanga).
    Serve a ricalcolare le valanghe per ogni time bin senza rifare la binarizzazione.
    Le valanghe che toccano l'inizio o la fine della serie sono scartate.
    """
    counts = np.asarray(counts)
    idx = np.flatnonzero(counts)
    if idx.size == 0:
        return []
    br = np.flatnonzero(np.diff(idx) > bin_len)
    starts, ends = np.r_[idx[0], idx[br + 1]], np.r_[idx[br], idx[-1]]
    return [bin_counts(counts[s:e + 1], bin_len) for s, e in zip(starts, ends)
            if s > 0 and e < len(counts) - 1]


def branching_ratio_bins(events, bin_len):
    """
    sigma = media di n(t+1)/n(t) sui bin con n(t) > 0 (Maschke 2024 / edgeofpy).
    events : eventi per campione (n_times,) o lista per epoca; i rapporti non attraversano le epoche.
    bin_len: durata del time bin in campioni.
    """
    ratios = []
    for ev in _as_list(events):
        n = bin_counts(ev, bin_len).astype(float)
        ok = n[:-1] > 0
        ratios.append(n[1:][ok] / n[:-1][ok])
    r = np.concatenate(ratios) if ratios else np.array([])
    return float(r.mean()) if r.size else np.nan


def branching_ratio_ancestors(profiles):
    """Beggs & Plenz 2003: media di n(2° bin) / n(1° bin) sulle valanghe con almeno 2 bin."""
    r = [p[1] / p[0] for p in profiles if len(p) > 1 and p[0] > 0]
    return float(np.mean(r)) if r else np.nan


def branching_ratio_geometric(profiles):
    """
    Sorrentino et al. 2021: media geometrica (su tutte le valanghe con almeno 2 bin) della media
    geometrica dei rapporti tra bin successivi dentro ogni valanga.
    Le valanghe costruite con il criterio dell'IEI non hanno bin interni vuoti, quindi i rapporti
    sono sempre > 0 (bin vuoti eventuali vengono saltati per sicurezza).
    """
    logs = []
    for p in profiles:
        p = np.asarray(p, dtype=float)
        if len(p) < 2:
            continue
        a, b = p[:-1], p[1:]
        ok = (a > 0) & (b > 0)
        if ok.any():
            logs.append(np.log(b[ok] / a[ok]).mean())     # log sigma_i
    return float(np.exp(np.mean(logs))) if logs else np.nan


def branching_ratio(events, sfreq, bin_s, profiles=None):
    """
    I tre stimatori di sigma per un time bin di durata bin_s (secondi).
    profiles: profili delle valanghe gia' calcolati con lo stesso bin (altrimenti ricalcolati).
    Ritorna dict(sigma_bins, sigma_ancestors, sigma_geometric, n_avalanches_multi_bin).
    """
    bin_len = max(1, int(round(bin_s * sfreq)))
    if profiles is None:
        profiles = [p for ev in _as_list(events) for p in avalanche_profiles(ev, bin_len)]
    return dict(sigma_bins=branching_ratio_bins(events, bin_len),
                sigma_ancestors=branching_ratio_ancestors(profiles),
                sigma_geometric=branching_ratio_geometric(profiles),
                n_avalanches_multi_bin=int(sum(len(p) > 1 for p in profiles)))


def branching_ratio_sweep(events, sfreq, bins_ms=(4, 6, 8, 10, 12, 14, 16, 18)):
    """
    sigma per ogni time bin in bins_ms (Wilting & Priesemann 2019; analisi preliminare 4-18 ms).
    Ritorna lista di dict (bin_ms + i campi di branching_ratio). Un bin piu' corto del periodo di
    campionamento viene arrotondato a 1 campione: controllare bin_len nei risultati.
    """
    rows = []
    for b in bins_ms:
        r = branching_ratio(events, sfreq, b / 1000.0)
        rows.append(dict(bin_ms=float(b), bin_len=max(1, int(round(b / 1000.0 * sfreq))), **r))
    return rows
