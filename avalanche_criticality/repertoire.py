# -*- coding: utf-8 -*-
"""
Repertorio delle valanghe: quanti pattern spaziali diversi e quanto si somigliano.

Pattern (configurazione) di una valanga = insieme dei canali attivi almeno una volta durante la
valanga, vettore binario di lunghezza n_canali (Sorrentino et al. 2021). L'ordine temporale dentro
la valanga si perde.

  - repertoire_size : numero di pattern unici (ogni pattern conta una volta sola, Sorrentino 2021)
  - repertoire_rate : pattern unici al secondo = repertoire_size / durata del segnale
                      (normalizzazione per la lunghezza del segnale di Maschke et al. 2024)
  - repertoire_per_avalanche : pattern unici / numero di valanghe (indipendente da quante valanghe ci
                      sono: utile perche' soglie diverse per gruppo producono numeri di valanghe diversi)
  - diversity       : mediana della distanza di Hamming normalizzata (canali diversi / n_canali)
                      tra tutte le coppie di pattern unici (Maschke 2024; Sorrentino 2021 usa la
                      distanza di Hamming tra pattern unici). Alta = pattern molto diversi
  - similarity      : 1 - diversity (alta = attivita' che ripete gli stessi gruppi di canali)

Riferimenti
  Sorrentino et al. (2021) Sci Rep 11:4051. doi:10.1038/s41598-021-83425-4
  Maschke et al. (2024) Commun Biol 7:946. doi:10.1038/s42003-024-06613-8
  edgeofpy: avl_repertoire, avl_pattern_dissimilarity (https://github.com/jnobyrne/edgeofpy)
"""
import numpy as np


def unique_patterns(patterns, min_channels=1):
    """
    Pattern unici e quante volte compare ciascuno.
    patterns : (n_avalanches, n_channels) bool. Pattern con meno di min_channels canali attivi
               sono esclusi (min_channels=1 toglie solo i pattern vuoti).
    Ritorna (unici (n_unique, n_channels) bool, conteggi (n_unique,)).
    """
    p = np.asarray(patterns, dtype=bool)
    if p.size == 0:
        return np.zeros((0, p.shape[1] if p.ndim == 2 else 0), bool), np.array([], int)
    p = p[p.sum(1) >= min_channels]
    if not len(p):
        return np.zeros((0, p.shape[1]), bool), np.array([], int)
    u, counts = np.unique(p, axis=0, return_counts=True)
    return u, counts


def hamming_matrix(rep, normalize=True):
    """
    Distanza di Hamming tra tutte le coppie di pattern (n, n): numero di canali in cui differiscono,
    divisa per n_canali se normalize. Stesso risultato di edgeofpy.avl_pattern_dissimilarity.
    """
    r = np.asarray(rep, dtype=np.int32)
    ones = r.sum(1)
    d = ones[:, None] + ones[None, :] - 2 * (r @ r.T)
    return d / r.shape[1] if normalize else d


def pattern_diversity(rep, max_patterns=4000, rng=0):
    """
    Mediana e media della distanza di Hamming normalizzata tra coppie di pattern unici (i < j).
    Con piu' di max_patterns pattern unici si usa un sottoinsieme casuale (la matrice n x n
    crescerebbe troppo); n_used dice quanti ne sono stati usati.
    Ritorna dict(diversity, diversity_mean, similarity, n_used).
    """
    rep = np.asarray(rep, dtype=bool)
    if len(rep) < 2:
        return dict(diversity=np.nan, diversity_mean=np.nan, similarity=np.nan, n_used=len(rep))
    if len(rep) > max_patterns:
        rep = rep[np.random.default_rng(rng).choice(len(rep), max_patterns, replace=False)]
    d = hamming_matrix(rep)[np.triu_indices(len(rep), 1)]
    med = float(np.median(d))
    return dict(diversity=med, diversity_mean=float(d.mean()), similarity=1 - med, n_used=len(rep))


def repertoire(patterns, duration_s, min_channels=1, max_patterns=4000, rng=0):
    """
    Tutte le misure di repertorio per un soggetto.
    patterns   : (n_avalanches, n_channels) bool (output di binarization.avalanches_from_epochs)
    duration_s : durata del segnale analizzato (s), per repertoire_rate.
    """
    rep, counts = unique_patterns(patterns, min_channels)
    n_av = int(counts.sum())
    out = dict(n_avalanches=n_av, repertoire_size=int(len(rep)),
               repertoire_rate=len(rep) / duration_s if duration_s else np.nan,
               repertoire_per_avalanche=len(rep) / n_av if n_av else np.nan,
               frac_patterns_once=float(np.mean(counts == 1)) if len(counts) else np.nan)
    out.update(pattern_diversity(rep, max_patterns, rng))
    return out, rep, counts


def shared_repertoire(rep_a, rep_b):
    """
    Pattern condivisi tra due repertori (es. due gruppi, Sorrentino 2021) e pattern specifici.
    I pattern devono avere gli stessi canali nello stesso ordine.
    Ritorna dict(n_shared, n_only_a, n_only_b, jaccard = condivisi / unione).
    """
    a = {r.tobytes() for r in np.asarray(rep_a, dtype=bool)}
    b = {r.tobytes() for r in np.asarray(rep_b, dtype=bool)}
    union = len(a | b)
    return dict(n_shared=len(a & b), n_only_a=len(a - b), n_only_b=len(b - a),
                jaccard=len(a & b) / union if union else np.nan)
