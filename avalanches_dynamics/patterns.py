# -*- coding: utf-8 -*-
"""
Repertorio delle valanghe: quali pattern spaziali ci sono, quanto sono diversi e quali canali si
attivano insieme. Funzioni generali: ricevono matrici di pattern (valanghe x canali, 0/1).

Pattern di una valanga = canali attivi almeno una volta durante la valanga (Sorrentino et al. 2021).

  pattern_frequencies : pattern unici ordinati per frequenza (per un soggetto o un gruppo)
  group_frequencies   : frequenza media tra i soggetti (ogni soggetto pesa uguale)
  hamming_mds         : mappa 2D dei pattern unici, vicini se differiscono in pochi canali
                        (MDS classico sulla distanza di Hamming; Torgerson 1952)
  pattern_pca         : PCA dei pattern -> componenti = modi di co-attivazione dei canali
  participation_ratio : dimensionalita' del repertorio di un soggetto (Gao et al. 2017)

Riferimenti
  Sorrentino et al. (2021) Sci Rep 11:4051. doi:10.1038/s41598-021-83425-4
  Torgerson (1952) Psychometrika 17:401-419. doi:10.1007/BF02288916
  Gao, Trautmann, Yu et al. (2017) bioRxiv 214262 (participation ratio). doi:10.1101/214262
"""
import numpy as np


def pattern_frequencies(patterns):
    """
    Pattern unici di una matrice (n_valanghe, n_canali) bool, ordinati dal piu' frequente.
    Ritorna (unici (n_unici, n_canali) bool, conteggi, frequenze relative).
    """
    p = np.asarray(patterns, dtype=bool)
    p = p[p.any(1)]
    if not len(p):
        return np.zeros((0, np.shape(patterns)[1]), bool), np.array([], int), np.array([])
    u, c = np.unique(p, axis=0, return_counts=True)
    order = np.argsort(-c, kind="stable")
    return u[order], c[order], c[order] / c.sum()


def group_frequencies(pattern_list):
    """
    Frequenza dei pattern in un gruppo come media tra i soggetti delle frequenze relative di ogni
    soggetto (un soggetto con molte valanghe non domina). pattern_list: una matrice per soggetto.
    Ritorna (unici, frequenza media, numero di soggetti in cui il pattern compare), ordinati.
    """
    freqs = {}
    n_sub = len(pattern_list)
    for p in pattern_list:
        u, _, f = pattern_frequencies(p)
        for row, fr in zip(u, f):
            key = row.tobytes()
            acc = freqs.setdefault(key, [row, 0.0, 0])
            acc[1] += fr / n_sub
            acc[2] += 1
    if not freqs:
        return np.zeros((0, 0), bool), np.array([]), np.array([], int)
    rows, f, n = zip(*freqs.values())
    order = np.argsort(-np.array(f), kind="stable")
    return np.array(rows)[order], np.array(f)[order], np.array(n)[order]


def hamming_mds(patterns, n_dims=2):
    """
    MDS classico (Torgerson) dei pattern sulla distanza di Hamming normalizzata (canali diversi /
    n_canali): pattern simili finiscono vicini. Ritorna (coordinate (n, n_dims), varianza spiegata
    dalle prime n_dims dimensioni).
    """
    r = np.asarray(patterns, dtype=float)
    n, n_ch = r.shape
    if n < 3:
        return np.zeros((n, n_dims)), np.nan
    ones = r.sum(1)
    d = (ones[:, None] + ones[None, :] - 2 * r @ r.T) / n_ch
    J = np.eye(n) - 1.0 / n
    B = -0.5 * J @ (d ** 2) @ J
    w, v = np.linalg.eigh(B)
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    pos = np.clip(w[:n_dims], 0, None)
    expl = pos.sum() / np.clip(w, 0, None).sum() if np.clip(w, 0, None).sum() > 0 else np.nan
    return v[:, :n_dims] * np.sqrt(pos), float(expl)


def pattern_pca(patterns, n_components=5):
    """
    PCA dei pattern (valanghe x canali, 0/1, centrati sulla media). Ogni componente e' un vettore di
    pesi sui canali: canali con lo stesso segno tendono ad attivarsi nella stessa valanga.
    Il segno di ogni componente e' fissato in modo che la somma dei pesi sia positiva.
    Ritorna dict(components (k, n_ch), explained_ratio (k,), mean (n_ch,), eigenvalues (n_ch,)).
    """
    x = np.asarray(patterns, dtype=float)
    mean = x.mean(0)
    xc = x - mean
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    ev = s ** 2 / max(len(x) - 1, 1)
    comps = vt[:n_components].copy()
    comps *= np.where(comps.sum(1) < 0, -1, 1)[:, None]
    return dict(components=comps, explained_ratio=ev[:n_components] / ev.sum(), mean=mean,
                eigenvalues=ev)


def explained_by(patterns, pca):
    """Frazione della varianza dei pattern di un soggetto spiegata da ogni componente di una PCA comune."""
    x = np.asarray(patterns, dtype=float)
    if len(x) < 2:
        return np.full(len(pca["components"]), np.nan)
    xc = x - x.mean(0)
    tot = (xc ** 2).sum()
    return ((xc @ pca["components"].T) ** 2).sum(0) / tot if tot > 0 else np.full(len(pca["components"]), np.nan)


def participation_ratio(patterns):
    """
    Dimensionalita' effettiva del repertorio: PR = (sum lambda)^2 / sum lambda^2 sugli autovalori
    della covarianza dei pattern. 1 = tutti i pattern variano lungo un solo asse; n_canali = la
    varianza e' distribuita uniformemente su tutte le direzioni.
    """
    x = np.asarray(patterns, dtype=float)
    if len(x) < 3:
        return np.nan
    ev = np.clip(np.linalg.eigvalsh(np.cov(x, rowvar=False)), 0, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum()) if (ev ** 2).sum() > 0 else np.nan


def balanced_sample(arrays, max_per_item, rng=0):
    """Concatena righe da piu' matrici prendendone al massimo max_per_item a caso da ciascuna."""
    rng = np.random.default_rng(rng)
    out = []
    for a in arrays:
        a = np.asarray(a)
        if len(a) > max_per_item:
            a = a[rng.choice(len(a), max_per_item, replace=False)]
        out.append(a)
    return np.concatenate(out) if out else np.zeros((0, 0))
