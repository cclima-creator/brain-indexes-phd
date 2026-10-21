# -*- coding: utf-8 -*-
"""
Sequenza delle valanghe: l'attivita' torna sempre agli stessi pattern (stati che "attirano") o
passa da un pattern all'altro senza memoria? Funzioni generali.

1. Stati. I pattern (valanghe x canali, 0/1) sono raggruppati in K stati con k-means, con centroidi
   comuni a tutti i soggetti (cosi' gli stati sono confrontabili tra soggetti e gruppi).
2. Matrice di transizione P[i, j] = P(stato della valanga n+1 = j | stato della valanga n = i), solo
   tra valanghe consecutive della stessa epoca (catena di Markov del primo ordine).
3. Memoria della sequenza: informazione mutua tra stati consecutivi
       I = H(S_n+1) - H(S_n+1 | S_n)   (bit)
   confrontata con le sequenze rimescolate dentro ogni epoca (stessi stati, ordine casuale):
   z = (I - media_null) / sd_null. z > 2: la valanga successiva dipende da quella precedente.
   self_transition: probabilita' media di restare nello stesso stato (diagonale di P).

Riferimenti
  Cover & Thomas (2006) Elements of Information Theory, 2nd ed., cap. 2 e 4 (informazione mutua,
    tasso di entropia di una catena di Markov)
  Rabuffo et al. (2021) eNeuro 8(5) (matrici di transizione delle valanghe nel cervello a riposo).
    doi:10.1523/ENEURO.0283-21.2021
"""
import numpy as np


def fit_states(patterns, k=8, rng=0):
    """Centroidi k-means comuni (scikit-learn) dei pattern, n_init=10. Ritorna il modello KMeans."""
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=k, n_init=10, random_state=rng).fit(np.asarray(patterns, float))


def sequences(states, epoch):
    """Stati in ordine temporale divisi per epoca (le epoche non sono contigue)."""
    states, epoch = np.asarray(states), np.asarray(epoch)
    return [states[epoch == e] for e in np.unique(epoch)]


def transition_counts(seqs, k):
    """Conteggi C[i, j] dei passaggi i -> j tra valanghe consecutive dentro ogni epoca."""
    C = np.zeros((k, k))
    for s in seqs:
        if len(s) > 1:
            np.add.at(C, (s[:-1], s[1:]), 1)
    return C


def _mutual_info(C):
    n = C.sum()
    if n == 0:
        return np.nan
    p = C / n
    pi, pj = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log2(p[nz] / (pi @ pj)[nz])).sum())


def sequence_memory(states, epoch, k, n_perm=200, rng=0):
    """
    Informazione mutua tra stati consecutivi e confronto con le sequenze rimescolate dentro le epoche.
    Ritorna dict(P (matrice di transizione, righe normalizzate), counts, mi_bits, mi_null_mean, mi_z,
    self_transition, n_transitions).
    """
    seqs = sequences(states, epoch)
    C = transition_counts(seqs, k)
    mi = _mutual_info(C)
    rng = np.random.default_rng(rng)
    null = np.array([_mutual_info(transition_counts([rng.permutation(s) for s in seqs], k))
                     for _ in range(n_perm)])
    with np.errstate(invalid="ignore", divide="ignore"):
        P = C / C.sum(1, keepdims=True)
    occupied = C.sum(1) > 0
    sd = null.std()
    return dict(P=P, counts=C, mi_bits=mi, mi_null_mean=float(null.mean()),
                mi_z=float((mi - null.mean()) / sd) if sd > 0 else np.nan,
                self_transition=float(np.nanmean(np.diag(P)[occupied])) if occupied.any() else np.nan,
                n_transitions=int(C.sum()))
