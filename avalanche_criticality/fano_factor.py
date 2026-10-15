# -*- coding: utf-8 -*-
"""
Fano factor della popolazione: variabilita' dell'attivita' rispetto alla sua media.

    FF = Var[A(t)] / E[A(t)]                    (Maschke et al. 2024, Methods; edgeofpy.fano_factor)

A(t) = numero di eventi (tutti i canali) nel time bin t, cioe' l'attivita' di popolazione
ottenuta dalla binarizzazione. FF = 1 per un processo di Poisson; nei modelli riverberanti FF cresce
avvicinandosi alla criticita' (Wilting & Priesemann 2019) ed e' atteso massimo al punto critico.

Curva di Fano (Teich et al. 1997): FF dei conteggi in finestre di lunghezza W crescente.
Per un processo senza memoria FF(W) resta ~1; correlazioni temporali lunghe la fanno crescere
come FF(W) ~ W^beta (beta = pendenza in log-log).

Riferimenti
  Fano (1947) Phys Rev 72(1):26-29. doi:10.1103/PhysRev.72.26
  Teich, Heneghan, Lowen, Ozaki & Kaplan (1997) J Opt Soc Am A 14(3):529-546. doi:10.1364/JOSAA.14.000529
  Wilting & Priesemann (2019) Cereb Cortex 29(6):2759-2770. doi:10.1093/cercor/bhz049
  Maschke et al. (2024) Commun Biol 7:946. doi:10.1038/s42003-024-06613-8
"""
import numpy as np
from scipy import stats

from .binarization import bin_counts


def _as_list(events):
    if isinstance(events, np.ndarray) and events.ndim == 1:
        return [events]
    return list(events)


def fano_factor(events, bin_len):
    """
    FF dell'attivita' di popolazione con time bin di bin_len campioni.
    events: eventi per campione (n_times,) o lista per epoca (i bin non attraversano le epoche;
    l'ultimo bin incompleto di ogni epoca e' scartato). Ritorna (FF, media, varianza).
    """
    a = np.concatenate([bin_counts(ev[:len(ev) // bin_len * bin_len], bin_len)
                        for ev in _as_list(events) if len(ev) >= bin_len] or [np.array([])])
    if a.size < 2 or a.mean() == 0:
        return np.nan, float(a.mean()) if a.size else np.nan, np.nan
    return float(a.var() / a.mean()), float(a.mean()), float(a.var())


def fano_curve(events, sfreq, windows_s=None, n_windows=15, min_count=10):
    """
    FF(W) per finestre log-spaziate da ~10 ms a 1/10 della durata dell'epoca piu' corta
    (almeno min_count finestre per stimare la varianza). Ritorna dict(windows_s, ff, beta, beta_se):
    beta = pendenza OLS di log10 FF contro log10 W (solo FF > 0).
    """
    ev = _as_list(events)
    shortest = min(len(e) for e in ev) / sfreq
    if windows_s is None:
        hi = shortest / min_count
        windows_s = np.logspace(np.log10(0.01), np.log10(hi), n_windows) if hi > 0.01 else np.array([])
    ws, ff = [], []
    for w in np.unique(np.round(np.asarray(windows_s) * sfreq).astype(int)):
        if w < 1:
            continue
        f, _, _ = fano_factor(ev, w)
        ws.append(w / sfreq)
        ff.append(f)
    ws, ff = np.array(ws), np.array(ff)
    ok = np.isfinite(ff) & (ff > 0)
    beta = beta_se = np.nan
    if ok.sum() >= 3:
        lr = stats.linregress(np.log10(ws[ok]), np.log10(ff[ok]))
        beta, beta_se = float(lr.slope), float(lr.stderr)
    return dict(windows_s=ws, ff=ff, beta=beta, beta_se=beta_se)
