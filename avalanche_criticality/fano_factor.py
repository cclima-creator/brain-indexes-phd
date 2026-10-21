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


def fano_two_slopes(windows_s, ff, min_points=3, n_grid=200):
    """
    Two-regime fit of the Fano curve: continuous broken line in log-log ("broken stick"),
        y = a + b1 x                       for x < x0
        y = a + b1 x0 + b2 (x - x0)        for x >= x0          (x = log10 W, y = log10 FF)
    The break point x0 is searched on a grid between the min_points-th window from each end
    (at least min_points windows per segment); for each x0 the line is an ordinary least
    squares fit on the basis [1, x, max(x - x0, 0)], and the x0 with the smallest residual
    sum of squares is kept.
    Model comparison with the single line (fano_curve's beta): BIC with k = 2 (one line) and
    k = 4 (intercept, two slopes, break point); delta_bic = BIC_one - BIC_two > 0 favours two
    regimes (> 6: strong, > 10: very strong; Kass & Raftery 1995).
    Returns dict(beta_short, beta_long, break_s, intercept, delta_bic, r2_one, r2_two, beta_one).
    """
    w, f = np.asarray(windows_s, float), np.asarray(ff, float)
    ok = np.isfinite(f) & (f > 0) & (w > 0)
    x, y = np.log10(w[ok]), np.log10(f[ok])
    out = dict(beta_short=np.nan, beta_long=np.nan, break_s=np.nan, intercept=np.nan,
               delta_bic=np.nan, r2_one=np.nan, r2_two=np.nan, beta_one=np.nan)
    n = x.size
    if n < 2 * min_points:
        return out
    sst = float(((y - y.mean()) ** 2).sum())
    one = np.polyfit(x, y, 1)
    sse_one = float(((y - np.polyval(one, x)) ** 2).sum())
    best = None
    for x0 in np.linspace(x[min_points - 1], x[n - min_points], n_grid):
        A = np.column_stack([np.ones(n), x, np.clip(x - x0, 0, None)])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        sse = float(((y - A @ coef) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, x0, coef)
    sse_two, x0, (a, b1, b2) = best
    tiny = 1e-12                                   # avoids log(0) for a perfect fit
    bic = lambda sse, k: n * np.log(max(sse, tiny) / n) + k * np.log(n)
    out.update(beta_short=float(b1), beta_long=float(b1 + b2), break_s=float(10 ** x0),
               intercept=float(a), delta_bic=float(bic(sse_one, 2) - bic(sse_two, 4)),
               r2_one=1 - sse_one / sst if sst > 0 else np.nan,
               r2_two=1 - sse_two / sst if sst > 0 else np.nan, beta_one=float(one[0]))
    return out


def two_slope_curve(fit, windows_s):
    """FF predicted by a fano_two_slopes fit at windows_s (for the figures)."""
    x, x0 = np.log10(np.asarray(windows_s, float)), np.log10(fit["break_s"])
    y = fit["intercept"] + fit["beta_short"] * x + (fit["beta_long"] - fit["beta_short"]) * np.clip(x - x0, 0, None)
    return 10 ** y
