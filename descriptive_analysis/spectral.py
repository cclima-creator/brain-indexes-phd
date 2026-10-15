# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 14:11:18 2026

@author: Chiara Clima
"""
"""
PSD
 ↓
integrazione nelle bande
 ↓
potenza assoluta
 ↓
potenza relativa
 ↓
log-potenza
 ↓
tabella pandas
"""


"""Spettro di potenza, potenza per banda ed esponente aperiodico (1/f). Funzioni pure su array numpy."""
import numpy as np                      ##array e matematica
import pandas as pd                     ##tabelle e risultati
from scipy.signal import welch          ## welch = metodo per stimare power spectral density
from scipy.integrate import trapezoid    ## integrali


def compute_psd(data_uv, sfreq, fmin=1.0, fmax=45.0, segment_s=None, overlap=0.5,
                window="hann"):
    """
    PSD di Welch per epoca e canale.

    data_uv : (n_epochs, n_channels, n_times) in uV               ##sono i dati in microvolt del codice di importazione
    segment_s : None = un solo segmento = intera epoca (risoluzione 1/T Hz, le frequenze
                intere cadono su un bin se T e' intero); altrimenti durata del segmento
                in secondi (piu' media, meno risoluzione).
    Ritorna freqs (n_f,), psd (n_epochs, n_channels, n_f) in uV^2/Hz.
    """
    n_t = data_uv.shape[-1]
    if segment_s is None:                             ##segment none= uso intera epoca come segmento
        nperseg, noverlap = n_t, 0
    else:
        nperseg = min(int(round(segment_s * sfreq)), n_t)
        noverlap = int(nperseg * overlap)
    freqs, psd = welch(data_uv, fs=sfreq, window=window, nperseg=nperseg,
                       noverlap=noverlap, axis=-1)
    m = (freqs >= fmin) & (freqs <= min(fmax, sfreq / 2))
    return freqs[m], psd[..., m]                 ##ho stima discreta!!


def integrate_band(psd, freqs, lo, hi):
    """Integrale della PSD tra lo (incluso) e hi (escluso). psd: (..., n_f)."""
    m = (freqs >= lo) & (freqs < hi)
    return trapezoid(psd[..., m], freqs[m], axis=-1)        ##rende continua integrando con trapezoid


def band_power_table(psd, freqs, ch_names, bands, total_range=(1.0, 45.0)):
    """
    Potenza per banda e canale in formato 'long'.   ##formato long solo una formalita per dire che le metto in riga e non in colonna nella tabella
    psd : (n_channels, n_f). Colonne: channel, band, abs_uV2, rel, log10_abs.
    """
    tot = integrate_band(psd, freqs, total_range[0], total_range[1] + 1e-6)
    rows = []                           ##lista vuota per accumulare le righe della futura tabella.
    for band, (lo, hi) in bands.items():        ##per ogni banda di freq
        bp = integrate_band(psd, freqs, lo, hi)     ##integra psd
        for i, ch in enumerate(ch_names):
            rows.append(dict(channel=ch, band=band, abs_uV2=bp[i],
                             rel=bp[i] / tot[i], log10_abs=np.log10(bp[i])))
    return pd.DataFrame(rows)


FOOOF_DEFAULTS = dict(peak_width_limits=(1.0, 12.0), max_n_peaks=8, min_peak_height=0.05,
                      peak_threshold=2.0)


def _aperiodic_line_fit(psd, freqs, frange):
    """
    Riserva se FOOOF non e' installato: retta robusta in log-log in 2 passi (come il primo passo
    di FOOOF in modalita' 'fixed'): retta su tutti i punti, poi solo sui punti sotto la retta.
    """
    m = (freqs >= frange[0]) & (freqs <= frange[1])
    x = np.log10(freqs[m])
    n_ch = psd.shape[0]
    out = {k: np.full(n_ch, np.nan) for k in ("exponent", "offset", "r2")}
    fit = np.full((n_ch, m.sum()), np.nan)
    for c in range(n_ch):
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.log10(psd[c, m])
        ok = np.isfinite(y)
        if ok.sum() < 5:
            continue
        b1, b0 = np.polyfit(x[ok], y[ok], 1)
        keep = ok & (y - (b0 + b1 * x) <= 0)
        if keep.sum() < 5:
            keep = ok
        b1, b0 = np.polyfit(x[keep], y[keep], 1)
        res = y[keep] - (b0 + b1 * x[keep])
        ss = ((y[keep] - y[keep].mean()) ** 2).sum()
        out["exponent"][c], out["offset"][c] = -b1, b0
        out["r2"][c] = 1 - (res ** 2).sum() / ss if ss > 0 else np.nan
        fit[c] = b0 + b1 * x
    nan = np.full(n_ch, np.nan)
    out.update(knee=nan.copy(), knee_hz=nan.copy(), error=nan.copy(), n_peaks=nan.copy(),
               alpha_cf=nan.copy(), alpha_pw=nan.copy(), fit_freqs=freqs[m], ap_fit=fit,
               method="line (FOOOF not installed)")
    return out


def aperiodic_fit(psd, freqs, frange=(2.0, 40.0), mode="knee", settings=None,
                  alpha_band=(8.0, 13.0)):
    """
    Parametrizzazione dello spettro con FOOOF (Donoghue et al., Nat Neurosci 2020), per canale.
    psd: (n_channels, n_f) in uV^2/Hz. Lo spettro (in log10) e' modellato come
        componente aperiodica + picchi gaussiani
    componente aperiodica:
        mode="fixed": log10 P = offset - log10(f^exponent)            (una retta in log-log)
        mode="knee" : log10 P = offset - log10(knee + f^exponent)     (piatta sotto il knee,
                      1/f^exponent sopra); frequenza del knee = knee^(1/exponent) Hz
    Ritorna dict di array per canale: exponent, offset, knee, knee_hz (NaN se non c'e' knee),
    r2 ed error (bonta' del fit completo), n_peaks, alpha_cf / alpha_pw (frequenza e potenza
    sopra il fondo 1/f del picco piu' alto in alpha_band: frequenza alfa individuale),
    fit_freqs e ap_fit (componente aperiodica fittata, log10, per i grafici), method.
    Senza FOOOF installato usa una retta robusta (_aperiodic_line_fit) e avvisa.
    """
    try:
        import warnings
        with warnings.catch_warnings(record=True):        # fooof 1.1 forza un avviso di
            from fooof import FOOOFGroup                    # deprecazione: lo si intercetta
    except ImportError:
        print("    ATTENZIONE: fooof non installato (pip install fooof): uso una retta in log-log")
        return _aperiodic_line_fit(psd, freqs, frange)

    fg = FOOOFGroup(**{**FOOOF_DEFAULTS, **(settings or {})}, aperiodic_mode=mode, verbose=False)
    fg.fit(freqs, psd, list(frange))
    ap = fg.get_params("aperiodic_params")
    ap = ap.reshape(psd.shape[0], -1)
    offset, exponent = ap[:, 0], ap[:, -1]
    knee = ap[:, 1] if mode == "knee" else np.zeros(psd.shape[0])
    with np.errstate(invalid="ignore", divide="ignore"):
        knee_hz = np.where(knee > 0, np.abs(knee) ** (1 / exponent), np.nan)
    n_ch = psd.shape[0]
    alpha_cf, alpha_pw, n_peaks = np.full(n_ch, np.nan), np.full(n_ch, np.nan), np.zeros(n_ch)
    pk = np.asarray(fg.get_params("peak_params")).reshape(-1, 4)    # CF, PW, BW, canale
    for cf, pw, _, c in pk:
        if not np.isfinite(cf):
            continue
        c = int(c)
        n_peaks[c] += 1
        if alpha_band[0] <= cf <= alpha_band[1] and not (pw <= np.nan_to_num(alpha_pw[c], nan=-np.inf)):
            alpha_cf[c], alpha_pw[c] = cf, pw
    f = fg.freqs
    with np.errstate(invalid="ignore", divide="ignore"):
        ap_fit = offset[:, None] - np.log10(knee[:, None] + f[None, :] ** exponent[:, None])
    return dict(exponent=exponent, offset=offset, knee=knee, knee_hz=knee_hz,
                r2=np.asarray(fg.get_params("r_squared"), float),
                error=np.asarray(fg.get_params("error"), float), n_peaks=n_peaks,
                alpha_cf=alpha_cf, alpha_pw=alpha_pw, fit_freqs=f, ap_fit=ap_fit,
                method=f"FOOOF ({mode})")

def psd_leave_one_out(P, freqs, subjects, bands=None, z_thresh=3.0, min_db=0.5):
    """
    Contributo Leave-One-Out (LOO) di ogni soggetto alla PSD media di un gruppo (come in fig1:
    media delle PSD lineari).
    P: (n_soggetti, n_f) PSD in uV^2/Hz sulla stessa griglia di frequenze.
    Per il soggetto i: delta_i(f) = 10*log10( media_tutti(f) / media_senza_i(f) ) in dB.
      > 0: il soggetto alza la media a quella frequenza; < 0: la abbassa; 0: nessun effetto.
    Riassunti per soggetto: media e massimo di |delta| su tutte le frequenze, frequenza del massimo,
    delta medio in ogni banda. 'influential' = media |delta| oltre mediana + z_thresh * MAD robusta
    del gruppo (soggetto che pesa sulla media molto piu' degli altri) E almeno min_db dB
    (cosi' quando tutti sono simili uno scarto di pochi centesimi di dB non viene segnalato).
    Ritorna (delta (n_soggetti, n_f), tabella per soggetto). Servono almeno 2 soggetti.
    """
    P = np.asarray(P, float)
    n = P.shape[0]
    if n < 2:
        return np.full_like(P, np.nan), pd.DataFrame()
    full = P.mean(0)
    loo = (P.sum(0)[None, :] - P) / (n - 1)                  # media senza il soggetto i
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = 10 * np.log10(full[None, :] / loo)
    ad = np.abs(delta)
    rows = []
    for i, s in enumerate(subjects):
        k = int(np.nanargmax(ad[i])) if np.isfinite(ad[i]).any() else 0
        r = dict(subject=s, mean_abs_change_dB=float(np.nanmean(ad[i])),
                 max_abs_change_dB=float(np.nanmax(ad[i])), freq_of_max_Hz=float(freqs[k]),
                 signed_change_at_max_dB=float(delta[i, k]))
        for b, (lo, hi) in (bands or {}).items():
            m = (freqs >= lo) & (freqs < hi)
            r[f"change_{b}_dB"] = float(np.nanmean(delta[i, m])) if m.any() else np.nan
        rows.append(r)
    t = pd.DataFrame(rows)
    med = t["mean_abs_change_dB"].median()
    mad = 1.4826 * (t["mean_abs_change_dB"] - med).abs().median()
    t["robust_z"] = (t["mean_abs_change_dB"] - med) / mad if mad > 0 else 0.0
    t["influential"] = (t["robust_z"] > z_thresh) & (t["mean_abs_change_dB"] >= min_db)
    return delta, t.sort_values("mean_abs_change_dB", ascending=False).reset_index(drop=True)


def group_psd_matrix(store, group, common_f):
    """PSD (media su epoche e canali) dei soggetti di un gruppo, interpolate su common_f."""
    items = [s for s in store if s["group"] == group]
    P = np.array([np.interp(common_f, s["f"], s["psd"]) for s in items]) if items else \
        np.empty((0, len(common_f)))
    return P, [s["subject"] for s in items]


"""
long:
O1 delta
O1 theta
O1 alpha
O1 beta



wide:
O1 | delta | theta | alpha | beta
"""