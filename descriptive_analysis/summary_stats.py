# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 14:11:51 2026

@author: Chiara Clima
"""

"""
Statistiche descrittive, fit gaussiano e confronti tra gruppi (su tabelle pandas).
Inoltre: numero di epoche, varianza, ACFW e stazionarieta' per epoca e canale,
confronto delle CDF tra gruppi (KS a permutazione) e test canale per canale.
"""
import numpy as np
import pandas as pd
from scipy import stats


def describe(x):
    x = np.asarray(x, dtype=float)                 ##converte x in numeri decimali
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(n=0, mean=np.nan, median=np.nan, sd=np.nan, skew=np.nan, kurtosis=np.nan)
    return dict(n=len(x), mean=x.mean(), median=np.median(x),
                sd=x.std(ddof=1) if len(x) > 1 else np.nan,
                skew=stats.skew(x) if len(x) > 2 else np.nan,           ##Asimmetria della distribuzione
                kurtosis=stats.kurtosis(x) if len(x) > 3 else np.nan)    ##importanza delle code


def gaussian_fit(x):
    """(mu, sigma) della gaussiana che meglio approssima x."""
    x = np.asarray(x, dtype=float)
    return stats.norm.fit(x[np.isfinite(x)])


def shapiro_p(x):                                ##how close to a nomal distribution
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return stats.shapiro(x).pvalue if len(x) >= 3 else np.nan


def band_group_summary(long, subj, group_col="group"):
    """
    Riassunto per gruppo e banda: media/mediana/sd (livello soggetto) della potenza
    assoluta, relativa e log10; fit gaussiano su log10 (canali x soggetti); skewness
    lineare vs log10; Shapiro sui soggetti (lineare vs log10).
    long: tabella per canale; subj: tabella per soggetto (media sui canali).
    """
    rows = []
    for (g, b), s in subj.groupby([group_col, "band"]):
        ch = long[(long[group_col] == g) & (long.band == b)]
        mu, sd = gaussian_fit(ch.log10_abs)
        a, r, lg = describe(s.abs_uV2), describe(s.rel), describe(s.log10_abs)
        rows.append(dict(
            group=g, band=b, n_subjects=a["n"],
            abs_mean=a["mean"], abs_median=a["median"], abs_sd=a["sd"],
            rel_mean=r["mean"], rel_median=r["median"], rel_sd=r["sd"],
            log10_mean=lg["mean"], log10_median=lg["median"], log10_sd=lg["sd"],
            gauss_mu_log10_channels=mu, gauss_sigma_log10_channels=sd,
            skew_channels_linear=stats.skew(ch.abs_uV2),
            skew_channels_log10=stats.skew(ch.log10_abs),
            shapiro_p_subjects_linear=shapiro_p(s.abs_uV2),
            shapiro_p_subjects_log10=shapiro_p(s.log10_abs)))
    return pd.DataFrame(rows)


def fdr_bh(p):
    """Correzione Benjamini-Hochberg dei p-value."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1.0)
    return out


def compare_groups(df, metrics, group_col="group"):
    """
    Confronto tra gruppi per ogni colonna in `metrics` (una riga per soggetto in df).
    2 gruppi: Mann-Whitney U; piu' gruppi: Kruskal-Wallis. Aggiunge p corretto (FDR-BH).
    """
    groups = list(df[group_col].dropna().unique())
    rows = []
    for m in metrics:
        samples = [df.loc[df[group_col] == g, m].dropna().values for g in groups]
        if len(groups) < 2 or any(len(s) < 2 for s in samples):
            continue
        if len(groups) == 2:
            test, p = "Mann-Whitney U", stats.mannwhitneyu(*samples).pvalue
        else:
            test, p = "Kruskal-Wallis", stats.kruskal(*samples).pvalue
        row = dict(metric=m, test=test, p=p)
        for g, s in zip(groups, samples):
            row[f"n_{g}"] = len(s)
            row[f"median_{g}"] = float(np.median(s))
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out["p_fdr"] = fdr_bh(out["p"].values)
    return out


# ================================================================== NUMERO DI EPOCHE
def epoch_counts(epoch_df, group_col="group"):
    """
    Epoche disponibili per soggetto: totale e per frequenza di stimolo (colonne 'n_<f>Hz';
    'n_no_stim' = epoche senza frequenza). epoch_df: una riga per epoca (group, subject, stim_freq).
    Ritorna (per_soggetto, per_gruppo). per_gruppo: n soggetti, media, SD, min, max, totale epoche.
    """
    df = epoch_df.copy()
    df["stim"] = [f"n_{f:g}Hz" if np.isfinite(f) else "n_no_stim" for f in df["stim_freq"]]
    per_subj = (df.pivot_table(index=[group_col, "subject"], columns="stim", values="epoch",
                               aggfunc="count", fill_value=0)
                .reset_index())
    per_subj.columns.name = None
    cols = sorted([c for c in per_subj.columns if c.startswith("n_") and c != "n_no_stim"],
                  key=lambda c: float(c[2:-2]))
    cols += ["n_no_stim"] if "n_no_stim" in per_subj.columns else []
    per_subj["n_epochs"] = per_subj[cols].sum(axis=1)
    per_subj = per_subj[[group_col, "subject", "n_epochs"] + cols]
    per_group = per_subj.groupby(group_col)["n_epochs"].agg(
        n_subjects="count", mean="mean", sd="std", min="min", max="max", total="sum").reset_index()
    return per_subj, per_group


# ================================================================== SEGNALE PER EPOCA E CANALE
def signal_variance(data_uv):
    """Varianza nel tempo di ogni epoca e canale (uV^2). data_uv: (n_ep, n_ch, n_t) -> (n_ep, n_ch)."""
    return data_uv.var(axis=-1)


def acf_width(data_uv, sfreq, max_lag_s=1.0):
    """
    Autocorrelation Function Width (ACFW) per epoca e canale, in ms.
    ACF normalizzata (lag 0 = 1) via FFT; ACFW = larghezza a meta' altezza (FWHM) =
    2 x primo lag in cui l'ACF scende sotto 0.5 (interpolazione lineare tra i due lag).
    ACF che decade lentamente (ACFW grande) = segnale con 'memoria' lunga / dinamica lenta.
    NaN se l'ACF non scende sotto 0.5 entro max_lag_s.
    """
    x = data_uv - data_uv.mean(-1, keepdims=True)
    n_t = x.shape[-1]
    nfft = 1 << int(np.ceil(np.log2(2 * n_t - 1)))
    spec = np.fft.rfft(x, nfft, axis=-1)
    acf = np.fft.irfft(spec * np.conj(spec), nfft, axis=-1)[..., :min(n_t, int(max_lag_s * sfreq) + 2)]
    with np.errstate(invalid="ignore", divide="ignore"):
        acf = acf / acf[..., :1]
    below = acf < 0.5
    first = np.where(below.any(-1), below.argmax(-1), -1)          # primo lag sotto 0.5
    out = np.full(first.shape, np.nan)
    ok = first > 0
    k = first[ok]
    a0 = np.take_along_axis(acf[ok], (k - 1)[:, None], 1)[:, 0]   # ACF al lag prima
    a1 = np.take_along_axis(acf[ok], k[:, None], 1)[:, 0]          # ACF al primo lag sotto 0.5
    lag = (k - 1) + (a0 - 0.5) / (a0 - a1)
    out[ok] = 2 * lag / sfreq * 1000
    return out


def stationarity_test(data_uv, n_segments=20):
    """
    Test di stazionarieta' per epoca e canale: Reverse Arrangement Test (Bendat & Piersol)
    sulla varianza di n_segments segmenti consecutivi dell'epoca.
    Se il segnale e' stazionario le varianze dei segmenti non hanno trend: il numero A di
    coppie (i<j) con var_i > var_j ha media N(N-1)/4 e varianza N(2N+5)(N-1)/72.
    z = (A - media)/SD, p bilaterale (approssimazione normale).
    Ritorna p (n_ep, n_ch): p < 0.05 = epoca/canale non stazionario (trend di ampiezza).
    Sotto stazionarieta' ci si aspetta circa il 5% di p < 0.05 per puro caso (un po' meno: A e'
    un numero intero, quindi i p possibili sono pochi; con 20 segmenti l'effetto e' piccolo).
    """
    n_ep, n_ch, n_t = data_uv.shape
    L = n_t // n_segments
    seg = data_uv[..., :L * n_segments].reshape(n_ep, n_ch, n_segments, L).var(-1)
    i, j = np.triu_indices(n_segments, 1)
    A = (seg[..., i] > seg[..., j]).sum(-1)
    N = n_segments
    mu, sd = N * (N - 1) / 4, np.sqrt(N * (2 * N + 5) * (N - 1) / 72)
    return 2 * stats.norm.sf(np.abs((A - mu) / sd))


def signal_tables(data_uv, sfreq, ch_names, labels, stim_freq, n_segments=20, max_lag_s=1.0,
                  alpha=0.05):
    """
    Varianza, ACFW e stazionarieta' di un soggetto.
    Ritorna (per_epoca_canale, per_canale, per_soggetto):
      per_epoca_canale: epoch, label, stim_freq, channel, var_uV2, log10_var, acfw_ms, stat_p
      per_canale: media sulle epoche + var_cv_epochs (SD/media della varianza tra epoche)
                  + frac_nonstationary (frazione di epoche con stat_p < alpha)
      per_soggetto: dict con le stesse misure mediate sui canali
    """
    v = signal_variance(data_uv)
    a = acf_width(data_uv, sfreq, max_lag_s)
    p = stationarity_test(data_uv, n_segments)
    n_ep, n_ch = v.shape
    with np.errstate(divide="ignore"):
        lv = np.log10(v)
    ec = pd.DataFrame(dict(
        epoch=np.repeat(np.arange(n_ep), n_ch), label=np.repeat(np.asarray(labels, object), n_ch),
        stim_freq=np.repeat(stim_freq, n_ch), channel=np.tile(ch_names, n_ep),
        var_uV2=v.ravel(), log10_var=lv.ravel(), acfw_ms=a.ravel(), stat_p=p.ravel()))
    with np.errstate(invalid="ignore", divide="ignore"):
        ch = pd.DataFrame(dict(
            channel=ch_names, var_uV2=v.mean(0), log10_var=np.nanmean(np.where(np.isfinite(lv), lv, np.nan), 0),
            var_cv_epochs=v.std(0, ddof=1) / v.mean(0) if n_ep > 1 else np.full(n_ch, np.nan),
            acfw_ms=np.nanmean(a, 0), frac_nonstationary=(p < alpha).mean(0)))
    subj = {c: float(np.nanmean(ch[c])) for c in
            ["var_uV2", "log10_var", "var_cv_epochs", "acfw_ms", "frac_nonstationary"]}
    return ec, ch, subj


# ================================================================== CDF TRA GRUPPI
def ks_permutation_test(df, value, subject_col="subject", group_col="group", n_perm=5000,
                        n_grid=2000, seed=0):
    """
    Confronto delle distribuzioni cumulative (CDF) tra 2 gruppi con Kolmogorov-Smirnov a permutazione.
    df: una riga per osservazione (es. epoca x canale) con soggetto e gruppo.
    - D = max |CDF_gruppoA - CDF_gruppoB| sui valori di tutte le osservazioni di ciascun gruppo
      (ogni soggetto pesa in proporzione alle sue osservazioni).
    - p: si rimescolano le etichette di gruppo tra i SOGGETTI (non tra le osservazioni, che dentro
      un soggetto non sono indipendenti) n_perm volte; p = (n. permutazioni con D >= D_oss + 1)/(n_perm + 1).
    Le CDF si valutano su una griglia di n_grid quantili dei dati (veloce e quasi esatto).
    Ritorna dict (vuoto se non ci sono esattamente 2 gruppi con >= 2 soggetti).
    """
    d = df[[subject_col, group_col, value]].replace([np.inf, -np.inf], np.nan).dropna()
    groups = list(pd.unique(d[group_col]))
    subj = d.groupby(subject_col)[group_col].first()
    if len(groups) != 2 or (subj.value_counts() < 2).any() or subj.nunique() != 2:
        return {}
    grid = np.unique(np.quantile(d[value].values, np.linspace(0, 1, n_grid)))
    names = list(subj.index)
    counts = np.array([(d[subject_col] == s).sum() for s in names], float)
    # numero di osservazioni <= ogni punto della griglia, per soggetto: (n_subj, n_grid)
    C = np.array([np.searchsorted(np.sort(d.loc[d[subject_col] == s, value].values), grid, "right")
                  for s in names], float)
    lab = (subj.values == groups[0])

    def D_of(mask):
        ca = C[mask].sum(0) / counts[mask].sum()
        cb = C[~mask].sum(0) / counts[~mask].sum()
        return np.abs(ca - cb).max()

    D = D_of(lab)
    rng = np.random.default_rng(seed)
    null = np.array([D_of(rng.permutation(lab)) for _ in range(n_perm)])
    return dict(metric=value, D=D, p_perm=(np.sum(null >= D) + 1) / (n_perm + 1), n_perm=n_perm,
                **{f"n_subj_{g}": int((subj == g).sum()) for g in groups},
                **{f"n_obs_{g}": int((d[group_col] == g).sum()) for g in groups},
                **{f"median_{g}": float(d.loc[d[group_col] == g, value].median()) for g in groups})


# ================================================================== PER CANALE
def channel_group_tests(chan, metrics, group_col="group"):
    """
    Test tra gruppi CANALE PER CANALE (per le mappe topografiche): per ogni misura e canale,
    Mann-Whitney / Kruskal-Wallis sui valori dei soggetti; p_fdr = FDR sui canali di quella misura
    (qui i canali sono davvero test multipli). chan: una riga per soggetto x canale.
    """
    out = []
    for m in metrics:
        rows = []
        for ch, d in chan.groupby("channel", sort=False):
            t = compare_groups(d, [m], group_col)
            if len(t):
                r = t.iloc[0].to_dict()
                r["channel"] = ch
                rows.append(r)
        if rows:
            t = pd.DataFrame(rows)
            t["p_fdr"] = fdr_bh(t["p"].values)
            out.append(t)
    if not out:
        return pd.DataFrame()
    t = pd.concat(out, ignore_index=True)
    return t[["metric", "channel"] + [c for c in t.columns if c not in ("metric", "channel")]]
