# -*- coding: utf-8 -*-
"""
Edge-of-chaos and complexity indexes. Generic functions on numpy arrays (no database code).

| function                         | measure                       | library                       |
|----------------------------------|-------------------------------|-------------------------------|
| compute_01_chaos_test            | K of the modified 0-1 test    | edgeofpy (or vectorized port) |
| compute_lyapunov_exponent        | lambda_max (Dahmen 2019)      | edgeofpy                      |
| compute_lempel_ziv_complexity    | LZC (LZ76, normalized)        | neurokit2                     |
| compute_spectral_slope           | aperiodic exponent beta       | specparam (fallback fooof)    |
| compute_katz_fd                  | Katz fractal dimension        | neurokit2                     |
| compute_renyi_entropy            | Renyi entropy H_alpha         | neurokit2                     |
| compute_multiscale_entropy       | MSE curve and area            | neurokit2                     |
| compute_entropy_production       | entropy production (Lynn 2021)| direct implementation         |
| compute_dfa                      | DFA exponent alpha            | neurokit2                     |
| compute_fei_ratio                | fE/I (Bruining 2020)          | port of the NBT/crosci code   |
| compute_covariance_width         | width of the covariance matrix| edgeofpy                      |

NOTE on lambda_max: edgeofpy.lambda_max is NOT the largest Lyapunov exponent of the time series.
It is the largest eigenvalue of the effective connectivity estimated from the dispersion of the
(integrated, time-lagged) covariances across channel pairs (Dahmen et al. 2019, PNAS):
    lambda_max = sqrt(1 - sqrt(1 / (1 + N * width^2))),   width = std(C) / mean(diag(C))
lambda_max -> 1 is the edge of chaos (linear stability boundary of the network). The classical
largest Lyapunov exponent of one channel (Rosenstein 1993) is available with method="rosenstein".

Input conventions: x = one channel (1-D array); data = (n_epochs, n_channels, n_times).
"""
import warnings

import numpy as np
from scipy import stats
from scipy.cluster.vq import kmeans2, vq
from scipy.signal import butter, sosfiltfilt

from synchrony_criticality.phase import bandpass_filter, band_phases, trim_edges


def _nk():
    import neurokit2 as nk
    if not hasattr(np, "trapezoid"):          # neurokit2 >= 0.2.13 calls np.trapezoid (numpy >= 2.0)
        np.trapezoid = np.trapz
    return nk


# ================================================================== SIGNAL HELPERS
def lowpass_filter(data, sfreq, cutoff, order=4):
    """Zero-phase Butterworth low-pass along the last axis."""
    sos = butter(order, min(cutoff, 0.99 * sfreq / 2), btype="lowpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def broadband(data, sfreq, band, order=4, trim_s=1.0):
    """Band-passed (e.g. 1-45 Hz) and edge-trimmed signal, epoch by epoch."""
    return trim_edges(bandpass_filter(data, sfreq, band, order), sfreq, trim_s)


def band_envelope(data, sfreq, band, order=4, trim_s=1.0):
    """Hilbert amplitude envelope of the band-passed signal, edge-trimmed. Same shape as data."""
    return band_phases(data, sfreq, band, order, trim_s)[1]


def concat_epochs(data):
    """(n_epochs, n_channels, n_times) -> (n_channels, n_epochs * n_times)."""
    return np.concatenate(list(data), axis=-1)


# ================================================================== 0-1 CHAOS TEST
def minmax_downsample(x):
    """Keeps only the local minima and maxima of x (Toker et al. 2022)."""
    from scipy.signal import argrelextrema
    idx = np.sort(np.concatenate([argrelextrema(x, np.less)[0], argrelextrema(x, np.greater)[0]]))
    return x[idx]


def z1_chaos_test_fast(x, sigma=0.5, n_c=1000, seed=0, batch=100):
    """
    Modified 0-1 test (Gottwald & Melbourne 2004, 2009; Dawes & Freeland 2008), same algorithm
    as edgeofpy.chaos.z1_chaos_test, vectorized: the mean-square displacement of the translation
    variables z(n) = p(n) + i q(n) = sum_j x_j exp(i j c) is computed for all lags at once,
        D(n) = 1/(N-n) sum_k |z_{k+n} - z_k|^2
             = 1/(N-n) [ sum_{k>=n} |z_k|^2 + sum_{k<N-n} |z_k|^2 - 2 Re sum_k z_{k+n} conj(z_k) ]
    (cross term by FFT autocorrelation). M(n) = D(n) - mean(x)^2 (1 - cos nc) / (1 - cos c)
    + sigma * U(-1/2, 1/2); K_c = corr(n, M(n)) for n = 1..N/10; K = median over n_c values of c.
    K ~ 0 regular, K ~ 1 chaotic. Only the random numbers differ from edgeofpy.
    """
    x = np.asarray(x, float)
    N = x.size
    n_cut = int(round(N / 10))
    rng = np.random.default_rng(seed)
    c = np.pi / 5 + rng.random(n_c) * 3 * np.pi / 5
    j = np.arange(1, N + 1)
    n = np.arange(1, n_cut + 1)
    L = 1 << int(np.ceil(np.log2(2 * N)))
    k = np.empty(n_c)
    for s in range(0, n_c, batch):
        cb = c[s:s + batch, None]
        z = np.cumsum(x * np.exp(1j * j * cb), axis=1)
        cs = np.concatenate([np.zeros((z.shape[0], 1)), np.cumsum(np.abs(z) ** 2, axis=1)], axis=1)
        F = np.fft.fft(z, L, axis=1)
        ac = np.fft.ifft(F * np.conj(F), axis=1)[:, n].real      # sum_k z_{k+n} conj(z_k)
        D = (cs[:, [N]] - cs[:, n] + cs[:, N - n] - 2 * ac) / (N - n)
        M = (D - x.mean() ** 2 * (1 - np.cos(n * cb)) / (1 - np.cos(cb))
             + sigma * (rng.random(D.shape) - 0.5))
        Mc = M - M.mean(axis=1, keepdims=True)
        nc = n - n.mean()
        k[s:s + batch] = (Mc @ nc) / (np.sqrt((Mc ** 2).sum(axis=1)) * np.sqrt((nc ** 2).sum()))
    return float(np.median(k))


def compute_01_chaos_test(x, sfreq=None, lowpass_hz=None, sigma=0.5, n_c=1000, denoise=False,
                          downsample="minmax", impl="fast", seed=0):
    """
    K statistic of the modified 0-1 chaos test for one channel, with the Toker et al. (2022)
    pipeline: [low-pass] -> [Schreiber denoising] -> [min/max downsampling] -> SD = 0.5 -> test.
    Returns NaN if fewer than 20 samples are left. impl: "fast" or "edgeofpy" (only n_c = 1000).
    """
    x = np.asarray(x, float)
    if lowpass_hz is not None:
        x = lowpass_filter(x, sfreq, lowpass_hz)
    if denoise:
        from edgeofpy.chaos import schreiber_denoise
        x = schreiber_denoise(x)
    if downsample == "minmax":
        x = minmax_downsample(x)
    if x.size < 20 or np.std(x) == 0:
        return np.nan
    x = x * (0.5 / np.std(x))
    if impl == "edgeofpy":
        from edgeofpy.chaos import z1_chaos_test
        return float(z1_chaos_test(x, sigma=sigma, rand_seed=seed))
    return z1_chaos_test_fast(x, sigma=sigma, n_c=n_c, seed=seed)


# ================================================================== COVARIANCE WIDTH / LAMBDA_MAX
def compute_covariance_width(data):
    """
    Width of the distribution of the integrated time-lagged covariances (Dahmen et al. 2019),
    edgeofpy implementation: covariances across epochs (trials), C = integral over lags,
        width = std(C_ij over all i, j) / mean(C_ii).
    data: (n_epochs, n_channels, n_times), at least 2 epochs. Returns dict(width, cov).
    """
    from edgeofpy.chaos import lambda_max
    n_ep, n_ch, n_t = data.shape
    if n_ep < 2:
        return dict(width=np.nan, cov=np.full((n_ch, n_ch), np.nan))
    # trial length = n_t samples exactly (s_freq = 1): edgeofpy reshapes back into the epochs
    _, width, cov = lambda_max(concat_epochs(data).astype(float), 1, n_ch, trial_len_sec=n_t)
    return dict(width=float(width), cov=cov)


def lambda_from_width(width, n_units):
    """lambda_max = sqrt(1 - sqrt(1 / (1 + N width^2))) (Dahmen et al. 2019, eq. in edgeofpy)."""
    return float(np.sqrt(1 - np.sqrt(1 / (1 + n_units * width ** 2))))


def compute_lyapunov_exponent(data, n_units=None, width=None, method="dahmen", sfreq=None):
    """
    method="dahmen" (default, edgeofpy): data (n_epochs, n_channels, n_times) -> dict(lambda_max,
      width). n_units = N (None = number of channels); width = precomputed covariance width
      (compute_covariance_width), to avoid computing the covariances twice.
    method="rosenstein": largest Lyapunov exponent of ONE channel (neurokit2, Rosenstein 1993),
      data = 1-D signal -> dict(lle) in 1/sample (x sfreq for 1/s if sfreq is given).
    """
    if method == "rosenstein":
        lle, _ = _nk().complexity_lyapunov(np.asarray(data, float), method="rosenstein1993")
        return dict(lle=float(lle) * (sfreq or 1.0))
    if width is None:
        width = compute_covariance_width(data)["width"]
    n_units = data.shape[1] if n_units is None else n_units
    lam = lambda_from_width(width, n_units) if np.isfinite(width) else np.nan
    return dict(lambda_max=lam, width=float(width))


# ================================================================== COMPLEXITY / ENTROPY
def compute_lempel_ziv_complexity(x, symbolize="median"):
    """Normalized Lempel-Ziv complexity (LZ76) of the binarized signal: C(n) log2(n) / n."""
    lzc, _ = _nk().complexity_lempelziv(np.asarray(x, float), symbolize=symbolize)
    return float(lzc)


def compute_katz_fd(x):
    """Katz fractal dimension (Katz 1988)."""
    fd, _ = _nk().fractal_katz(np.asarray(x, float))
    return float(fd)


def compute_renyi_entropy(x, alpha=2, bins=16):
    """
    Renyi entropy of order alpha (nats) of the amplitude distribution, estimated with `bins`
    equal-width bins: H_alpha = log(sum p_i^alpha) / (1 - alpha). Maximum log(bins).
    """
    h, _ = _nk().entropy_renyi(np.asarray(x, float), alpha=alpha, symbolize=int(bins))
    return float(h)


def compute_multiscale_entropy(x, scales=tuple(range(1, 21)), dimension=2, tolerance=0.15,
                               method="MSEn"):
    """
    Multiscale (sample) entropy (Costa et al. 2002), neurokit2. tolerance in SD units of x
    (r = tolerance * SD, the same r at every scale). Returns dict(auc = neurokit2 MSE index,
    i.e. mean of the finite values over scales, curve = SampEn per scale, scales).
    """
    x = np.asarray(x, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auc, info = _nk().entropy_multiscale(x, scale=list(scales), dimension=dimension,
                                             tolerance=tolerance * np.std(x, ddof=1), method=method)
    curve = np.asarray(info["Value"], float)
    curve[~np.isfinite(curve)] = np.nan
    return dict(auc=float(auc), curve=curve, scales=np.asarray(info["Scale"]))


def _states(data, n_states, seed, max_fit=20000):
    """Discrete state of every sample. data: (n_epochs, n_channels, n_times) -> (n_epochs, n_times)."""
    n_ep, n_ch, n_t = data.shape
    X = data.transpose(0, 2, 1).reshape(-1, n_ch)
    X = (X - X.mean(0)) / X.std(0)
    if n_ch == 1:                                   # one channel: quantile bins
        edges = np.quantile(X[:, 0], np.linspace(0, 1, n_states + 1)[1:-1])
        return np.digitize(X[:, 0], edges).reshape(n_ep, n_t)
    rng = np.random.default_rng(seed)
    fit = X[rng.choice(len(X), min(max_fit, len(X)), replace=False)]
    centroids, _ = kmeans2(fit, n_states, minit="++", seed=seed)
    return vq(X, centroids)[0].reshape(n_ep, n_t)


def compute_entropy_production(data, sfreq, n_states=8, lag_s=0.01, pseudocount=0.5, seed=0):
    """
    Entropy production of the coarse-grained dynamics (Lynn et al. 2021, PNAS): the multichannel
    z-scored signal is clustered into n_states states (k-means; quantile bins for one channel),
    transitions i -> j at lag tau are counted inside each epoch, and
        EP = sum_ij P_ij log(P_ij / P_ji)          (nats per lag step, >= 0)
    EP = 0 for a reversible process (detailed balance); EP > 0 measures the broken time-reversal
    symmetry. Finite data bias EP upward: compare groups with the same n_states, lag and length.
    data: (n_epochs, n_channels, n_times), (n_channels, n_times) or 1-D.
    Returns dict(ep, ep_per_s = EP / tau, P = joint transition matrix).
    """
    data = np.asarray(data, float)
    if data.ndim == 1:
        data = data[None, None]
    elif data.ndim == 2:
        data = data[None]
    s = _states(data, n_states, seed)
    lag = max(1, int(round(lag_s * sfreq)))
    C = np.full((n_states, n_states), float(pseudocount))
    for e in range(s.shape[0]):
        np.add.at(C, (s[e, :-lag], s[e, lag:]), 1)
    P = C / C.sum()
    ep = float(np.sum(P * np.log(P / P.T)))
    return dict(ep=ep, ep_per_s=ep / (lag / sfreq), P=P)


# ================================================================== LONG-RANGE TEMPORAL CORRELATIONS
def dfa_windows(n_samples, sfreq, fit_s=(1.0, 10.0), n_scales=20):
    """Log-spaced window sizes (samples) in fit_s, upper limit <= n_samples / 10 (Hardstone 2012)."""
    lo = fit_s[0] * sfreq
    hi = min(fit_s[1] * sfreq, n_samples / 10)
    if hi <= lo:
        return np.array([], int)
    return np.unique(np.round(np.logspace(np.log10(lo), np.log10(hi), n_scales)).astype(int))


def compute_dfa(x, sfreq, fit_s=(1.0, 10.0), n_scales=20, overlap=True):
    """
    Detrended fluctuation analysis exponent (Peng et al. 1994), neurokit2; usually applied to the
    amplitude envelope of a band (Linkenkaer-Hansen et al. 2001). alpha = 0.5 uncorrelated,
    0.5 < alpha < 1 long-range temporal correlations (critical-like), 1.5 Brownian.
    Returns dict(alpha, windows (samples), fluctuations); alpha NaN if < 4 windows fit.
    """
    x = np.asarray(x, float)
    win = dfa_windows(x.size, sfreq, fit_s, n_scales)
    if win.size < 4:
        return dict(alpha=np.nan, windows=win, fluctuations=np.full(win.size, np.nan))
    alpha, info = _nk().fractal_dfa(x, scale=win, overlap=overlap)
    return dict(alpha=float(alpha), windows=win, fluctuations=np.asarray(info["Fluctuations"]).ravel())


def compute_fei_ratio(envelope, sfreq, window_s=5.0, overlap=0.8, dfa_alpha=None,
                      dfa_threshold=0.6):
    """
    Functional excitation/inhibition ratio (Bruining et al. 2020, Sci Rep), port of the NBT /
    crosci code. envelope: amplitude envelope of one band (1-D).
      profile S = cumsum(A - mean A); for every window (window_s, overlap):
        wAmp = mean amplitude in the window
        wnF  = RMS residual of the linear fit of S / wAmp (normalized fluctuation)
      fE/I = 1 - corr(wnF, wAmp)
    fE/I > 1 excitation-dominated, < 1 inhibition-dominated, ~1 balanced (critical). It is set to
    NaN when dfa_alpha <= dfa_threshold (no long-range correlations: the ratio is not defined).
    Returns dict(fei, w_amp, w_dnf).
    """
    a = np.asarray(envelope, float)
    win = int(round(window_s * sfreq))
    step = max(1, int(np.floor(win * (1 - overlap))))
    starts = np.arange(0, a.size - win + 1, step)
    nan = dict(fei=np.nan, w_amp=np.array([]), w_dnf=np.array([]))
    if starts.size < 3 or a.min() == a.max():
        return nan
    idx = starts[:, None] + np.arange(win)[None, :]
    profile = np.cumsum(a - a.mean())
    w_amp = a[idx].mean(axis=1)
    xs = profile[idx] / w_amp[:, None]
    t = np.arange(win)
    coef = np.polyfit(t, xs.T, 1)                                  # (2, n_windows)
    resid = xs - (coef[0][:, None] * t + coef[1][:, None])
    w_dnf = np.sqrt((resid ** 2).mean(axis=1))
    fei = 1 - stats.pearsonr(w_dnf, w_amp)[0]
    if dfa_alpha is not None and not (dfa_alpha > dfa_threshold):
        fei = np.nan
    return dict(fei=float(fei), w_amp=w_amp, w_dnf=w_dnf)


# ================================================================== APERIODIC EXPONENT
def _specparam_fit(freqs, psd, frange, mode):
    """specparam (2.x) group fit; raises if specparam is missing or its API differs."""
    from specparam import SpectralGroupModel
    from descriptive_analysis.spectral import FOOOF_DEFAULTS
    fg = SpectralGroupModel(**FOOOF_DEFAULTS, aperiodic_mode=mode, verbose=False)
    fg.fit(freqs, psd, list(frange))
    ap = np.asarray(fg.get_params("aperiodic_params")).reshape(psd.shape[0], -1)
    return dict(exponent=ap[:, -1], offset=ap[:, 0],
                r2=np.asarray(fg.get_params("r_squared"), float), method=f"specparam ({mode})")


def compute_spectral_slope(data, sfreq, frange=(2.0, 40.0), mode="fixed", segment_s=2.0,
                           fmax=45.0):
    """
    Aperiodic (1/f^beta) exponent per channel: Welch PSD of every epoch, mean over epochs,
    spectral parameterization (Donoghue et al. 2020). specparam if installed, otherwise fooof
    (descriptive_analysis.spectral.aperiodic_fit, which falls back to a robust log-log line).
    data: (n_epochs, n_channels, n_times) in uV. Returns dict(exponent, offset, r2 per channel,
    freqs, psd (mean over epochs), method).
    """
    from descriptive_analysis.spectral import compute_psd, aperiodic_fit
    freqs, psd = compute_psd(data, sfreq, fmin=1.0, fmax=fmax, segment_s=segment_s)
    psd = psd.mean(axis=0)
    try:
        out = _specparam_fit(freqs, psd, frange, mode)
    except Exception:
        fit = aperiodic_fit(psd, freqs, frange, mode=mode)
        out = dict(exponent=fit["exponent"], offset=fit["offset"], r2=fit["r2"], method=fit["method"])
    out.update(freqs=freqs, psd=psd)
    return out
