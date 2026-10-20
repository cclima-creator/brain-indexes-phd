# -*- coding: utf-8 -*-
"""
Default settings of the edge-of-chaos analysis (MAIN_chaos.py / plot_chaos.py / report_chaos.py).
No database-specific paths here: a database MAIN (e.g. autism/MAIN_AUT_CHAOS.py) imports this
module, overrides what it needs (cfg.BANDS = ..., cfg.MEASURES = [...]) and calls
MAIN_chaos.run(groups, out_dir, cfg).

Every measure is computed on every signal of SIGNAL_BANDS (broadband 1-45 Hz and each band);
columns are named <measure>_<band> (lzc_alpha, dfa_broadband, ...).
Measure -> where it is computed (functions in chaos_indexes.py):
  per channel, per epoch (then mean over epochs), band-passed signal:
      LZC, Katz FD, Renyi H2, MSE
  per channel, long signal (epochs concatenated if CONCATENATE_EPOCHS), see below:
      0-1 chaos test K (band-passed signal; for broadband also low-passed at K_LOWPASS_HZ),
      DFA alpha and fE/I (amplitude envelope of the signal)
  per channel, PSD averaged over epochs (whole spectrum, broadband only):
      aperiodic exponent beta (specparam, falls back to fooof)
  whole montage (one value per subject and signal):
      covariance width and lambda_max (Dahmen et al. 2019, edgeofpy), entropy production
"""

# ---- signal
SFREQ = None                  # None = keep the sampling rate of the data; a number = resample to it
                              # (Hz) before the analysis (MSE and the 0-1 test scale with N)
BROADBAND = (1.0, 45.0)       # band-pass of the "broadband" signal (Hz)
BANDS = dict(delta=(1.0, 4.0), theta=(4.0, 8.0), alpha=(8.0, 13.0), beta=(13.0, 30.0),
             gamma=(30.0, 45.0))
SIGNAL_BANDS = ["broadband", "delta", "theta", "alpha", "beta", "gamma"]   # signals analysed
FILTER_ORDER = 4              # Butterworth order (sos, forward-backward -> zero phase)
EDGE_TRIM_S = 1.0             # seconds removed at both ends of every filtered epoch
CONCATENATE_EPOCHS = True     # K, DFA, fE/I on the concatenation of the (trimmed) epochs.
                              # Right for continuous recordings cut in fixed-length epochs;
                              # for truly separate trials set False (computed per epoch, then mean)
MAX_EPOCHS = None             # None = all epochs; an int = only the first MAX_EPOCHS (quick runs)

# measures to compute (names of the functions in chaos_indexes.py without "compute_")
MEASURES = ["01_chaos_test", "lyapunov_exponent", "lempel_ziv_complexity", "spectral_slope",
            "katz_fd", "renyi_entropy", "multiscale_entropy", "entropy_production", "dfa",
            "fei_ratio", "covariance_width"]

# ---- 0-1 chaos test (Gottwald & Melbourne 2004, 2009; Toker et al. 2022 pipeline)
K_LOWPASS_HZ = (4.0, 8.0, 13.0)   # low-pass cutoffs: one K per cutoff (column k_lp<cutoff>).
                                  # Toker 2022 apply the test to slow (low-passed) activity; the
                                  # best cutoff is a methodological choice of the study
K_SIGMA = 0.5                 # noise added to the mean-square displacement
K_N_C = 1000                  # random frequencies c in (pi/5, 3pi/5); K = median over them
K_DENOISE = False             # Schreiber (1993) denoising before the test (slow: O(N^2))
K_DOWNSAMPLE = "minmax"       # keep only local minima/maxima (Toker 2022); None = no downsampling
K_IMPL = "fast"               # "fast" = vectorized (FFT) version in chaos_indexes.py,
                              # "edgeofpy" = edgeofpy.chaos.z1_chaos_test (same algorithm, slow)

# ---- covariance width / lambda_max (Dahmen et al. 2019, edgeofpy.lambda_max)
LAMBDA_N_UNITS = None         # N in lambda_max = sqrt(1 - sqrt(1 / (1 + N * width^2))):
                              # None = number of channels

# ---- Lempel-Ziv complexity (neurokit2)
LZC_SYMBOLIZE = "median"      # binarization of the signal before the LZ76 count

# ---- aperiodic exponent (specparam / fooof)
PSD_SEGMENT_S = 2.0           # Welch segment (s)
APERIODIC_FRANGE = (2.0, 40.0)
APERIODIC_MODE = "fixed"      # "fixed" (line in log-log) or "knee"

# ---- Renyi entropy (neurokit2)
RENYI_ALPHA = 2               # order of the entropy (2 = collision entropy H2)
RENYI_BINS = 16               # equal-width amplitude bins used to estimate the distribution

# ---- multiscale entropy (neurokit2)
MSE_SCALES = list(range(1, 21))   # coarse-graining factors
MSE_DIMENSION = 2
MSE_TOLERANCE = 0.15          # r, in SD units of each signal (r = 0.15 * SD)
MSE_METHOD = "MSEn"           # "MSEn" (Costa 2002), "CMSEn", "RCMSEn" (more stable on short data)

# ---- entropy production (Lynn et al. 2021, coarse-grained states)
EP_N_STATES = 8               # k-means states of the multichannel z-scored signal
EP_LAG_S = 0.01               # transition lag (s)
EP_PSEUDOCOUNT = 0.5          # added to every transition count (avoids log(0))
SEED = 0

# ---- DFA on the amplitude envelope (Linkenkaer-Hansen 2001; Hardstone 2012), every SIGNAL_BANDS
# signal (classically alpha / beta; the broadband envelope is less standard)
DFA_FIT_S = (1.0, 10.0)       # window range used for the fit (s); the upper limit is lowered to
                              # 1/10 of the signal length if the signal is short
DFA_N_SCALES = 20             # log-spaced windows in DFA_FIT_S
DFA_OVERLAP = True            # 50 % overlapping windows

# ---- fE/I (Bruining et al. 2020)
FEI_WINDOW_S = 5.0
FEI_OVERLAP = 0.8
FEI_DFA_THRESHOLD = 0.6       # fE/I = NaN where DFA alpha <= threshold (no LRTC, ratio undefined)

# ---- group statistics
ALPHA = 0.05

# ---- per-subject cache: if it exists with the same parameters it is not recomputed
OVERWRITE = False


def params():
    """Parameters that change the results: if different from the cached ones, the subject is redone."""
    names = ["SFREQ", "BROADBAND", "BANDS", "SIGNAL_BANDS", "FILTER_ORDER", "EDGE_TRIM_S", "CONCATENATE_EPOCHS",
             "MAX_EPOCHS", "MEASURES", "K_LOWPASS_HZ", "K_SIGMA", "K_N_C", "K_DENOISE",
             "K_DOWNSAMPLE", "K_IMPL", "LAMBDA_N_UNITS", "LZC_SYMBOLIZE", "PSD_SEGMENT_S",
             "APERIODIC_FRANGE", "APERIODIC_MODE", "RENYI_ALPHA", "RENYI_BINS", "MSE_SCALES",
             "MSE_DIMENSION", "MSE_TOLERANCE", "MSE_METHOD", "EP_N_STATES", "EP_LAG_S",
             "EP_PSEUDOCOUNT", "SEED", "DFA_FIT_S", "DFA_N_SCALES", "DFA_OVERLAP",
             "FEI_WINDOW_S", "FEI_OVERLAP", "FEI_DFA_THRESHOLD"]
    g = globals()
    return {n: g[n] for n in names}
