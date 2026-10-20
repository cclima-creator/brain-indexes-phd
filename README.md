# Brain Criticality Metrics

Code for the computation of brain criticality metrics across three dimensions:

## 1. Avalanche Criticality
- Branching ratio
- Deviation from criticality coefficient
- Power-law proximity

Generic functions in `avalanche_criticality/` (no database-specific code):

| module | content | references |
|---|---|---|
| `binarization.py` | channel z-score, ±k SD events at excursion peaks, divergence point from the Gaussian, avalanches by inter-event interval, spatial patterns | Beggs & Plenz 2003; Shriki et al. 2013; Maschke et al. 2024 |
| `power_law.py` | discrete MLE exponent, Clauset xmin and bootstrap p-value, power-law range and upper cutoff, truncated power law, likelihood-ratio tests, log-log MSE / reduced chi² / residuals, 1/σνz and DCC | Clauset et al. 2009; Alstott et al. 2014; Marshall et al. 2016; Ma et al. 2019; Maschke et al. 2024 |
| `branching_ratio.py` | σ per time bin (Beggs & Plenz / edgeofpy), first-generation σ, geometric σ per avalanche, sweep over time bins | Beggs & Plenz 2003; Sorrentino et al. 2021 |
| `repertoire.py` | unique avalanche patterns, repertoire per second and per avalanche, median normalized Hamming distance, shared patterns between groups | Sorrentino et al. 2021; Maschke et al. 2024 |
| `fano_factor.py` | Fano factor of population activity and FF(W) curve | Fano 1947; Teich et al. 1997; Wilting & Priesemann 2019 |

Photosensitive database: `photosensitive/MAIN_IPS_AVALAN.py` (analysis, independent from `MAIN_IPS.py`,
per-subject cache), `photosensitive/plot_ips_avalan.py` (figures only), settings in
`photosensitive/config_ips_avalan.py`. Requires `numpy scipy pandas matplotlib mne openpyxl powerlaw`
(`powerlaw` only for the likelihood-ratio tests). Tests: `python avalanche_criticality/test_avalanches.py`.

## 2. Synchronization Criticality
- Phase coherence
- Phase Locking Value (PLV)
- Global synchrony (Kuramoto mean)
- Metastability (Kuramoto variance)

Generic functions in `synchrony_criticality/` (no database-specific code):

| module | content | references |
|---|---|---|
| `phase.py` | zero-phase Butterworth band-pass (sos), Hilbert analytic signal, instantaneous phase, edge trimming | Le Van Quyen et al. 2001; Lachaux et al. 1999 |
| `kuramoto.py` | order parameter R(t), global synchrony (mean R), metastability (std R), random-phase level, circular-shift surrogates | Kuramoto 1984; Acebrón et al. 2005; Arenas et al. 2008; Shanahan 2010 |
| `phase_coherence.py` | phase coherence matrix (time average per epoch), upper triangle, node strength | Mormann et al. 2000; Lachaux et al. 1999 |
| `plv.py` | PLV across epochs (time-resolved and averaged), pairwise phase consistency (unbiased) | Lachaux et al. 1999; Vinck et al. 2010 |
| `lag_insensitive.py` | PLI, weighted PLI, debiased wPLI, imaginary coherence (only the lagged part of the coupling) | Nolte et al. 2004; Stam et al. 2007; Vinck et al. 2011 |

Kuramoto GS / M, PC and PLV are inflated (or, for R, cancelled by opposite polarities) by zero-lag
mixing: volume conduction and, in the 5-component dataset, channels rebuilt from 5 ICA components.
At sensor level they are read as a relative comparison between groups; the lag-insensitive indices
measure coupling that the mixing cannot produce.

Photosensitive database: `photosensitive/MAIN_IPS_SYNC.py` (analysis, per-subject cache),
`photosensitive/plot_ips_sync.py` (figures only), settings in `photosensitive/config_ips_sync.py`.
Results in `syncron_results` next to `criticality_results` (`syncron_results/occipital` for the visual ROI).
Tests: `python synchrony_criticality/test_synchrony.py`.

## 3. Edge of Chaos
- 0–1 chaos test
- Lyapunov exponent (λmax of Dahmen et al. 2019) and covariance width
- Complexity / entropy: Lempel-Ziv, Katz FD, Rényi H₂, multiscale entropy, entropy production
- Scale-free dynamics: aperiodic exponent, DFA, fE/I

Generic functions in `edge_of_chaos/chaos_indexes.py` (no database-specific code):

| function | measure | library / references |
|---|---|---|
| `compute_01_chaos_test` | K of the modified 0-1 test (low-pass, min/max downsampling) | edgeofpy algorithm, vectorized; Gottwald & Melbourne 2004, 2009; Toker et al. 2022 |
| `compute_lyapunov_exponent` | λmax of the effective connectivity (`method="rosenstein"`: largest Lyapunov exponent of one channel) | edgeofpy; Dahmen et al. 2019; Rosenstein et al. 1993 |
| `compute_covariance_width` | width of the integrated time-lagged covariance matrix | edgeofpy; Dahmen et al. 2019 |
| `compute_lempel_ziv_complexity` | normalized LZ76 | neurokit2; Lempel & Ziv 1976 |
| `compute_spectral_slope` | aperiodic exponent β | specparam (fallback fooof); Donoghue et al. 2020 |
| `compute_katz_fd` | Katz fractal dimension | neurokit2; Katz 1988 |
| `compute_renyi_entropy` | Rényi H₂ of the amplitude distribution | neurokit2 |
| `compute_multiscale_entropy` | MSE curve and mean | neurokit2; Costa et al. 2002 |
| `compute_entropy_production` | entropy production of k-means coarse-grained states | direct; Lynn et al. 2021 |
| `compute_dfa` | DFA α of the band envelope | neurokit2; Linkenkaer-Hansen et al. 2001; Hardstone et al. 2012 |
| `compute_fei_ratio` | fE/I | port of NBT/crosci; Bruining et al. 2020 |

Pipeline: `edge_of_chaos/MAIN_chaos.py` (`run(groups, out_dir, cfg)`, per-subject cache, also a
command line), defaults in `edge_of_chaos/config_chaos.py` (a database MAIN overrides them),
tables and group tests in `report_chaos.py`, figures in `plot_chaos.py`, I/O in `io_chaos.py`.
Requires `numpy scipy pandas matplotlib mne openpyxl neurokit2 edgeofpy` and `specparam` or `fooof`.
Tests: `python edge_of_chaos/test_chaos.py`.

Photosensitive database: `photosensitive/MAIN_IPS_CHAOS.py` (calls `run` with the IPS groups,
excluded subjects and channel choice, 7 subjects in parallel). Every measure is computed on
broadband 1-45 Hz and on each band (columns `<measure>_<band>`; the aperiodic exponent only
broadband). Results and figures in `edge_of_chaos_results` next to `criticality_results`
(`edge_of_chaos_results/occipital` for the visual ROI).

## One figure per index (all modules)
`descriptive_analysis/index_figures.py` draws one image per index with one panel per band
(broadband first), Mann-Whitney p and FDR across the bands of that index. Written in
`per_index/` by `plot_ips_sync.py`, `edge_of_chaos/plot_chaos.py` (also `topo_<index>.png`) and
`photosensitive/plot_ips_avalan_per_index.py` (from `avalanche_results.pkl`, no re-analysis).
