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

## 3. Edge of Chaos
- 0–1 chaos test
- Lyapunov exponent
