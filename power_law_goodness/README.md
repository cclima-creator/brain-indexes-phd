# Power-law goodness (set aside)

The search for power-law avalanche statistics was disappointing in every database: the size and
duration distributions were mostly not power laws (Clauset goodness of fit) and the best model was
usually another distribution. This analysis was set aside on 2026-11-02. Its code is kept here, and
the avalanche pipelines (`photosensitive/MAIN_IPS_AVALAN.py`, `psychedelics/MAIN_AYA_AVALAN.py`) no
longer compute it.

| file | content |
|---|---|
| `power_law.py` | discrete / continuous MLE exponent, Clauset xmin and bootstrap p-value, power-law range and upper cutoff, truncated power law, likelihood-ratio tests between distributions, log-log MSE / reduced chi² / residuals, 1/σνz (gamma) and DCC (was `avalanche_criticality/power_law.py`) |
| `MAIN_IPS_AVALAN_POWER_LAW.py` | frozen copy of the IPS avalanche pipeline with the power-law part: goodness of fit, slopes of size (τ) and duration (α), gamma = 1/(σνz), DCC (main, Clauset xmin, power-law range, "rigorous"), best distribution. Results in `criticality_results/power law goodness/` |
| `plot_ips_avalan_power_law.py` | its figures (figA3 fits, figA4 ⟨S⟩(T) scaling, figA5 exponents and DCC, figA6 fit quality, figA15 distribution table) |
| `MAIN_IPS_AVALAN_BINS_PL.py` | the same distribution fits with fixed time bins (4, 8, 20 ms) |
| `test_power_law.py` | tests on synthetic data: `python power_law_goodness/test_power_law.py` |

The frozen copies use the old processing (each epoch filtered on its own). The amplitude
distribution analysis (`photosensitive/MAIN_IPS_AMPLITUDE_DIST.py`) still imports `power_law.py`
from here for its continuous fits.

References: Clauset, Shalizi & Newman 2009; Alstott et al. 2014; Marshall et al. 2016; Ma et al. 2019;
Maschke et al. 2024.
