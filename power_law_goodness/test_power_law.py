# -*- coding: utf-8 -*-
"""Test of the power-law fits on synthetic data. Run: python power_law_goodness/test_power_law.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from power_law_goodness import power_law as pl


def _branching_process(m=1.0, n_av=4000, max_t=2000, seed=0):
    """Valanghe di un processo di branching con figli Poisson(m): critico se m = 1."""
    rng = np.random.default_rng(seed)
    S, T = [], []
    for _ in range(n_av):
        n, s, d = 1, 0, 0
        while n and d < max_t:
            s += n
            d += 1
            n = rng.poisson(m * n)
        if n == 0:
            S.append(s)
            T.append(d)
    return np.array(S), np.array(T)


def test_mle_recovers_exponent():
    x = pl.sample_power_law(5000, 2.0, 1, None, True, rng=1)
    a, se, n = pl.mle_alpha(x, 1)
    assert abs(a - 2.0) < 3 * se and n == 5000


def test_truncated_range_mle_and_cdf():
    x = pl.sample_power_law(5000, 1.5, 1, 50, True, rng=2)
    assert x.max() <= 50
    a, se, _ = pl.mle_alpha(x, 1, 50)
    assert abs(a - 1.5) < 3 * se
    assert np.isclose(pl.model_cdf(50, a, 1, 50), 1.0) and np.isclose(pl.model_cdf(0, a, 1, 50), 0.0)


def test_power_law_not_rejected_exponential_rejected():
    x = pl.sample_power_law(2000, 1.8, 1, None, True, rng=3)
    f = pl.fit_power_law(x, n_boot=50, n_sim=30, lr_tests=False)
    assert f["p_gof"] >= 0.1
    y = np.ceil(np.random.default_rng(4).exponential(5, 2000))
    g = pl.fit_power_law(y, n_boot=50, n_sim=30, lr_tests=False)
    assert g["p_gof"] < 0.1


def test_truncated_power_law_finds_cutoff():
    rng = np.random.default_rng(5)
    y = np.floor(rng.pareto(0.5, 30000) + 1)
    y = y[rng.random(y.size) < np.exp(-y / 50)][:4000]
    a, lam, _ = pl.fit_truncated_power_law(y, 1.0)
    assert 30 < 1 / lam < 80


def test_dcc_near_zero_for_critical_branching():
    # teoria (campo medio): tau = 3/2, alpha = 2, 1/(sigma nu z) = 2 -> DCC = 0
    S, T = _branching_process(1.0, n_av=8000)
    r = pl.avalanche_criticality(S, T, xmin=None, n_boot=0, n_sim=0)
    assert abs(r["tau"] - 1.5) < 0.1 and abs(r["alpha"] - 2.0) < 0.15
    assert r["dcc"] < 0.15 and np.isclose(r["dcc"], r["dcc_clauset"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
