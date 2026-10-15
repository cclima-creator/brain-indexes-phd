# -*- coding: utf-8 -*-
"""Test su dati sintetici. Lancio: pytest avalanche_criticality/   (oppure python avalanche_criticality/test_avalanches.py)"""
import sys
from pathlib import Path

import numpy as np

# cartella principale del repository (brain-indexes-phd/) nel path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from avalanche_criticality.binarization import (zscore_channels, binarize, detect_avalanches,
                                                avalanches_from_epochs, divergence_threshold)
from avalanche_criticality.branching_ratio import (branching_ratio_bins, branching_ratio_geometric,
                                                   avalanche_profiles)
from avalanche_criticality import power_law as pl
from avalanche_criticality.repertoire import repertoire, hamming_matrix, shared_repertoire
from avalanche_criticality.fano_factor import fano_factor


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


def test_one_event_per_excursion_at_peak():
    z = np.zeros((1, 20))
    z[0, 3:7] = [2.5, 3.0, 4.0, 2.2]          # escursione positiva, picco al campione 5
    z[0, 12:14] = [-3.0, -5.0]                 # escursione negativa, picco al campione 13
    ev = binarize(z, 2.0, mode="peak")
    assert list(np.flatnonzero(ev[0])) == [5, 13]
    assert binarize(z, 2.0, mode="excursion").sum() == 6


def test_avalanches_split_by_iei_and_edges_dropped():
    ev = np.zeros((2, 100), dtype=np.int8)
    ev[0, [10, 12]] = 1                        # stessa valanga (distanza 2 <= 4)
    ev[1, 13] = 1
    ev[0, 40] = 1                              # nuova valanga (distanza 27 > 4)
    ev[1, 99] = 1                              # tocca la fine: scartata
    av, _, info = detect_avalanches(ev, sfreq=1000, max_iei_s=0.004)
    assert info["bin_len"] == 4
    assert [a["size"] for a in av] == [3, 1]
    assert av[0]["pattern"].tolist() == [True, True]
    assert av[0]["dur_bin"] == 1               # campioni 10-13: un bin da 4


def test_epochs_zscore_uses_all_epochs():
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, (5, 3, 1000))
    z = zscore_channels(data)
    assert np.allclose(z.mean(axis=(0, 2)), 0) and np.allclose(z.std(axis=(0, 2)), 1)
    res = avalanches_from_epochs(data, 1000.0, 2.0, 0.008)
    assert res["patterns"].shape == (len(res["sizes"]), 3)
    assert res["n_samples"] == 5000


def test_divergence_threshold_first_heavier_bin_beyond_min_sd():
    c = np.linspace(-5, 5, 101)
    heavier = (np.abs(c) < 0.3) | (np.abs(c) > 1.75)   # picco centrale e code (da 1.8 SD)
    thr, pos, neg = divergence_threshold(c, heavier, min_sd=1.0)
    assert np.isclose(pos, 1.8) and np.isclose(neg, 1.8) and np.isclose(thr, 1.8)


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


def test_branching_ratio_estimators():
    counts = np.zeros(200, int)
    counts[[10, 11, 12]] = [2, 4, 2]           # n(t+1)/n(t): 2, 0.5
    assert np.isclose(branching_ratio_bins(counts, 1), np.mean([2.0, 0.5, 0.0]))
    profiles = avalanche_profiles(counts, 1)
    assert len(profiles) == 1 and np.isclose(branching_ratio_geometric(profiles), 1.0)


def test_repertoire_and_hamming():
    p = np.array([[1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0], [1, 1, 0, 0]], bool)
    rep, uniq, counts = repertoire(p, duration_s=2.0)
    assert rep["repertoire_size"] == 3 and rep["repertoire_rate"] == 1.5
    assert np.allclose(hamming_matrix(uniq, normalize=False), hamming_matrix(uniq) * 4)
    assert shared_repertoire(uniq[:2], uniq[1:])["n_shared"] == 1


def test_fano_poisson_is_one():
    ev = np.random.default_rng(6).poisson(2.0, 200_000)
    ff, mu, var = fano_factor(ev, 1)
    assert abs(ff - 1) < 0.05


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
