# -*- coding: utf-8 -*-
"""Tests on synthetic data. Run: pytest synchrony_criticality/   (or python synchrony_criticality/test_synchrony.py)"""
import sys
from pathlib import Path

import numpy as np

# repository root (brain-indexes-phd/) on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synchrony_criticality.phase import band_phases, bandpass_filter, analytic_signal
from synchrony_criticality.lag_insensitive import lag_insensitive_matrices, lag_insensitive_epochs
from synchrony_criticality.kuramoto import (kuramoto_order_parameter, global_synchrony, metastability,
                                            kuramoto_epochs, random_phase_R)
from synchrony_criticality.phase_coherence import (phase_coherence_matrix, phase_coherence_loop,
                                                   phase_coherence_epochs, upper, node_strength)
from synchrony_criticality.plv import plv_matrix, plv_loop, plv_summary

SF = 256.0


def test_identical_phases_give_R_one():
    ph = np.tile(np.linspace(0, 20, 500), (8, 1))
    R = kuramoto_order_parameter(ph)
    assert np.allclose(R, 1)
    assert np.isclose(global_synchrony(R), 1) and np.isclose(metastability(R), 0)


def test_uniform_phases_give_rayleigh_R():
    rng = np.random.default_rng(0)
    n_ch = 21
    R = kuramoto_order_parameter(rng.uniform(-np.pi, np.pi, (n_ch, 20000)))
    assert abs(R.mean() - random_phase_R(n_ch)) < 0.01


def test_metastability_detects_switching():
    # half of the time in synchrony, half in disorder: high M; always partially synced: low M
    rng = np.random.default_rng(1)
    n_ch, n_t = 16, 4000
    sync = np.tile(np.linspace(0, 50, n_t), (n_ch, 1))
    switch = sync.copy()
    switch[:, n_t // 2:] = rng.uniform(-np.pi, np.pi, (n_ch, n_t // 2))
    steady = sync + rng.normal(0, 1.2, (n_ch, n_t))
    assert metastability(kuramoto_order_parameter(switch)) > 3 * metastability(kuramoto_order_parameter(steady))


def test_phase_coherence_matches_loop_and_constant_lag():
    rng = np.random.default_rng(2)
    t = np.arange(2000) / SF
    base = 2 * np.pi * 10 * t
    ph = np.vstack([base, base + 0.7, rng.uniform(-np.pi, np.pi, t.size)])
    PC = phase_coherence_matrix(ph)
    assert np.allclose(PC, phase_coherence_loop(ph))
    assert np.isclose(PC[0, 1], 1) and PC[0, 2] < 0.1
    assert np.allclose(phase_coherence_epochs(ph[None])[0], PC)
    assert np.allclose(upper(PC), [PC[0, 1], PC[0, 2], PC[1, 2]])
    assert np.isclose(node_strength(PC)[0], (PC[0, 1] + PC[0, 2]) / 2)


def test_plv_locked_vs_random_and_ppc_unbiased():
    rng = np.random.default_rng(3)
    n_ep, n_t = 40, 300
    t = np.arange(n_t) / SF
    base = 2 * np.pi * 8 * t + rng.uniform(-np.pi, np.pi, (n_ep, 1))   # random phase per epoch
    ph = np.stack([base, base + 1.0 + rng.normal(0, 0.1, (n_ep, n_t)),
                   rng.uniform(-np.pi, np.pi, (n_ep, n_t))], axis=1)
    P = plv_matrix(ph)
    assert np.allclose(P, plv_loop(ph))
    s = plv_summary(ph)
    assert P[0, 1] > 0.95                                  # locked phase difference
    assert abs(s["ppc"][0, 2]) < 0.05                      # PPC ~ 0 without locking
    assert P[0, 2] > s["ppc"][0, 2]                        # PLV biased upward (~1/sqrt(M))


def test_band_phases_recovers_sinusoid_phase():
    t = np.arange(int(6 * SF)) / SF
    x = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 30 * t)
    ph, amp = band_phases(x[None, None], SF, (8, 13), trim_s=1.0)
    true = np.angle(np.exp(1j * (2 * np.pi * 10 * t - np.pi / 2)))[int(SF):-int(SF)]
    err = np.angle(np.exp(1j * (ph[0, 0] - true)))
    assert np.abs(err).max() < 0.05 and np.allclose(amp[0, 0], 1, atol=0.05)


def test_delta_filter_is_stable():
    rng = np.random.default_rng(4)
    y = bandpass_filter(rng.normal(size=(2, 12 * 1024)), 1024.0, (1, 4))
    assert np.isfinite(y).all() and y.std() < 1


def test_kuramoto_epochs_surrogates():
    rng = np.random.default_rng(5)
    n_ep, n_ch, n_t = 4, 10, 2000
    common = 2 * np.pi * 10 * np.arange(n_t) / SF
    ph = common + rng.normal(0, 0.5, (n_ep, n_ch, n_t))
    s, ep, R, hist = kuramoto_epochs(ph, n_surrogates=10)
    assert s["gs"] > 0.8 and s["gs_norm"] > 2 and s["gs_p_surrogate"] < 0.1
    assert ep["gs"].shape == (n_ep,) and R.shape == (n_ep, n_t)
    assert np.isclose((hist["density"] * np.diff(hist["edges"])).sum(), 1)


def _mixed_sources(lag_s, n_t=8000, seed=6):
    """Two channels mixing two independent 10 Hz-band sources; channel 1 also gets source A delayed."""
    rng = np.random.default_rng(seed)
    a, b = (bandpass_filter(rng.normal(size=n_t + 200), SF, (8, 12)) for _ in range(2))
    d = int(round(lag_s * SF))
    ch0 = a[200:] + 0.6 * b[200:]
    ch1 = (np.roll(a, d)[200:] if d else 0.0 * a[200:]) + 0.6 * b[200:] + 0.8 * a[200:] * (d == 0)
    noise = 0.3 * np.vstack([bandpass_filter(rng.normal(size=n_t), SF, (8, 12)) for _ in range(2)])
    return analytic_signal(np.vstack([ch0, ch1]) + noise)        # independent sensor noise


def test_lag_insensitive_ignores_zero_lag_mixing():
    z0 = _mixed_sources(0.0)                       # pure instantaneous mixing
    zl = _mixed_sources(0.025)                     # source A reaches channel 1 after 25 ms (~pi/2 at 10 Hz)
    m0, ml = lag_insensitive_matrices(z0), lag_insensitive_matrices(zl)
    assert phase_coherence_matrix(np.angle(z0))[0, 1] > 0.8          # PC fooled by mixing
    assert m0["wpli"][0, 1] < 0.1 and m0["icoh"][0, 1] < 0.05 and abs(m0["dwpli"][0, 1]) < 0.05
    assert ml["wpli"][0, 1] > 0.7 and ml["dwpli"][0, 1] > 0.5 and ml["icoh"][0, 1] > 0.2
    assert ml["pli"][0, 1] > m0["pli"][0, 1]
    assert np.allclose(m0["wpli"], m0["wpli"].T) and np.all(np.diag(m0["wpli"]) == 0)
    mats, ep = lag_insensitive_epochs(np.stack([zl, zl]))
    assert np.allclose(mats["wpli"], ml["wpli"]) and ep["wpli"].shape == (2,)


def test_regional_within_between():
    from synchrony_criticality.regional import region_indices, within_between, regional_kuramoto
    ch = ["Fp1", "Fp2", "O1", "O2", "T5"]                                # T5 = P7
    idx = region_indices(ch, {"front": ["FP1", "fp2"], "back": ["O1", "O2", "P7"], "none": ["Cz"]})
    assert idx == {"front": [0, 1], "back": [2, 3, 4]}
    m = np.full((5, 5), 0.2)
    m[np.ix_([2, 3, 4], [2, 3, 4])] = 0.9                               # strong coupling at the back
    np.fill_diagonal(m, 1)
    wb = within_between(m, idx)
    assert np.isclose(wb[("back", "back")], 0.9) and np.isclose(wb[("front", "front")], 0.2)
    assert np.isclose(wb[("front", "back")], 0.2)
    # back channels share one phase, front channels are random: within back R = 1, within front low
    rng = np.random.default_rng(8)
    n_t = 4000
    common = np.linspace(0, 200, n_t)
    ph = np.stack([rng.uniform(-np.pi, np.pi, n_t), rng.uniform(-np.pi, np.pi, n_t), common, common, common])
    k = regional_kuramoto(ph[None], idx)
    assert np.isclose(k[("back", "back")][0], 1) and np.isclose(k[("back", "back")][1], 0, atol=1e-9)
    assert k[("front", "front")][0] < 0.8 and k[("front", "back")][0] < 0.8


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
