# -*- coding: utf-8 -*-
"""Tests on synthetic data. Run: pytest edge_of_chaos/   (or python edge_of_chaos/test_chaos.py)"""
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

# repository root (brain-indexes-phd/) on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from edge_of_chaos import chaos_indexes as ci

SF = 256.0


def logistic(r, n, x0=0.4, burn=200):
    x = np.empty(n + burn)
    x[0] = x0
    for i in range(1, x.size):
        x[i] = r * x[i - 1] * (1 - x[i - 1])
    return x[burn:]


def power_law_noise(beta, n, rng):
    """Noise with PSD ~ 1/f^beta (random phases)."""
    f = np.fft.rfftfreq(n)
    amp = np.zeros(f.size)
    amp[1:] = f[1:] ** (-beta / 2)
    return np.fft.irfft(amp * np.exp(2j * np.pi * rng.random(f.size)), n)


def test_01_test_periodic_vs_chaotic():
    assert ci.compute_01_chaos_test(logistic(3.5, 1000), downsample=None) < 0.1
    assert ci.compute_01_chaos_test(logistic(3.97, 1000), downsample=None) > 0.9


def test_01_fast_matches_edgeofpy():
    for r in (3.5, 3.97):
        x = logistic(r, 600)
        fast = ci.compute_01_chaos_test(x, downsample=None)
        ref = ci.compute_01_chaos_test(x, downsample=None, impl="edgeofpy")
        assert abs(fast - ref) < 0.05


def test_dfa_white_and_brownian():
    rng = np.random.default_rng(0)
    w = rng.normal(size=30000)
    assert abs(ci.compute_dfa(w, 100)["alpha"] - 0.5) < 0.1
    assert abs(ci.compute_dfa(np.cumsum(w), 100)["alpha"] - 1.5) < 0.15
    assert np.isnan(ci.compute_dfa(w[:500], 100)["alpha"])           # too short for 1-10 s windows


def test_lzc_katz_renyi_order():
    rng = np.random.default_rng(1)
    noise, sine = rng.normal(size=3000), np.sin(np.arange(3000) / 10)
    assert ci.compute_lempel_ziv_complexity(noise) > 5 * ci.compute_lempel_ziv_complexity(sine)
    assert ci.compute_katz_fd(noise) > ci.compute_katz_fd(sine)
    h_u = ci.compute_renyi_entropy(rng.uniform(size=20000), 2, 16)
    assert abs(h_u - np.log(16)) < 0.01                               # uniform: maximal H2
    assert ci.compute_renyi_entropy(noise, 2, 16) < h_u


def test_mse_noise_types():
    rng = np.random.default_rng(2)
    white, pink = rng.normal(size=3000), power_law_noise(1.0, 3000, rng)
    mw = ci.compute_multiscale_entropy(white, scales=range(1, 11))
    mp = ci.compute_multiscale_entropy(pink, scales=range(1, 11))
    # Costa 2002: white-noise entropy falls with scale, 1/f noise stays nearly constant
    assert mw["curve"][0] > mw["curve"][-1] + 0.5
    assert mp["curve"][-1] > mw["curve"][-1]
    assert mw["curve"].shape == (10,)


def test_entropy_production_reversible_vs_cycle():
    rng = np.random.default_rng(3)
    noise = rng.normal(size=(10, 4, 2000))
    th = np.arange(20000) * 0.3                                       # rotation: broken balance
    cyc = np.stack([np.cos(th), np.sin(th)]) + 0.1 * rng.normal(size=(2, th.size))
    ep_noise = ci.compute_entropy_production(noise, SF)["ep"]
    ep_cyc = ci.compute_entropy_production(cyc, SF, lag_s=1 / SF)["ep"]
    assert ep_noise < 0.02 and ep_cyc > 1
    assert ci.compute_entropy_production(rng.normal(size=5000), SF, n_states=4)["P"].shape == (4, 4)


def test_spectral_slope_recovers_beta():
    rng = np.random.default_rng(4)
    data = np.stack([[power_law_noise(b, int(20 * SF), rng) for b in (1.0, 2.0)] for _ in range(4)])
    sp = ci.compute_spectral_slope(data * 10, SF, (2, 40))
    assert np.allclose(sp["exponent"], [1.0, 2.0], atol=0.15)


def test_fei_and_dfa_threshold():
    rng = np.random.default_rng(5)
    env = np.abs(rng.normal(size=int(120 * SF)))
    r = ci.compute_fei_ratio(env, SF)
    assert np.isfinite(r["fei"]) and 0 < r["fei"] < 2
    assert r["w_amp"].size == r["w_dnf"].size > 10
    assert np.isnan(ci.compute_fei_ratio(env, SF, dfa_alpha=0.5)["fei"])
    # amplitude-modulated noise: windows with larger amplitude have larger normalized fluctuations
    # only if the profile is not normalized -> fE/I differs from the iid value
    t = np.arange(env.size) / SF
    mod = env * (1 + 0.8 * np.sin(2 * np.pi * t / 30))
    assert np.isfinite(ci.compute_fei_ratio(mod, SF)["fei"])


def test_covariance_width_and_lambda():
    rng = np.random.default_rng(6)
    d = rng.normal(size=(15, 6, 400))
    cw = ci.compute_covariance_width(d)
    assert cw["cov"].shape == (6, 6) and np.isfinite(cw["width"])
    lam = ci.compute_lyapunov_exponent(d, width=cw["width"])
    assert 0 <= lam["lambda_max"] < 1
    assert np.isclose(lam["lambda_max"], ci.lambda_from_width(cw["width"], 6))
    assert np.isnan(ci.compute_covariance_width(d[:1])["width"])      # needs >= 2 epochs


def test_rosenstein_lle_sign():
    lle_chaos = ci.compute_lyapunov_exponent(logistic(4.0, 2000), method="rosenstein")["lle"]
    assert lle_chaos > 0


def test_pipeline_end_to_end():
    """Two groups of synthetic EEG files -> run() -> Excel, pickle, figures."""
    import mne
    from edge_of_chaos import config_chaos as cfg
    from edge_of_chaos.MAIN_chaos import run

    saved = cfg.params()
    cfg.MSE_SCALES, cfg.K_N_C, cfg.K_LOWPASS_HZ = list(range(1, 6)), 100, (8.0,)
    rng = np.random.default_rng(7)
    ch = ["O1", "O2", "Pz", "Cz", "Fz", "C3"]
    sf, n_ep, n_t = 128.0, 8, int(12 * 128)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        groups = {}
        for g, beta in (("A", 1.0), ("B", 2.0)):
            (tmp / g).mkdir()
            groups[g] = tmp / g
            for s in range(3):
                x = np.stack([[power_law_noise(beta, n_t, rng) for _ in ch] for _ in range(n_ep)])
                x += 0.5 * np.sin(2 * np.pi * 10 * np.arange(n_t) / sf)          # alpha rhythm
                info = mne.create_info(ch, sf, "eeg")
                mne.EpochsArray(x * 1e-5, info, verbose="ERROR").save(
                    tmp / g / f"{g}{s}-epo.fif", verbose="ERROR")
        try:
            tables = run(groups, tmp / "out", cfg, pattern="*-epo.fif", n_jobs=2)
            subj = tables["subject_metrics"]
            assert len(subj) == 6
            for col in ("k_lp8_broadband", "k_broadband", "k_alpha", "lzc_broadband", "lzc_gamma",
                        "katz_fd_theta", "renyi_h2_beta", "mse_auc_delta", "aperiodic_exponent_broadband",
                        "dfa_alpha", "dfa_broadband", "lambda_max_broadband", "cov_width_alpha",
                        "entropy_production_broadband"):
                assert col in subj and subj[col].notna().all(), col
            # synthetic noise has no LRTC (DFA <= 0.6): fE/I is NaN by construction
            assert "fei_alpha" in subj
            # steeper spectrum in group B
            assert subj.loc[subj.group == "B", "aperiodic_exponent_broadband"].min() > \
                subj.loc[subj.group == "A", "aperiodic_exponent_broadband"].max()
            assert (tmp / "out" / "chaos_results.xlsx").exists()
            assert (tmp / "out" / "figures" / "per_index" / "lzc.png").exists()
            assert (tmp / "out" / "figures" / "per_index" / "topo_lzc.png").exists()
            assert len(list((tmp / "out" / "subjects").glob("*.pkl"))) == 6
            run(groups, tmp / "out", cfg, pattern="*-epo.fif", plots=False)  # second run: cache
        finally:
            for k, v in saved.items():
                setattr(cfg, k, v)


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
