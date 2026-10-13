# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 16:14:53 2026

@author: Chiara Clima
"""

"""Test su segnali sintetici. Lancio: pytest photosensitive/   (oppure python photosensitive/test_modules.py)"""
import sys
from pathlib import Path

import numpy as np

# cartella principale del repository (brain-indexes-phd/) nel path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from descriptive_analysis.spectral import compute_psd, band_power_table
from descriptive_analysis.snr import snr_at_frequency, snr_broadband
from descriptive_analysis.amplitude import amplitude_distribution

SF, T = 500, 11


def _signal(f0=12.0, amp=2.0, noise=5.0, n_ep=4, n_ch=3, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, T, 1 / SF)
    return amp * np.sin(2 * np.pi * f0 * t) + rng.normal(0, noise, (n_ep, n_ch, t.size))


def test_integer_stim_freq_falls_on_bin():
    freqs, _ = compute_psd(_signal(), SF)
    assert np.isclose(freqs[np.argmin(abs(freqs - 12.0))], 12.0)


def test_narrowband_snr_high_at_stim_low_elsewhere():
    freqs, psd = compute_psd(_signal(), SF)
    p = psd.mean(0)
    assert snr_at_frequency(p, freqs, 12.0).mean() > 15
    assert abs(snr_at_frequency(p, freqs, 20.0).mean()) < 3


def test_band_power_peak_in_alpha():
    freqs, psd = compute_psd(_signal(f0=10.0, amp=10.0, noise=1.0), SF)
    df = band_power_table(psd.mean(0), freqs, ["a", "b", "c"],
                          {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)})
    m = df.groupby("band").abs_uV2.mean()
    assert m["alpha"] > 10 * m["theta"] and m["alpha"] > 10 * m["beta"]


def test_amplitude_gaussian_noise_is_gaussian():
    a = amplitude_distribution(np.random.default_rng(1).normal(0, 5, (4, 3, 5000)))
    s = a["stats"]
    assert abs(s["excess_kurtosis"]) < 0.2 and s["ks_D_vs_gauss"] < 0.01
    assert abs(s["frac_beyond_3sd"] - 0.0027) < 0.001


def test_amplitude_detects_heavy_tails():
    x = np.random.default_rng(2).standard_t(3, (4, 3, 5000))
    assert amplitude_distribution(x)["stats"]["excess_kurtosis"] > 1


def test_broadband_snr_nan_if_noise_range_outside_spectrum():
    freqs, psd = compute_psd(_signal(amp=0, noise=5), SF)
    assert np.isnan(snr_broadband(psd.mean(0), freqs, (1, 30), (400, 450))).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)