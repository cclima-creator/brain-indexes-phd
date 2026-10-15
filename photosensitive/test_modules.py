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
import pandas as pd

from descriptive_analysis.spectral import (compute_psd, band_power_table, aperiodic_fit,
                                           psd_leave_one_out)
from descriptive_analysis.snr import snr_at_frequency, snr_broadband
from descriptive_analysis.amplitude import amplitude_distribution
from descriptive_analysis.summary_stats import (acf_width, stationarity_test, epoch_counts,
                                                ks_permutation_test)

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


def test_acf_width_of_sinusoid():
    # ACF di una sinusoide a f Hz = cos(2*pi*f*lag): scende a 0.5 a lag = 1/(6f), FWHM = 1/(3f)
    t = np.arange(0, T, 1 / SF)
    x = np.sin(2 * np.pi * 5.0 * t)[None, None, :]
    assert abs(acf_width(x, SF)[0, 0] - 1000 / 15) < 3          # 66.7 ms


def test_acf_width_white_noise_is_short():
    x = np.random.default_rng(4).normal(size=(2, 2, SF * T))
    assert np.nanmax(acf_width(x, SF)) < 2 * 1000 / SF           # meno di 2 campioni


def test_stationarity_white_noise_vs_trend():
    rng = np.random.default_rng(5)
    noise = rng.normal(size=(40, 3, SF * T))
    trend = noise * np.linspace(1, 4, SF * T)                      # ampiezza che cresce
    assert (stationarity_test(noise) < 0.05).mean() < 0.12
    assert (stationarity_test(trend) < 0.05).mean() > 0.9


def test_aperiodic_exponent_recovered_with_alpha_peak():
    # spettro con knee a 6 Hz, esponente 1.8 e picco alfa a 10.5 Hz
    f = np.arange(1, 70, 0.1)
    psd = 10 ** (1.0 - np.log10(6.0 ** 1.8 + f ** 1.8)) * (1 + 3 * np.exp(-(f - 10.5) ** 2 / 2))
    r = aperiodic_fit(psd[None], f, (2, 40), mode="knee")
    if r["method"].startswith("FOOOF"):
        assert abs(r["exponent"][0] - 1.8) < 0.15
        assert abs(r["knee_hz"][0] - 6.0) < 1.0
        assert abs(r["alpha_cf"][0] - 10.5) < 0.3
    # senza knee: retta 1/f^1.5 con picco alfa, anche con il metodo di riserva
    psd = 10 ** (1.0 - 1.5 * np.log10(f)) * (1 + 8 * np.exp(-(f - 10) ** 2 / 2))
    r = aperiodic_fit(psd[None], f, (2, 40), mode="fixed")
    assert abs(r["exponent"][0] - 1.5) < 0.1


def test_epoch_counts():
    ep = pd.DataFrame(dict(group=["a"] * 5 + ["b"] * 3, subject=["s1"] * 3 + ["s2"] * 2 + ["s3"] * 3,
                           epoch=range(8), stim_freq=[10, 10, 20, 10, np.nan, 20, 20, 20]))
    per_subj, per_group = epoch_counts(ep)
    assert per_subj.set_index("subject")["n_epochs"].to_dict() == {"s1": 3, "s2": 2, "s3": 3}
    assert per_subj.set_index("subject").loc["s2", "n_no_stim"] == 1
    assert per_group.set_index("group").loc["a", "total"] == 5


def test_ks_permutation_same_vs_different():
    rng = np.random.default_rng(6)
    rows = []
    for k in range(12):
        g = "a" if k < 6 else "b"
        for shift, name in ((0, "same"), (1.0 if g == "a" else 0, "diff")):
            rows += [dict(subject=f"s{k}", group=g, metric=name, v=v)
                     for v in rng.normal(shift, 1, 200)]
    df = pd.DataFrame(rows)
    same = ks_permutation_test(df[df.metric == "same"], "v", n_perm=999)
    diff = ks_permutation_test(df[df.metric == "diff"], "v", n_perm=999)
    assert same["p_perm"] > 0.05 and diff["p_perm"] < 0.01


def test_old_and_new_channel_names_are_unified():
    from descriptive_analysis.topography import unify_channel_names
    chan = pd.DataFrame(dict(group="a", subject=["s1", "s1", "s2", "s2"],
                             channel=["T3", "T7", "T7", "O1"], v=[1.0, 3.0, 5.0, 7.0]))
    out, renamed = unify_channel_names(chan)
    assert renamed == {"T3": "T7"}
    assert out.set_index(["subject", "channel"])["v"].to_dict() == {
        ("s1", "T7"): 2.0, ("s2", "T7"): 5.0, ("s2", "O1"): 7.0}


def test_leave_one_out_flags_outlier_subject():
    f = np.arange(1, 45.05, 0.1)
    rng = np.random.default_rng(8)
    P = np.array([10 / f * rng.uniform(0.8, 1.2) for _ in range(8)])
    P[3] *= 6                                                   # soggetto 6 volte piu' potente
    delta, t = psd_leave_one_out(P, f, [f"s{i}" for i in range(8)])
    assert t.loc[t.influential, "subject"].tolist() == ["s3"]
    assert (delta[3] > 0).all() and (delta[np.arange(8) != 3] < 0).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)