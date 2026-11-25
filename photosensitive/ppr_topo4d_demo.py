# -*- coding: utf-8 -*-
"""
EEG SIMULATO per provare ppr_topo4d (nessun dato reale di pazienti).

Una PPR finta: fondo 1/f + alfa posteriore; flash a 18 Hz da 12 a 26 s (SSVEP occipitale con
armonica); pre 1 (18-20 s) aggancio occipito-parietale crescente e burst; PPR 20-23 s punta-onda
a 3.5 Hz generalizzata, piu' ampia fronto-centrale, che parte dall'occipite con ritardi di
40-100 ms; post 23-25 s attivita' depressa.

  python -m photosensitive.ppr_topo4d_demo  ->  ppr_plan/ppr_topo4d_DEMO_simulato.html
"""
from pathlib import Path

import numpy as np

from photosensitive import ppr_topo4d

CH = ["Fp1", "AF7", "Fp2", "AF8", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4",
      "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
SF, DUR, ONSET, END, STIM = 256.0, 30.0, 20.0, 23.0, (12.0, 26.0)


def simulate(seed=0):
    import mne
    rng = np.random.default_rng(seed)
    info = mne.create_info(CH, SF, "eeg")
    info.set_montage("standard_1020", match_case=False)
    xy = ppr_topo4d.topo_coords(info)
    yfront = xy[:, 1]                                  # >0 anteriore
    post = np.clip(-yfront / 0.47, 0, 1)               # peso posteriore 0..1
    front = np.clip(yfront / 0.47, 0, 1)
    cent = 1 - np.abs(yfront) / 0.47
    n = int(SF * DUR)
    t = np.arange(n) / SF
    f = np.fft.rfftfreq(n, 1 / SF)
    spec = (rng.standard_normal((len(CH), f.size)) + 1j * rng.standard_normal((len(CH), f.size)))
    spec /= np.maximum(f, 1.0)
    x = np.fft.irfft(spec, n)
    x = 8 * x / x.std(1, keepdims=True)
    d = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    x = (np.exp(-(d / 0.12) ** 2) / np.exp(-(d / 0.12) ** 2).sum(1, keepdims=True)) @ x * 2.0

    def ramp(a, b, tau=0.3):
        return 1 / (1 + np.exp(-(t - a) / tau)) * (1 - 1 / (1 + np.exp(-(t - b) / tau)))

    x += 6 * post[:, None] * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 6, (len(CH), 1)))
    st = ramp(*STIM, 0.05)
    build = np.clip((t - 16) / 4, 0, 1) * (t < ONSET)  # aggancio crescente prima della PPR
    reach = post[:, None] + 0.6 * cent[:, None] * build[None]                 # l'SSVEP si estende verso il centro
    lag = rng.normal(0, 0.6, (len(CH), 1)) * (1 - build)   # fasi casuali -> agganciate
    x += st * (5 * reach) * np.sin(2 * np.pi * 18 * t + lag)
    x += st * (2 * reach) * np.sin(2 * np.pi * 36 * t + 2 * lag)
    for c in rng.uniform(17, ONSET, 25):               # burst brevi e diffusi nel pre 1
        g = np.exp(-((t - c) / 0.03) ** 2)
        x += 25 * (post + 0.5 * cent)[:, None] * g * rng.choice([-1, 1], (len(CH), 1))
    delay = 0.04 * (1 - post) + 0.06 * front           # l'esordio parte dall'occipite
    gain = 0.6 * post + 1.0 * cent + 0.9 * front
    for i in range(len(CH)):
        tt = t - delay[i]
        ph = 2 * np.pi * 3.5 * tt
        sw = -40 * np.exp(-((np.mod(ph, 2 * np.pi) - 1.0) / 0.25) ** 2) + 15 * np.sin(ph - 1.0)
        x[i] += gain[i] * sw * ramp(ONSET + delay[i], END + delay[i], 0.05)
    x *= (1 - 0.5 * ramp(END, END + 2, 0.3))[None]    # depressione post
    return mne.io.RawArray(x * 1e-6, info, verbose="ERROR")


def main():
    here = Path(__file__).parent
    out_dir = here / "ppr_plan"
    out_dir.mkdir(exist_ok=True)
    fif = out_dir / "demo_simulato_raw.fif"
    simulate().save(fif, overwrite=True, verbose="ERROR")
    ppr_topo4d.main(["--file", str(fif), "--onset", str(ONSET), "--end", str(END),
                     "--stim-on", str(STIM[0]), "--stim-off", str(STIM[1]), "--stim-freq", "18",
                     "--eyes-closed", "17.0", "--before", "6", "--after", "5", "--patient", "SIMULATO",
                     "--ppr-label", "PPR demo", "--csd", "--simulated",
                     "--out", str(out_dir / "ppr_topo4d_DEMO_simulato.html")])
    fif.unlink()


if __name__ == "__main__":
    main()
