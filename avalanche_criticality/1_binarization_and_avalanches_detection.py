"""##binarization:
## 1- identify a threshold separating the distribution of neuronal
activity from the approximately Gaussian distribution expected for background
fluctuations. Events exceeding this threshold are subsequently used to
construct neuronal avalanches

## 2- time bin SHOULD BE IEI_average (average time between events)
"""

"""
Binarizzazione EEG e rilevamento di neuronal avalanches
=========================================================

Pensato per confrontare gruppi con distribuzioni di ampiezza di forma
diversa (es. Photosensitive: coda pesante / non-gaussiana;
Non-Photosensitive: quasi gaussiana), come nei dati discussi.

Punto chiave: la standardizzazione usa mediana + MAD (robusta agli
outlier) invece di media + SD, per non far dipendere la soglia dalla
coda pesante di un gruppo. Ogni canale/soggetto viene standardizzato
rispetto al PROPRIO baseline, non rispetto al gruppo, così i due
gruppi restano confrontabili anche se hanno potenza assoluta diversa.

Dipendenze: numpy, scipy (opzionale: powerlaw per il fit finale)
"""

import numpy as np


def robust_zscore(x, axis=-1):
    """Z-score robusto basato su mediana e MAD (Median Absolute Deviation).

    Il fattore 1.4826 rende la MAD comparabile alla SD sotto normalità,
    ma qui il vantaggio è che mediana e MAD non vengono "gonfiate" da
    pochi eventi estremi in coda, a differenza di media/SD classiche.
    """
    median = np.median(x, axis=axis, keepdims=True)
    mad = np.median(np.abs(x - median), axis=axis, keepdims=True)
    mad = np.where(mad == 0, 1e-12, mad)  # evita divisione per zero
    return (x - median) / (1.4826 * mad)


def binarize(data, threshold=2.5, two_sided=True):
    """Binarizza una matrice (canali x tempo) già z-scorata.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_timepoints)
        Segnale già standardizzato (es. output di robust_zscore).
    threshold : float
        Soglia in unità di z-score robusto.
    two_sided : bool
        Se True, considera attivo sia sopra +threshold sia sotto -threshold
        (utile su ampiezza grezza). Se False, solo sopra +threshold
        (tipico se si lavora sull'inviluppo di Hilbert, sempre >= 0).
    """
    if two_sided:
        return (np.abs(data) > threshold).astype(int)
    return (data > threshold).astype(int)


def find_avalanches(binary_matrix):
    """Estrae le valanghe da una matrice binaria (canali x tempo).

    Ritorna una lista di dict con 'size' (somma attività nei bin coinvolti),
    'duration' (numero di bin) e 'start'/'end' (indici temporali).
    """
    network_activity = binary_matrix.sum(axis=0)  # canali attivi per bin
    active = network_activity > 0

    avalanches = []
    in_avalanche = False
    start = None
    size_accum = 0

    for t, is_active in enumerate(active):
        if is_active and not in_avalanche:
            in_avalanche = True
            start = t
            size_accum = network_activity[t]
        elif is_active and in_avalanche:
            size_accum += network_activity[t]
        elif not is_active and in_avalanche:
            avalanches.append({
                "start": start,
                "end": t - 1,
                "duration": t - start,
                "size": int(size_accum),
            })
            in_avalanche = False

    if in_avalanche:  # valanga che arriva fino alla fine della registrazione
        avalanches.append({
            "start": start,
            "end": len(active) - 1,
            "duration": len(active) - start,
            "size": int(size_accum),
        })

    return avalanches


def threshold_sweep(data, thresholds=(1.5, 2.0, 2.5, 3.0, 3.5), two_sided=True):
    """Ripete binarizzazione + estrazione valanghe per più soglie.

    Utile per verificare la stabilità della distribuzione delle
    dimensioni (e del suo eventuale esponente power-law) al variare
    della soglia scelta -- passo raccomandato prima di trarre conclusioni.
    """
    results = {}
    z = robust_zscore(data)
    for thr in thresholds:
        binmat = binarize(z, threshold=thr, two_sided=two_sided)
        avalanches = find_avalanches(binmat)
        sizes = [a["size"] for a in avalanches]
        durations = [a["duration"] for a in avalanches]
        results[thr] = {"avalanches": avalanches, "sizes": sizes, "durations": durations}
    return results


if __name__ == "__main__":
    # Esempio con dati simulati: sostituisci con i tuoi dati reali
    # shape attesa: (n_channels, n_timepoints)
    rng = np.random.default_rng(0)
    n_channels, n_timepoints = 19, 5000
    fake_data = rng.normal(0, 1, size=(n_channels, n_timepoints))

    sweep = threshold_sweep(fake_data)
    for thr, res in sweep.items():
        n_av = len(res["avalanches"])
        mean_size = np.mean(res["sizes"]) if res["sizes"] else 0
        print(f"Soglia {thr}: {n_av} valanghe, size media = {mean_size:.2f}")

    # Per il fit power-law vero e proprio (richiede: pip install powerlaw):
    #
    # import powerlaw
    # sizes = sweep[2.5]["sizes"]
    # fit = powerlaw.Fit(sizes, discrete=True)
    # print("alpha (esponente):", fit.power_law.alpha)
    # R, p = fit.distribution_compare("power_law", "lognormal")
    # print("Confronto power-law vs lognormal: R =", R, "p =", p)
    # -> se R < 0 e p significativo, il lognormale è preferito alla power-law:
    #    utile per verificare se la coda pesante di Photosensitive è vera
    #    criticità o solo un processo moltiplicativo/lognormale.
