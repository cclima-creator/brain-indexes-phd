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


def pool_zscores(list_of_subject_data):
    """Concatena i valori z-scorati (per canale, per soggetto) in un unico
    vettore aggregato per il gruppo.

    Parameters
    ----------
    list_of_subject_data : list of ndarray
        Ogni elemento ha forma (n_channels, n_timepoints) e contiene i dati
        GIA' z-scorati (output di robust_zscore) di UN soggetto.

    Returns
    -------
    pooled : ndarray, 1D
        Tutti i valori di tutti i canali di tutti i soggetti, concatenati.
        Attenzione: e' un pooling (append), NON una media -- ogni punto
        temporale di ogni canale resta un'osservazione indipendente.
    """
    return np.concatenate([subj.ravel() for subj in list_of_subject_data])


def fit_core_gaussian(pooled, core_percentile=(20, 80)):
    """Fitta una gaussiana usando SOLO la parte centrale della distribuzione,
    per evitare che la coda (il "segnale" che vuoi rilevare) distorca la
    stima di mu e sigma del "rumore" di fondo.

    Returns
    -------
    mu, sigma : float
        Parametri della gaussiana di riferimento (il modello di rumore).
    """
    lo, hi = np.percentile(pooled, core_percentile)
    core = pooled[(pooled >= lo) & (pooled <= hi)]
    mu = np.mean(core)
    sigma = np.std(core)
    return mu, sigma


def find_deviation_threshold(pooled, mu, sigma, bins=200, tol_ratio=1.5,
                              search_range=(0.2, 5.0), n_search=100):
    """Trova il punto (in unita' di sigma dalla mu del fit centrale) in cui
    la densita' empirica comincia a superare sistematicamente quella
    gaussiana attesa -- cioe' dove i dati smettono di essere spiegabili
    come semplice rumore gaussiano.

    Metodo: calcola l'istogramma empirico (densita'), calcola la pdf
    gaussiana teorica sugli stessi bin, e scansiona da vicino al centro
    verso l'esterno cercando il primo punto in cui
    densita_empirica / densita_gaussiana > tol_ratio.

    Returns
    -------
    threshold_sigma : float
        Soglia in unita' di sigma (dal fit centrale) -- questo e' il numero
        da usare poi in binarize(), ma calcolato su z-score robusti quindi
        va applicato di conseguenza (vedi note in fondo al file).
    diagnostics : dict
        Contiene bin_centers, empirical_density, gaussian_density per
        poterli plottare e verificare visivamente il punto trovato.
    """
    from scipy.stats import norm

    counts, edges = np.histogram(pooled, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    gaussian_density = norm.pdf(centers, loc=mu, scale=sigma)

    # evita divisioni per zero nelle code dove la gaussiana teorica è quasi 0
    gaussian_density_safe = np.where(gaussian_density < 1e-12, 1e-12, gaussian_density)
    ratio = counts / gaussian_density_safe

    candidate_sigmas = np.linspace(search_range[0], search_range[1], n_search)
    threshold_sigma = None
    for s in candidate_sigmas:
        # guarda entrambi i lati (positivo e negativo) allo stesso |sigma|
        mask = (np.abs(centers - mu) >= s * sigma) & (np.abs(centers - mu) < (s + 0.2) * sigma)
        if mask.sum() == 0:
            continue
        local_ratio = ratio[mask].mean()
        if local_ratio > tol_ratio:
            threshold_sigma = s
            break

    diagnostics = {
        "bin_centers": centers,
        "empirical_density": counts,
        "gaussian_density": gaussian_density,
        "ratio": ratio,
    }
    return threshold_sigma, diagnostics


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


def group_threshold_from_data(pooled_zscored, tol_ratio=1.5):
    """Pipeline completa per UN gruppo: fit centrale + ricerca punto di
    divergenza. Da chiamare separatamente per Photosensitive e
    Non-Photosensitive -- i due gruppi possono (e probabilmente devono)
    avere soglie diverse.
    """
    mu, sigma = fit_core_gaussian(pooled_zscored)
    threshold_sigma, diagnostics = find_deviation_threshold(
        pooled_zscored, mu, sigma, tol_ratio=tol_ratio
    )
    return threshold_sigma, mu, sigma, diagnostics


if __name__ == "__main__":
    # Esempio con dati simulati: sostituisci con i tuoi dati reali
    # shape attesa per ogni soggetto: (n_channels, n_timepoints), GIA' z-scorati
    rng = np.random.default_rng(0)

    def simulate_subject(n_channels=19, n_timepoints=5000, heavy_tail=False):
        base = rng.normal(0, 1, size=(n_channels, n_timepoints))
        if heavy_tail:
            # aggiunge eventi rari ed estremi per simulare Photosensitive
            mask = rng.random(size=base.shape) < 0.01
            base[mask] += rng.exponential(4, size=mask.sum())
        return base

    # Simula 10 soggetti per gruppo, GIA' z-scorati per canale
    photosensitive_subjects = [
        robust_zscore(simulate_subject(heavy_tail=True)) for _ in range(10)
    ]
    nonphoto_subjects = [
        robust_zscore(simulate_subject(heavy_tail=False)) for _ in range(10)
    ]

    pooled_photo = pool_zscores(photosensitive_subjects)
    pooled_nonphoto = pool_zscores(nonphoto_subjects)

    thr_photo, mu_p, sigma_p, diag_p = group_threshold_from_data(pooled_photo)
    thr_nonphoto, mu_n, sigma_n, diag_n = group_threshold_from_data(pooled_nonphoto)

    print(f"Soglia data-driven Photosensitive:     {thr_photo} sigma")
    print(f"Soglia data-driven Non-Photosensitive: {thr_nonphoto} sigma")

    # A questo punto usi thr_photo per binarizzare i soggetti Photosensitive
    # e thr_nonphoto per i Non-Photosensitive (soglie diverse per gruppo,
    # ma applicate poi canale per canale, soggetto per soggetto -- vedi
    # binarize() sopra). Per plottare diag_p/diag_n (bin_centers,
    # empirical_density, gaussian_density) e verificare visivamente il
    # punto trovato, usa matplotlib o passami i numeri e te lo genero.

    sweep = threshold_sweep(pooled_photo.reshape(1, -1))
    for thr, res in sweep.items():
        n_av = len(res["avalanches"])
        mean_size = np.mean(res["sizes"]) if res["sizes"] else 0
        print(f"[check sweep] Soglia {thr}: {n_av} valanghe, size media = {mean_size:.2f}")

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
    #    criticità o solo un processo moltiplicativo/lognormale.    # sizes = sweep[2.5]["sizes"]
    # fit = powerlaw.Fit(sizes, discrete=True)
    # print("alpha (esponente):", fit.power_law.alpha)
    # R, p = fit.distribution_compare("power_law", "lognormal")
    # print("Confronto power-law vs lognormal: R =", R, "p =", p)
    # -> se R < 0 e p significativo, il lognormale è preferito alla power-law:
    #    utile per verificare se la coda pesante di Photosensitive è vera
    #    criticità o solo un processo moltiplicativo/lognormale.
