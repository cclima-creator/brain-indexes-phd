# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 14:07:50 2026

@author: Chiara Clima
"""



"""
result
│
├── stats
│   ├── mean
│   ├── median
│   ├── SD
│   ├── RMS
│   ├── median absolute amplitude
│   ├── skewness
│   ├── excess kurtosis
│   ├── fraction >3 SD
│   └── KS distance
│
├── hist_raw
│   ├── centers
│   ├── density
│   ├── mean
│   └── SD
│
└── hist_z
    ├── centers
    └── density
    
"""




"""Distribuzione delle ampiezze del segnale nel tempo e confronto con la gaussiana."""
import numpy as np
from scipy import stats


def amplitude_distribution(data_uv, n_bins=200, n_sigma=6.0, max_per_channel=2_000_000,
                           max_ks_samples=200_000, seed=0):
    """
    data_uv : (n_epochs, n_channels, n_times) in uV.
    n_bins : in quanti intervalli asse ampiezze per isto
    n_sigma : quanto lontano dalla media si mostra histo. se distribuzione e' molto NON-gaussiana servono tante sigma'
    max_per_channel : massimo 2 milioni di campioni temporali complessivi per ciascun canale che verranno utilizzati nell'analisi.
            es: 100 epoche, 24 canali, 12288 campioni/epoca = 1228800 che e' < 2000000 quindi tutto bene.'
                300 epoche, 24 ch, 12288 campioni > 2000000 quinid scelgo sotto campionamento per evitare troppi dati
    max_ks_samples : specifico per il test Kolmogorov-Smirnov : Il test KS confronta una distribuzione osservata con una distribuzione teorica, qui la normale


    Ritorna dict con:
      stats    : media, mediana, sd, rms, skewness, curtosi in eccesso (medie sui canali),
                 frazione di campioni oltre 3 SD (attesa se gaussiana: 0.0027) e distanza
                 di Kolmogorov-Smirnov D dalla gaussiana (sui dati z-scored per canale).
                 I p-value non sono riportati: con milioni di campioni qualsiasi test
                 rifiuta la gaussianita', D e curtosi misurano QUANTO ci si discosta.
      hist_raw : istogramma (densita') dei campioni in uV, tutti i canali insieme,
                 con media e sigma per il fit gaussiano.
      hist_z   : istogramma dei dati z-scored per canale (da confrontare con N(0,1)).
    Registrazioni molto lunghe: si sottocampiona (passo regolare) a max_per_channel.
    """
    n_ep, n_ch, n_t = data_uv.shape
    x = data_uv.transpose(1, 0, 2).reshape(n_ch, -1)    ##ri organizza struttura dati
    if x.shape[1] > max_per_channel:              ##se ho troppi dati scelgo sotto campionamento 
        x = x[:, ::int(np.ceil(x.shape[1] / max_per_channel))]
    sd_ch = x.std(1, keepdims=True)               ##st dev per ogni canale
    sd_ch[sd_ch == 0] = np.nan
    z = (x - x.mean(1, keepdims=True)) / sd_ch     ##calcola z score pe rogni canale
    z = z[np.isfinite(z).all(1)]            # scarta canali piatti

    rng = np.random.default_rng(seed)       ## crea un generatore di numeri casuali
    zf = z.ravel()
    samp = zf[rng.integers(0, zf.size, size=min(zf.size, max_ks_samples))]

    st = dict(n_samples=int(x.size), mean_uV=float(x.mean()),       ##dizionario di metriche : singola statistica riassuntiva dell intero dataset!!
              median_uV=float(np.median(x)), std_uV=float(x.std()), ##mediana dei campioni x. std dev standard globale, quanto ampiezze sono distribuite intorno a media
              rms_uV=float(np.sqrt((x ** 2).mean())),               ## Root Mean Square = magnitudine complessiva del segnale
              median_abs_uV=float(np.median(np.abs(x))),            ##mediana dei vvalori assoluti (faccio prima il modulo di tutto)
              skew=float(stats.skew(x, axis=1).mean()),             ## ASIMMETRIA della distribuzione
              excess_kurtosis=float(stats.kurtosis(x, axis=1).mean()), ##quanto sono pesanti le code della distribuzione rispetto alla normale
              frac_beyond_3sd=float((np.abs(zf) > 3).mean()),       ##frazione di campioni che sono oltre ±3 deviazioni standard
              ks_D_vs_gauss=float(stats.kstest(samp, "norm").statistic)) ##Kolmogorov-Smirnov test confronta la distribuzione osservata con una distribuzione teorica : D piccolo circa distrib normale; D grande molto diversa 

    mu, sigma = float(x.mean()), float(x.std())         ##gaussiana di riferimento
    edges = np.linspace(mu - n_sigma * sigma, mu + n_sigma * sigma, n_bins + 1)  ##Crea i confini dei bin
    dens, _ = np.histogram(x, bins=edges, density=True)  ##Costruisce l'altezza dell'istogramma delle ampiezze EEG in µV.
    hist_raw = dict(centers=(edges[:-1] + edges[1:]) / 2, density=dens, mu=mu, sigma=sigma)

    edges_z = np.linspace(-n_sigma, n_sigma, n_bins + 1)    ##-n_sigma,+n_sigma sono i limiti del grafico, quanti dati lontani dalla media sto mostrando
    dens_z, _ = np.histogram(zf, bins=edges_z, density=True)
    hist_z = dict(centers=(edges_z[:-1] + edges_z[1:]) / 2, density=dens_z)  ##istogramma delle ampiezze normalizzate in unità di SD.
    return dict(stats=st, hist_raw=hist_raw, hist_z=hist_z)