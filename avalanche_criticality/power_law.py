# -*- coding: utf-8 -*-
"""
Power law delle valanghe: esponente (slope), cutoff, quanto la curva e' davvero una power law, DCC.

1. ESPONENTE - massima verosimiglianza (MLE), mai minimi quadrati sull'istogramma log-log
   (Clauset, Shalizi & Newman 2009). Dati discreti (S e T sono interi): MLE esatta
       log L(a) = -a * sum(ln x_i) - n * ln( sum_{k=xmin}^{xmax} k^-a )
   (senza xmax la somma e' la zeta di Hurwitz zeta(a, xmin)). Errore standard dalla curvatura di log L.
   Stessi valori del pacchetto powerlaw (Alstott 2014) usato da Maschke 2024 / edgeofpy.
2. FIT PRINCIPALE (alpha per x >= xmin, usato per tau, alpha e DCC): xmin = minimo dei dati come
   in edgeofpy.fit_powerlaw (xmin='min', quindi Maschke 2024), oppure scelto alla Clauset (valore
   che minimizza la distanza di Kolmogorov-Smirnov D, Clauset 2009 sez. 3.3). p_gof = p-value con
   bootstrap semiparametrico (sez. 4.1): p >= 0.1 -> power law plausibile.
   Il fit di Clauset e' sempre calcolato anche come diagnostica (clauset_*): con poche decine di
   canali le distribuzioni hanno un taglio da dimensione finita e il minimo di KS cade spesso nella
   coda, con esponenti molto ripidi (> 4) che non descrivono il resto della distribuzione.
3. RANGE POWER LAW E CUTOFF SUPERIORE ("dove la distribuzione smette di essere power law"):
   tra tante coppie (xmin, xmax) si fitta la power law troncata su [xmin, xmax] e si testa con KS e
   p-value Monte Carlo; si sceglie il range piu' ampio (in decadi, log10(xmax/xmin)) in cui la power
   law NON e' rifiutata (p >= 0.1), come nell'NCC toolbox (Marshall et al. 2016) e nei fit troncati
   delle valanghe (Ma et al. 2019). xmax_cutoff = estremo superiore di questo range.
   Con dati che hanno un taglio esponenziale, il fit di Clauset (2.) tende a spostare xmin nella
   coda esponenziale: il range (3.) e' la stima da guardare per "dove e' power law e con che slope".
   In piu': power law con taglio esponenziale P(x) ~ x^-a e^(-lambda x) (pacchetto powerlaw), scala
   di cutoff 1/lambda (Maschke 2024 trova "truncated power law" nella maggior parte dei soggetti).
4. QUANTO E' POWER LAW:
   - p_gof (Clauset) e p_range (range scelto)
   - test del rapporto di verosimiglianza normalizzato (Vuong; Clauset sez. 5) contro esponenziale,
     lognormale, power law troncata, esponenziale stirata (pacchetto powerlaw, come edgeofpy):
     R > 0 favorisce la power law, p = significativita' del segno di R
   - decadi coperte dal range (meno di ~1 decade = evidenza debole)
   - MSE, chi quadro ridotto e residui in log-log (project plan, eq. 3.6-3.8) sulla PDF a bin
     logaritmici nel range; residual_trend_rho = Spearman tra residui e log x (trend sistematico =
     curvatura, non power law)
5. DCC (Maschke et al. 2024, Methods; Ma et al. 2019):
       P(S) ~ S^-tau,  P(T) ~ T^-alpha,  <S>(T) ~ T^(1/(sigma nu z))
       DCC = | (alpha - 1) / (tau - 1)  -  1/(sigma nu z) |
   1/(sigma nu z) = pendenza OLS di log10<S> contro log10 T (Friedman et al. 2012) sulle durate nel
   range power law. DCC = 0: relazione di scaling della criticita' rispettata.

Riferimenti
  Clauset, Shalizi & Newman (2009) SIAM Rev 51(4):661-703. doi:10.1137/070710111
  Alstott, Bullmore & Plenz (2014) PLoS ONE 9(1):e85777. doi:10.1371/journal.pone.0085777
  Beggs & Plenz (2003) J Neurosci 23(35):11167-77. doi:10.1523/JNEUROSCI.23-35-11167.2003
  Friedman et al. (2012) Phys Rev Lett 108:208102. doi:10.1103/PhysRevLett.108.208102
  Ma et al. (2019) Neuron 104(4):655-664. doi:10.1016/j.neuron.2019.08.031
  Marshall et al. (2016) Front Physiol 7:250. doi:10.3389/fphys.2016.00250
  Maschke et al. (2024) Commun Biol 7:946. doi:10.1038/s42003-024-06613-8
"""
import warnings

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.integrate import trapezoid
from scipy.special import erfc, zeta

try:
    import powerlaw
    HAVE_POWERLAW = True
except ImportError:                                   # senza powerlaw saltano solo i confronti LR
    HAVE_POWERLAW = False

P_PLAUSIBLE = 0.1                                     # Clauset 2009: p < 0.1 -> power law rifiutata
K_EXACT = 100_000                                     # campionamento discreto esatto fino a qui


# ================================================================== MLE e KS (dati discreti o continui)
def _tail(x, xmin, xmax=None):
    x = np.asarray(x, dtype=float)
    m = x >= xmin
    if xmax is not None:
        m &= x <= xmax
    return x[m]


def _norm(a, xmin, xmax, discrete):
    """Costante di normalizzazione su [xmin, xmax] (xmax None = infinito)."""
    if discrete:
        if xmax is None:
            return zeta(a, xmin)
        if a > 1:
            return zeta(a, xmin) - zeta(a, xmax + 1)
        return np.sum(np.arange(xmin, xmax + 1, dtype=float) ** -a)
    hi = 0.0 if xmax is None else xmax ** (1 - a)
    return (xmin ** (1 - a) - hi) / (a - 1) if a != 1 else np.log((xmax or np.inf) / xmin)


def _neg_loglik(a, n, sum_log, xmin, xmax, discrete):
    return a * sum_log + n * np.log(_norm(a, xmin, xmax, discrete))


def mle_alpha(x, xmin, xmax=None, discrete=True):
    """
    Esponente MLE su [xmin, xmax]. Ritorna (alpha, errore standard, n nel range).
    Continua senza xmax: formula chiusa (Clauset eq. 3.1-3.2). Altrimenti massimo numerico.
    Con un xmax finito la power law e' normalizzabile anche con alpha <= 1, quindi il limite
    inferiore scende a 0.01; senza xmax serve alpha > 1.
    """
    t = _tail(x, xmin, xmax)
    n = t.size
    if n < 2 or np.unique(t).size < 2:
        return np.nan, np.nan, n
    sum_log = np.log(t).sum()
    if not discrete and xmax is None:
        a = 1 + n / np.log(t / xmin).sum()
        return float(a), float((a - 1) / np.sqrt(n)), n
    bounds = (1.0001, 8.0) if xmax is None else (0.01, 8.0)
    if not discrete and xmax is not None:
        bounds = (1.0001, 8.0)                        # (formula continua con a != 1)
    res = minimize_scalar(_neg_loglik, bounds=bounds, method="bounded",
                          args=(n, sum_log, xmin, xmax, discrete), options=dict(xatol=1e-5))
    a = float(res.x)
    h = 1e-3                                          # curvatura numerica -> errore standard
    f = lambda v: _neg_loglik(v, n, sum_log, xmin, xmax, discrete)
    d2 = (f(a + h) - 2 * f(a) + f(a - h)) / h ** 2 if a - h > bounds[0] else np.nan
    se = float(1 / np.sqrt(d2)) if np.isfinite(d2) and d2 > 0 else np.nan
    return a, se, n


def model_cdf(v, a, xmin, xmax=None, discrete=True):
    """P(X <= v) della power law su [xmin, xmax]."""
    v = np.asarray(v, dtype=float)
    if discrete:
        if xmax is None:
            return 1 - zeta(a, v + 1) / zeta(a, xmin)
        if a > 1:
            vv = np.clip(v, xmin - 1, xmax)
            return 1 - (zeta(a, vv + 1) - zeta(a, xmax + 1)) / _norm(a, xmin, xmax, True)
        w = np.cumsum(np.arange(xmin, xmax + 1, dtype=float) ** -a)
        i = np.clip(v - xmin, -1, xmax - xmin).astype(int)
        return np.where(i < 0, 0.0, w[np.maximum(i, 0)] / w[-1])
    hi = 0.0 if xmax is None else xmax ** (1 - a)
    return (xmin ** (1 - a) - v ** (1 - a)) / (xmin ** (1 - a) - hi)


def ks_distance(x, a, xmin, xmax=None, discrete=True):
    """D = max |CDF empirica - CDF del modello| sui dati in [xmin, xmax] (Clauset 2009, eq. 3.9)."""
    t = np.sort(_tail(x, xmin, xmax))
    if t.size == 0 or not np.isfinite(a):
        return np.nan
    u = np.unique(t)
    emp = np.searchsorted(t, u, side="right") / t.size
    return float(np.max(np.abs(emp - model_cdf(u, a, xmin, xmax, discrete))))


def _grid_alpha_discrete(sum_log, n, xmins):
    """
    MLE discreta senza xmax per tanti xmin insieme (vettoriale): griglia di alpha a passo 0.02,
    poi rifinitura a passo 0.001 attorno al minimo. Serve a rendere veloce la ricerca di xmin.
    """
    A = np.arange(1.01, 6.0, 0.02)[:, None]
    nll = A * sum_log + n * np.log(zeta(A, xmins))
    a0 = A[nll.argmin(0), 0]
    F = np.maximum(a0[None, :] + np.arange(-0.02, 0.0201, 0.001)[:, None], 1.0005)
    nll = F * sum_log + n * np.log(zeta(F, xmins))
    return F[nll.argmin(0), np.arange(len(xmins))]


def find_xmin(x, discrete=True, min_tail=30, max_candidates=150):
    """
    Fit di Clauset: xmin che minimizza D tra i valori distinti dei dati con almeno min_tail
    osservazioni >= xmin. Ritorna (xmin, alpha, alpha_se, D, n_tail).
    Discreta: alpha di ogni candidato da una griglia fine (passo 0.001) per la scelta di xmin,
    poi MLE esatta sull'xmin scelto.
    Continua: se i valori distinti sono piu' di max_candidates, i candidati sono max_candidates
    quantili dei dati (con dati continui ogni valore e' distinto e provarli tutti e' troppo lento).
    """
    x = np.sort(np.asarray(x, dtype=float))
    x = x[x > 0]
    best = (np.nan, np.nan, np.nan, np.inf, 0)
    if discrete and x.size:
        u = np.unique(x)
        first = np.searchsorted(x, u)                 # indice del primo x >= u
        n_t = x.size - first
        u, first, n_t = u[n_t >= min_tail], first[n_t >= min_tail], n_t[n_t >= min_tail]
        if not u.size:
            return best
        suffix = np.r_[np.cumsum(np.log(x)[::-1])[::-1], 0.0]
        a_c = _grid_alpha_discrete(suffix[first], n_t, u)
        ds = np.full(u.size, np.inf)
        for i, (c, a) in enumerate(zip(u, a_c)):
            t = x[first[i]:]
            uu = np.unique(t)
            emp = np.searchsorted(t, uu, side="right") / t.size
            ds[i] = np.max(np.abs(emp - (1 - zeta(a, uu + 1) / zeta(a, c))))
        c = float(u[np.argmin(ds)])
        a, se, n = mle_alpha(x, c, None, True)
        return c, a, se, ks_distance(x, a, c, None, True), n
    cand = np.unique(x)
    if cand.size > max_candidates:                    # continui: candidati ai quantili (veloce)
        cand = np.unique(np.quantile(x[:max(x.size - min_tail, 1)], np.linspace(0, 1, max_candidates)))
    for c in cand:
        if np.sum(x >= c) < min_tail:
            break
        a, se, n = mle_alpha(x, c, None, discrete)
        d = ks_distance(x, a, c, None, discrete)
        if np.isfinite(d) and d < best[3]:
            best = (float(c), a, se, d, n)
    return best


# ================================================================== SIMULAZIONI E P-VALUE
def sample_power_law(n, a, xmin, xmax=None, discrete=True, rng=None):
    """
    n campioni dalla power law su [xmin, xmax].
    Discreta: esatta (pmf k^-a) fino a min(xmax, K_EXACT); senza xmax la massa oltre K_EXACT
    (trascurabile per a > 1.5) e' campionata con l'approssimazione continua di Clauset (eq. D.6).
    Continua: CDF inversa.
    """
    rng = np.random.default_rng(rng)
    n = int(n)
    if discrete:
        top = int(min(xmax, K_EXACT)) if xmax is not None else K_EXACT
        k = np.arange(int(xmin), top + 1, dtype=float)
        p = k ** -a
        tail_mass = 0.0                               # massa oltre top (solo se a > 1)
        if a > 1 and (xmax is None or xmax > top):
            tail_mass = zeta(a, top + 1) - (zeta(a, xmax + 1) if xmax is not None else 0.0)
        out = rng.choice(k, size=n, p=p / p.sum())
        if tail_mass > 0:
            far = rng.random(n) < tail_mass / (p.sum() + tail_mass)
            lo = (top + 0.5) ** (1 - a)
            hi = 0.0 if xmax is None else (xmax + 0.5) ** (1 - a)
            u = rng.random(far.sum())
            out[far] = np.floor((lo - u * (lo - hi)) ** (1 / (1 - a)) + 0.5)
        return out
    u = rng.random(n)
    if xmax is None:
        return xmin * (1 - u) ** (-1 / (a - 1))
    lo, hi = xmin ** (1 - a), xmax ** (1 - a)
    return (lo - u * (lo - hi)) ** (1 / (1 - a))


def gof_pvalue(x, xmin, a, d_obs, discrete=True, n_boot=200, min_tail=30, rng=None,
               refit_xmin=True):
    """
    p-value di Clauset 2009 (sez. 4.1), bootstrap semiparametrico: ogni dataset sintetico ha la parte
    sotto xmin ricampionata dai dati e la coda dalla power law fittata; si rifittano alpha (e xmin se
    refit_xmin, cioe' se anche sui dati xmin era stato scelto con KS) e si calcola D.
    p = frazione di D sintetici >= D osservato.
    Precisione ~ 1/(2 sqrt(n_boot)): 2500 repliche per +-0.01 (Clauset), 100-200 per esplorare.
    """
    if n_boot <= 0 or not np.isfinite(a):
        return np.nan
    rng = np.random.default_rng(rng)
    x = np.asarray(x, dtype=float)
    body = x[x < xmin]
    n, n_tail = x.size, int(np.sum(x >= xmin))
    hits = 0
    for _ in range(n_boot):
        k = rng.binomial(n, n_tail / n)
        b = rng.choice(body, n - k) if body.size and n - k else np.array([])
        synth = np.r_[b, sample_power_law(k, a, xmin, None, discrete, rng)]
        if refit_xmin:
            d = find_xmin(synth, discrete, min_tail)[3]
        else:
            d = ks_distance(synth, mle_alpha(synth, xmin, None, discrete)[0], xmin, None, discrete)
        hits += bool(np.isfinite(d) and d >= d_obs)
    return hits / n_boot


def range_pvalue(x, xmin, xmax, a, d_obs, discrete=True, n_sim=100, rng=None):
    """
    p-value della power law troncata su [xmin, xmax] fissati: tanti dati quanti nel range simulati
    dal modello fittato, alpha rifittato, D ricalcolato. p = frazione di D simulati >= D osservato.
    """
    if n_sim <= 0 or not np.isfinite(a):
        return np.nan
    rng = np.random.default_rng(rng)
    n = _tail(x, xmin, xmax).size
    hits = 0
    for _ in range(n_sim):
        s = sample_power_law(n, a, xmin, xmax, discrete, rng)
        a_s = mle_alpha(s, xmin, xmax, discrete)[0]
        hits += bool(ks_distance(s, a_s, xmin, xmax, discrete) >= d_obs)
    return hits / n_sim


# ================================================================== RANGE POWER LAW (CUTOFF)
def power_law_range(x, discrete=True, n_xmin=8, n_xmax=15, n_sim=100, min_tail=30,
                    min_decades=0.5, p_thresh=P_PLAUSIBLE, rng=None):
    """
    Range [xmin, xmax] piu' ampio in cui la power law troncata non e' rifiutata (p >= p_thresh).
    Candidati: n_xmin valori bassi e n_xmax valori alti (log-spaziati tra i valori distinti dei dati).
    Le coppie si provano dalla piu' ampia alla piu' stretta: la prima non rifiutata e' il risultato
    (cosi' servono pochi test). Coppie con meno di min_tail dati o meno di min_decades decadi escluse.
    Ritorna dict(xmin, xmax, decades, alpha, alpha_se, n, D, p, n_tested) (NaN se nessun range regge).
    """
    x = np.asarray(x, dtype=float)
    x = x[x > 0]
    u = np.unique(x)
    out = dict(xmin=np.nan, xmax=np.nan, decades=np.nan, alpha=np.nan, alpha_se=np.nan, n=0,
               D=np.nan, p=np.nan, n_tested=0)
    if u.size < 3:
        return out

    def pick(vals, k):
        idx = np.unique(np.round(np.geomspace(1, vals.size, min(k, vals.size)) - 1).astype(int))
        return vals[idx]

    lows = pick(u[:max(1, int(np.searchsorted(u, np.quantile(x, 0.9))))], n_xmin)
    highs = pick(u, n_xmax)
    pairs = [(lo, hi) for lo in lows for hi in highs
             if hi > lo and np.log10(hi / lo) >= min_decades]
    pairs.sort(key=lambda p: (-np.log10(p[1] / p[0]), p[0]))
    rng = np.random.default_rng(rng)
    for lo, hi in pairs:
        n = _tail(x, lo, hi).size
        if n < min_tail:
            continue
        a, se, _ = mle_alpha(x, lo, hi, discrete)
        d = ks_distance(x, a, lo, hi, discrete)
        p = range_pvalue(x, lo, hi, a, d, discrete, n_sim, rng)
        out["n_tested"] += 1
        if p >= p_thresh:
            out.update(xmin=float(lo), xmax=float(hi), decades=float(np.log10(hi / lo)), alpha=a,
                       alpha_se=se, n=int(n), D=d, p=p)
            break
    return out


# ================================================================== BONTA' IN LOG-LOG
def loglog_gof(x, a, xmin, xmax=None, discrete=True, n_bins=20):
    """
    MSE, chi quadro ridotto e residui (project plan eq. 3.6-3.8) tra la probabilita' empirica in bin
    logaritmici e quella del modello, nel range [xmin, xmax]. sigma_i dei punti in log10 dal conteggio
    poissoniano del bin: sigma_i = 1 / (ln10 sqrt(count_i)). Parametri fittati p = 1 (alpha).
    Ritorna dict(mse, chi2_red, residuals, centers, residual_trend_rho, residual_trend_p).
    """
    t = _tail(x, xmin, xmax)
    hi = float(t.max()) if t.size else np.nan
    out = dict(mse=np.nan, chi2_red=np.nan, residuals=np.array([]), centers=np.array([]),
               residual_trend_rho=np.nan, residual_trend_p=np.nan)
    if t.size < 10 or not hi > xmin or not np.isfinite(a):
        return out
    if discrete:                                      # bin [e_i, e_{i+1}) di interi
        edges = np.unique(np.floor(np.geomspace(xmin, hi + 1, n_bins + 1)))
        edges[-1] = hi + 1
        model_p = np.diff(model_cdf(edges - 1, a, xmin, xmax, True))
        centers = np.sqrt(edges[:-1] * (edges[1:] - 1))
    else:
        edges = np.geomspace(xmin, hi, n_bins + 1)
        model_p = np.diff(model_cdf(edges, a, xmin, xmax, False))
        centers = np.sqrt(edges[:-1] * edges[1:])
    counts, _ = np.histogram(t, bins=edges)
    model_p = model_p / model_p.sum()                 # probabilita' condizionata a xmin <= x <= max dei dati
    ok = (counts > 0) & (model_p > 0)
    if ok.sum() < 3:
        return out
    r = np.log10(counts[ok] / t.size) - np.log10(model_p[ok])
    sig = 1 / (np.log(10) * np.sqrt(counts[ok]))
    rho, p = stats.spearmanr(np.log10(centers[ok]), r)
    out.update(mse=float(np.mean(r ** 2)), chi2_red=float(np.sum((r / sig) ** 2) / (ok.sum() - 1)),
               residuals=r, centers=centers[ok], residual_trend_rho=float(rho),
               residual_trend_p=float(p))
    return out


# ================================================================== CONFRONTO CON ALTRE DISTRIBUZIONI
LR_ALTERNATIVES = ("exponential", "lognormal", "stretched_exponential")   # (+ truncated_power_law, sotto)


def _tpl_norm(a, lam, xmin, k_exact=10_000):
    """Z = sum_{k >= xmin} k^-a e^(-lambda k): somma esatta fino a xmin + k_exact, poi integrale in log."""
    k = np.arange(xmin, xmin + k_exact, dtype=float)
    z = np.sum(np.exp(-a * np.log(k) - lam * k))
    k0 = xmin + k_exact - 0.5
    end = k0 + 50 / lam
    if end > k0 + 1:
        kk = np.geomspace(k0, end, 2000)
        z += trapezoid(np.exp(-a * np.log(kk) - lam * kk), kk)
    return z


def fit_truncated_power_law(x, xmin, k_exact=10_000):
    """
    Power law con taglio esponenziale, discreta, per x >= xmin (MLE):
        p(k) = k^-a e^(-lambda k) / Z,   Z = sum_{k >= xmin} k^-a e^(-lambda k)
    Z: somma esatta per k < xmin + k_exact, oltre (funzione liscia) integrale numerico in scala
    logaritmica fino a dove e^(-lambda k) < e^-50. Stessi risultati del pacchetto powerlaw, che
    pero' per i dati discreti (mpmath) richiede ~10 s per fit.
    Ritorna (alpha, lambda, log-verosimiglianza).
    """
    from scipy.optimize import minimize
    t = _tail(x, xmin)
    n, sum_log, s = t.size, np.log(t).sum(), t.sum()

    def nll(p):
        a, lam = p[0], np.exp(p[1])
        if not 0 <= a <= 8:                           # stesso dominio dell'esponente della power law
            return np.inf
        return a * sum_log + lam * s + n * np.log(_tpl_norm(a, lam, xmin, k_exact))

    a0 = mle_alpha(t, xmin, None, True)[0]
    best = None
    for lam0 in (1e-3, 1e-2, 1e-1) if np.isfinite(a0) else ():
        r = minimize(nll, [min(max(a0, 0.5), 5.0), np.log(lam0 / max(t.mean(), 1))],
                     method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-6, maxiter=2000))
        if best is None or r.fun < best.fun:
            best = r
    if best is None:
        return np.nan, np.nan, np.nan
    return float(best.x[0]), float(np.exp(best.x[1])), float(-best.fun)


def _tpl_log_norm_continuous(a, lam, xmin):
    """log Z, Z = integrale da xmin a infinito di x^-a e^(-lambda x) dx (in u = ln x, con quad)."""
    from scipy.integrate import quad
    u0 = np.log(xmin)
    shift = -a * u0 - lam * xmin + u0                 # valore dell'integrando in u0 (log), per stabilita'
    f = lambda u: np.exp(-a * u - lam * np.exp(u) + u - shift)
    hi = np.log(xmin + 60 / lam) if lam > 0 else np.inf
    val, _ = quad(f, u0, hi, limit=200)
    return np.log(val) + shift if val > 0 else np.inf


def fit_truncated_power_law_continuous(x, xmin):
    """
    Power law con taglio esponenziale, dati continui, per x >= xmin (MLE):
        p(x) = x^-a e^(-lambda x) / Z
    Scritta qui perche' il pacchetto powerlaw (versione 2.0.0) con dati continui e xmin fissato lascia
    l'esponente al valore iniziale 3.0 (fitta solo lambda), cosi' come la power law semplice.
    Ritorna (alpha, lambda, log-verosimiglianza, log-verosimiglianze per punto).
    """
    from scipy.optimize import minimize
    t = _tail(x, xmin)
    n, sum_log, s = t.size, np.log(t).sum(), t.sum()
    if n < 10:
        return np.nan, np.nan, np.nan, None

    def nll(p):
        a, lam = p[0], np.exp(p[1])
        if not 0 <= a <= 12:
            return np.inf
        lz = _tpl_log_norm_continuous(a, lam, xmin)
        return a * sum_log + lam * s + n * lz if np.isfinite(lz) else np.inf

    a0 = mle_alpha(t, xmin, None, False)[0]
    best = None
    for lam0 in (1e-3, 1e-1, 1.0, 10.0):
        r = minimize(nll, [min(max(a0, 0.5), 10.0), np.log(lam0 / max(t.mean(), 1e-6))],
                     method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-6, maxiter=3000))
        if best is None or r.fun < best.fun:
            best = r
    a, lam = float(best.x[0]), float(np.exp(best.x[1]))
    ll = -a * np.log(t) - lam * t - _tpl_log_norm_continuous(a, lam, xmin)
    return a, lam, float(-best.fun), ll


def compare_distributions(x, xmin, discrete=True, alternatives=LR_ALTERNATIVES):
    """
    Power law contro distribuzioni alternative, sopra lo stesso xmin (Clauset 2009, sez. 5):
      - exponential, lognormal, stretched_exponential (non annidate): rapporto di verosimiglianza
        normalizzato di Vuong con il pacchetto powerlaw (Alstott 2014, come edgeofpy.fit_powerlaw).
        R_vs_<alt> > 0 = power law preferita; p_vs_<alt> = significativita' del segno di R.
      - truncated_power_law (annidata: lambda = 0 e' la power law): R = logL_pl - logL_tpl (<= 0) e
        p = chi2(1 gdl) di 2|R|: p < 0.05 -> il taglio esponenziale migliora davvero il fit.
        In piu' tpl_alpha, tpl_lambda e la scala del cutoff tpl_cutoff = 1/lambda.
    """
    out = {f"R_vs_{alt}": np.nan for alt in (*alternatives, "truncated_power_law")}
    out.update({f"p_vs_{alt}": np.nan for alt in (*alternatives, "truncated_power_law")})
    out.update(tpl_alpha=np.nan, tpl_lambda=np.nan, tpl_cutoff=np.nan)
    if not np.isfinite(xmin):
        return out
    x = np.asarray(x, dtype=float)
    if discrete:
        t = _tail(x, xmin)
        a, _, n = mle_alpha(t, xmin, None, True)
        ll_pl = -_neg_loglik(a, n, np.log(t).sum(), xmin, None, True)
        ta, tl, ll_tpl = fit_truncated_power_law(t, xmin)
        if np.isfinite(ll_tpl):
            R = min(ll_pl - ll_tpl, 0.0)
            out.update(R_vs_truncated_power_law=R, p_vs_truncated_power_law=float(stats.chi2.sf(-2 * R, 1)),
                       tpl_alpha=ta, tpl_lambda=tl, tpl_cutoff=1 / tl if tl > 0 else np.inf)
    if not HAVE_POWERLAW:
        return out
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        fit = powerlaw.Fit(x, xmin=xmin, discrete=discrete, verbose=False)
        for alt in alternatives:
            try:
                R, p = fit.distribution_compare("power_law", alt, normalized_ratio=True)
                out[f"R_vs_{alt}"], out[f"p_vs_{alt}"] = float(R), float(p)
            except Exception:
                pass
    return out


def best_distribution(lr):
    """
    'power_law' se nessuna alternativa e' significativamente migliore (R < 0 con p < 0.05),
    altrimenti l'alternativa con R piu' negativo tra quelle significative.
    """
    worst, best = 0.0, "power_law"
    for alt in (*LR_ALTERNATIVES, "truncated_power_law"):
        R, p = lr.get(f"R_vs_{alt}", np.nan), lr.get(f"p_vs_{alt}", np.nan)
        if np.isfinite(R) and R < worst and p < 0.05:
            worst, best = R, alt
    return best


MODELS = ("power_law", "truncated_power_law", "lognormal", "exponential", "stretched_exponential")
N_PARAMS = {"power_law": 1, "exponential": 1, "truncated_power_law": 2, "lognormal": 2,
            "stretched_exponential": 2}
# coppie annidate (modello semplice, modello che lo contiene): power law = troncata con lambda = 0;
# esponenziale = troncata con esponente 0 = esponenziale stirata con beta = 1
NESTED = {("power_law", "truncated_power_law"), ("exponential", "truncated_power_law"),
          ("exponential", "stretched_exponential")}


def _vuong(l1, l2):
    """Rapporto di verosimiglianza normalizzato di Vuong (Clauset 2009, sez. 5): (R, p)."""
    d = l1 - l2
    n, sd = d.size, d.std()
    if n < 2 or sd == 0:
        return 0.0, 1.0
    R = d.sum() / (np.sqrt(n) * sd)
    return float(R), float(erfc(abs(R) / np.sqrt(2)))


def compare_all_distributions(x, xmin="min", discrete=True, alpha=0.05):
    """
    Quale distribuzione descrive meglio i dati (x >= xmin): power law, power law troncata
    (taglio esponenziale), lognormale, esponenziale, esponenziale stirata. Stessi modelli di
    edgeofpy.fit_powerlaw / Maschke 2024 e di Clauset 2009 (sez. 5).

    - Ogni modello e' fittato con la massima verosimiglianza sugli stessi dati (power law e troncata
      con le funzioni di questo modulo, le altre con il pacchetto powerlaw).
    - Ogni coppia e' confrontata con il rapporto di verosimiglianza normalizzato di Vuong: R > 0 =
      il primo modello spiega meglio i dati, p = significativita' del segno di R. Unica eccezione:
      power law contro troncata (modelli annidati) usa il test del rapporto di verosimiglianza con
      chi2 a 1 grado di liberta'. Lo stesso per le altre coppie annidate (NESTED): esponenziale contro
      troncata e contro esponenziale stirata. Per modelli annidati il test di Vuong non e' adatto.
    - Un modello "vince" contro un altro se R va nella sua direzione con p < alpha.
    - Miglior modello: tra quelli che non perdono contro nessuno, quello con meno parametri
      (parsimonia: se il modello piu' grande non migliora in modo significativo si tiene il piu'
      semplice), poi quello che ne batte di piu', poi la somma degli R.
    - Confidenza: "strong" = batte in modo significativo tutti gli altri modelli, esclusi quelli che
      lo contengono e non lo migliorano (che non si possono "battere" per costruzione);
      "moderate" = non perde contro nessuno ma almeno un modello alternativo non e' distinguibile;
      "ambiguous" = ogni modello perde contro almeno un altro (vince la somma degli R).
    - equivalent: altri modelli che non perdono contro nessuno (non distinguibili dal migliore).

    Ritorna dict con best, confidence, n_wins, wins (modello -> vittorie), R_<a>_vs_<b>, p_<a>_vs_<b>,
    tpl_cutoff (1/lambda), n.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    xm = float(x.min()) if (isinstance(xmin, str) and xmin == "min") else float(xmin)
    t = _tail(x, xm)
    out = dict(best="n.a.", confidence="n.a.", n=int(t.size), tpl_cutoff=np.nan)
    if t.size < 30 or np.unique(t).size < 3 or not HAVE_POWERLAW:
        return out
    ll = {}
    a, _, _ = mle_alpha(t, xm, None, discrete)
    ll["power_law"] = -a * np.log(t) - np.log(_norm(a, xm, None, discrete))
    if discrete:                                      # troncata: fit di questo modulo (non quella del pacchetto)
        ta, tl, _ = fit_truncated_power_law(t, xm)
        if np.isfinite(tl):
            ll["truncated_power_law"] = -ta * np.log(t) - tl * t - np.log(_tpl_norm(ta, tl, xm))
    else:
        ta, tl, _, v = fit_truncated_power_law_continuous(t, xm)
        if v is not None and np.all(np.isfinite(v)):
            ll["truncated_power_law"] = v
    if np.isfinite(tl):
        out["tpl_cutoff"] = 1 / tl if tl > 0 else np.inf
        out.update(tpl_alpha=ta, tpl_lambda=tl)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        fit = powerlaw.Fit(t, xmin=xm, discrete=discrete, verbose=False)
        for m in ("lognormal", "exponential", "stretched_exponential"):
            try:
                v = np.asarray(getattr(fit, m).loglikelihoods(t), dtype=float)
                if np.all(np.isfinite(v)):
                    ll[m] = v
            except Exception:
                pass
    models = [m for m in MODELS if m in ll]
    wins = {m: set() for m in models}
    score = {m: 0.0 for m in models}
    for i, m1 in enumerate(models):
        for m2 in models[i + 1:]:
            pair = (m1, m2) if (m1, m2) in NESTED else (m2, m1) if (m2, m1) in NESTED else None
            if pair:                                  # annidati: test LR con chi2 a 1 gdl
                simple, big = pair
                Rs = min(ll[simple].sum() - ll[big].sum(), 0.0)
                p = float(stats.chi2.sf(-2 * Rs, 1))
                R = Rs if m1 == simple else -Rs
            else:
                R, p = _vuong(ll[m1], ll[m2])
            out[f"R_{m1}_vs_{m2}"], out[f"p_{m1}_vs_{m2}"] = R, p
            score[m1] += R
            score[m2] -= R
            if p < alpha and R != 0:
                (wins[m1] if R > 0 else wins[m2]).add(m2 if R > 0 else m1)
    lost = {m: any(m in wins[o] for o in models if o != m) for m in models}
    unbeaten = [m for m in models if not lost[m]]
    if unbeaten:
        best = min(unbeaten, key=lambda m: (N_PARAMS[m], -len(wins[m]), -score[m]))
        supersets = {big for simple, big in NESTED if simple == best}
        required = set(models) - {best} - supersets
        conf = "strong" if required <= wins[best] else "moderate"
        equivalent = sorted(m for m in unbeaten if m != best)
    else:
        best, conf, equivalent = max(models, key=lambda m: score[m]), "ambiguous", []
    out.update(best=best, confidence=conf, n_wins=len(wins[best]), equivalent=equivalent,
               wins={m: sorted(w) for m, w in wins.items()})
    return out


# ================================================================== FIT COMPLETO
def fit_power_law(x, discrete=True, xmin="min", n_boot=200, n_sim=100, min_tail=30,
                  find_range=True, lr_tests=True, rng=0):
    """
    Tutto quello che serve per dire se e dove una distribuzione (dimensioni o durate) e' power law.

    xmin : xmin del fit principale (quello usato per tau, alpha e DCC).
           "min" = minimo dei dati (default di edgeofpy.fit_powerlaw, quindi di Maschke 2024);
           None = Clauset (minimo di KS); oppure un numero.
    Ritorna dict con
      fit principale (x >= xmin, senza limite superiore):
        alpha, alpha_se, xmin, n, n_tail, frac_tail, D, p_gof, plausible (p_gof >= 0.1)
      fit di Clauset (xmin scelto con KS), sempre calcolato come diagnostica:
        clauset_xmin, clauset_alpha, clauset_alpha_se, clauset_D, clauset_n_tail
      range power law (power_law_range):
        range_xmin, xmax_cutoff, decades, alpha_range, alpha_range_se, n_range, D_range, p_range
      bontà in log-log del fit principale: mse, chi2_red, residual_trend_rho, residual_trend_p,
        residuals, residual_centers; e nel range: mse_range, chi2_red_range, residual_trend_rho_range
      confronti (sopra l'xmin del fit principale):
        R_vs_*, p_vs_*, tpl_alpha, tpl_lambda, tpl_cutoff, best_fit
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    res = dict(n=int(x.size))
    rng = np.random.default_rng(rng)
    if x.size < min_tail or np.unique(x).size < 3:
        return res | dict(alpha=np.nan, alpha_range=np.nan, plausible=False,
                          note="troppo pochi dati per il fit")
    cx, ca, cse, cd, cn = find_xmin(x, discrete, min_tail)
    res.update(clauset_xmin=cx, clauset_alpha=ca, clauset_alpha_se=cse, clauset_D=cd,
               clauset_n_tail=int(cn))
    if xmin is None:
        xm, a, se, d, n_tail = cx, ca, cse, cd, cn
    else:
        xm = float(x.min()) if xmin == "min" else float(xmin)
        a, se, n_tail = mle_alpha(x, xm, None, discrete)
        d = ks_distance(x, a, xm, None, discrete)
    p = gof_pvalue(x, xm, a, d, discrete, n_boot, min_tail, rng, refit_xmin=xmin is None)
    res.update(alpha=a, alpha_se=se, xmin=xm, n_tail=int(n_tail), frac_tail=n_tail / x.size,
               D=d, p_gof=p, plausible=bool(np.isfinite(p) and p >= P_PLAUSIBLE))
    g = loglog_gof(x, a, xm, None, discrete)
    res.update(mse=g["mse"], chi2_red=g["chi2_red"], residual_trend_rho=g["residual_trend_rho"],
               residual_trend_p=g["residual_trend_p"], residuals=g["residuals"],
               residual_centers=g["centers"])

    r = power_law_range(x, discrete, n_sim=n_sim, min_tail=min_tail, rng=rng) if find_range else {}
    res.update(range_xmin=r.get("xmin", np.nan), xmax_cutoff=r.get("xmax", np.nan),
               decades=r.get("decades", np.nan), alpha_range=r.get("alpha", np.nan),
               alpha_range_se=r.get("alpha_se", np.nan), n_range=r.get("n", 0),
               D_range=r.get("D", np.nan), p_range=r.get("p", np.nan))
    gr = (loglog_gof(x, res["alpha_range"], res["range_xmin"], res["xmax_cutoff"], discrete)
          if np.isfinite(res["xmax_cutoff"]) else {})
    res.update(mse_range=gr.get("mse", np.nan), chi2_red_range=gr.get("chi2_red", np.nan),
               residual_trend_rho_range=gr.get("residual_trend_rho", np.nan))
    if lr_tests:
        lr = compare_distributions(x, xm, discrete)
        res.update(lr, best_fit=best_distribution(lr) if HAVE_POWERLAW else "n.a.")
    return res


# ================================================================== DCC
def third_exponent(sizes, durations, tmin=None, tmax=None, method="ols"):
    """
    Esponente 1/(sigma nu z) di <S>(T) ~ T^(1/(sigma nu z)).
    method "ols": pendenza di log10<S> contro log10 T (Friedman 2012; Ma 2019) sulle durate in
                  [tmin, tmax] (il range in cui le durate seguono la power law).
    method "edgeofpy": come edgeofpy.fit_third_exponent (power law fittata con il pacchetto powerlaw
                  sulla distribuzione delle <S>): solo per confronto con quel codice, non e' una
                  pendenza di <S>(T).
    Ritorna dict(third, third_se, third_r2, n_points, T, mean_S).
    """
    s, T = np.asarray(sizes, dtype=float), np.asarray(durations, dtype=float)
    m = np.ones(T.size, bool)
    if tmin is not None and np.isfinite(tmin):
        m &= T >= tmin
    if tmax is not None and np.isfinite(tmax):
        m &= T <= tmax
    uT = np.unique(T[m])
    mS = np.array([s[m][T[m] == t].mean() for t in uT])
    out = dict(third=np.nan, third_se=np.nan, third_r2=np.nan, n_points=int(uT.size), T=uT, mean_S=mS)
    if uT.size < 3:
        return out
    if method == "edgeofpy":
        if not HAVE_POWERLAW:
            return out
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = powerlaw.Fit(mS, discrete=True, verbose=False, xmin=mS.min())
        return out | dict(third=float(f.power_law.alpha), third_se=float(f.power_law.sigma))
    lr = stats.linregress(np.log10(uT), np.log10(mS))
    return out | dict(third=float(lr.slope), third_se=float(lr.stderr), third_r2=float(lr.rvalue ** 2))


def dcc(tau, alpha, third):
    """DCC = |(alpha - 1)/(tau - 1) - third| (Maschke 2024; Ma 2019; edgeofpy.dcc)."""
    return float(np.abs((alpha - 1) / (tau - 1) - third))


def avalanche_criticality(sizes, durations, xmin="min", n_boot=200, n_sim=100, min_tail=30,
                          third_method="ols", rng=0):
    """
    tau (dimensioni), alpha (durate), 1/(sigma nu z) e DCC per un soggetto, in due versioni:
      dcc       : tau, alpha del fit principale (x >= xmin; xmin='min' come edgeofpy / Maschke);
                  1/(sigma nu z) sulle durate >= xmin delle durate
      dcc_clauset : tau, alpha con xmin di Clauset (minimo di KS) e 1/(sigma nu z) sulle durate >= xmin.
                  Su un processo di branching critico simulato e' meno distorto di xmin='min' (le
                  distribuzioni sono power law solo asintoticamente), ma con pochi canali l'xmin di
                  Clauset puo' cadere nella coda da dimensione finita: guardare entrambi.
      dcc_range : tau, alpha fittati solo nel range in cui la power law regge;
                  1/(sigma nu z) sulle durate nel range power law delle durate
    scaling_diff = 1/(sigma nu z) - (alpha-1)/(tau-1), con segno. Il segno NON fa parte della
    definizione di Maschke / Ma (il DCC e' un valore assoluto): usarlo per parlare di regime
    sub/supercritico e' una scelta propria, da dichiarare.
    """
    fs = fit_power_law(sizes, True, xmin, n_boot, n_sim, min_tail, rng=rng)
    fd = fit_power_law(durations, True, xmin, n_boot, n_sim, min_tail, rng=rng)
    tau, alpha = fs["alpha"], fd["alpha"]
    th = third_exponent(sizes, durations, fd.get("xmin"), None, third_method)
    pred = (alpha - 1) / (tau - 1)
    out = dict(tau=tau, alpha=alpha, third=th["third"], third_se=th["third_se"],
               third_r2=th["third_r2"], predicted_third=pred, dcc=dcc(tau, alpha, th["third"]),
               scaling_diff=th["third"] - pred, size_fit=fs, duration_fit=fd, third_fit=th)
    tc, ac = fs.get("clauset_alpha", np.nan), fd.get("clauset_alpha", np.nan)
    thc = third_exponent(sizes, durations, fd.get("clauset_xmin"), None, third_method)
    out.update(tau_clauset=tc, alpha_clauset=ac, third_clauset=thc["third"],
               dcc_clauset=dcc(tc, ac, thc["third"]))
    tr, ar = fs["alpha_range"], fd["alpha_range"]
    thr = third_exponent(sizes, durations, fd.get("range_xmin"), fd.get("xmax_cutoff"), third_method)
    out.update(tau_range=tr, alpha_range=ar, third_range=thr["third"],
               predicted_third_range=(ar - 1) / (tr - 1), dcc_range=dcc(tr, ar, thr["third"]),
               third_fit_range=thr)
    return out
