# -*- coding: utf-8 -*-
"""
Dinamica dentro le valanghe: traiettorie nello spazio degli stati, campo di deriva (phase portrait)
e punti fissi. Funzioni generali: non sanno nulla di gruppi o database.

1. Traiettorie. Ogni valanga e' una sequenza di time bin; in ogni bin lo stato e' il vettore degli
   eventi per canale (n_canali). Si usa la stessa binarizzazione e lo stesso time bin delle valanghe
   (avalanche_criticality.binarization).
2. Spazio degli stati. PCA dei vettori di stato di tutti i bin: le prime 2 componenti formano il
   piano in cui si disegna la dinamica.
3. Campo di deriva (primo coefficiente di Kramers-Moyal): in ogni cella di una griglia sul piano,
   media dello spostamento da un bin al successivo,  D1(x) = < x(t+1) - x(t) | x(t) = x >.
   Si usano solo i passaggi dentro una valanga (bin consecutivi della stessa valanga).
4. Punti fissi. Il campo e' approssimato con un polinomio in (PC1, PC2) fittato ai minimi quadrati
   su tutti i passaggi; i punti fissi sono gli zeri del polinomio nella zona coperta dai dati.
   Stabilita' dalla mappa x -> x + D1(x): autovalori mu della sua Jacobiana I + J,
     |mu| < 1 per entrambi -> stabile (attrattore; "focus" se complessi: la traiettoria ruota)
     |mu| > 1 per entrambi -> instabile (repulsore)
     uno < 1 e uno > 1     -> sella
5. Misura per soggetto: fit lineare D1(x) = A x + b; autovalore di A con parte reale massima.
   < 0: lo stato torna verso un punto fisso (rilassamento, piu' negativo = piu' veloce);
   ~ 0: dinamica marginale (attesa vicino alla criticita'); > 0: lo stato si allontana.

Riferimenti
  Friedrich, Peinke, Sahimi & Tabar (2011) Phys Rep 506:87-162 (coefficienti di Kramers-Moyal da
    dati). doi:10.1016/j.physrep.2011.05.003
  Strogatz (2015) Nonlinear Dynamics and Chaos, 2nd ed., cap. 6 e 10 (punti fissi, stabilita' di mappe)
  Beggs & Plenz (2003) J Neurosci 23:11167 (valanghe)
"""
import numpy as np
from scipy.optimize import root

from avalanche_criticality.binarization import zscore_channels, binarize, detect_avalanches


# ================================================================== TRAIETTORIE
def avalanche_trajectories(data, sfreq, threshold, bin_s, mode="peak", thresh_type="both"):
    """
    Valanghe con la loro evoluzione canale per canale. data: (n_epoche, n_canali, n_tempi).
    Canali piatti: nessun evento (restano nell'ordine dei canali).
    Ritorna dict:
      patterns     (n_valanghe, n_canali) bool : canali attivi almeno una volta
      trajectories lista di (n_bin, n_canali) uint8 : eventi per canale in ogni time bin
      epoch, start : epoca e campione iniziale di ogni valanga (ordine temporale)
      bin_len      : campioni per bin
    """
    data = np.asarray(data, dtype=float)
    if data.ndim == 2:
        data = data[None]
    z = np.nan_to_num(zscore_channels(data))
    pats, trajs, ep_idx, starts = [], [], [], []
    bin_len = max(1, int(round(bin_s * sfreq)))
    for k, zk in enumerate(z):
        ev = binarize(zk, threshold, mode, thresh_type)
        avs, _, _ = detect_avalanches(ev, sfreq, bin_s)
        for a in avs:
            seg = ev[:, a["start"]:a["end"] + 1]
            nb = int(np.ceil(seg.shape[1] / bin_len))
            pad = np.zeros((seg.shape[0], nb * bin_len), dtype=np.int16)
            pad[:, :seg.shape[1]] = seg
            trajs.append(pad.reshape(seg.shape[0], nb, bin_len).sum(-1).T.astype(np.uint8))
            pats.append(a["pattern"])
            ep_idx.append(k)
            starts.append(a["start"])
    n_ch = data.shape[1]
    return dict(patterns=np.array(pats, bool).reshape(-1, n_ch), trajectories=trajs,
                epoch=np.array(ep_idx, int), start=np.array(starts, int), bin_len=bin_len)


def state_vectors(trajectories):
    """Tutti i bin di tutte le valanghe impilati: (n_bin_totali, n_canali)."""
    return np.vstack(trajectories).astype(float) if trajectories else np.zeros((0, 0))


def transitions(trajectories, components, mean):
    """
    Passaggi tra bin consecutivi della stessa valanga, proiettati sulle componenti:
    ritorna (x_t (n, k), dx = x_{t+1} - x_t (n, k)). Valanghe di un solo bin non danno passaggi.
    """
    xs, dxs = [], []
    for t in trajectories:
        if len(t) < 2:
            continue
        p = (t.astype(float) - mean) @ components.T
        xs.append(p[:-1])
        dxs.append(np.diff(p, axis=0))
    if not xs:
        k = len(components)
        return np.zeros((0, k)), np.zeros((0, k))
    return np.vstack(xs), np.vstack(dxs)


# ================================================================== CAMPO DI DERIVA
def drift_field(x, dx, edges_x, edges_y, min_count=20):
    """
    D1 su griglia: media di dx per cella (prime 2 componenti). Celle con meno di min_count passaggi = NaN.
    Ritorna dict(cx, cy (centri), u, v (deriva), count).
    """
    ix = np.digitize(x[:, 0], edges_x) - 1
    iy = np.digitize(x[:, 1], edges_y) - 1
    nx, ny = len(edges_x) - 1, len(edges_y) - 1
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    cnt = np.zeros((ny, nx))
    su, sv = np.zeros((ny, nx)), np.zeros((ny, nx))
    np.add.at(cnt, (iy[ok], ix[ok]), 1)
    np.add.at(su, (iy[ok], ix[ok]), dx[ok, 0])
    np.add.at(sv, (iy[ok], ix[ok]), dx[ok, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        u, v = su / cnt, sv / cnt
    u[cnt < min_count] = np.nan
    v[cnt < min_count] = np.nan
    return dict(cx=(edges_x[:-1] + edges_x[1:]) / 2, cy=(edges_y[:-1] + edges_y[1:]) / 2,
                u=u, v=v, count=cnt)


def _poly_features(x, y, degree):
    return np.column_stack([x ** i * y ** j for i in range(degree + 1)
                            for j in range(degree + 1 - i)])


def fit_drift_polynomial(x, dx, degree=3):
    """Minimi quadrati di dx (2 componenti) su un polinomio di grado `degree` in (PC1, PC2)."""
    F = _poly_features(x[:, 0], x[:, 1], degree)
    coef, *_ = np.linalg.lstsq(F, dx[:, :2], rcond=None)
    pred = F @ coef
    ss = ((dx[:, :2] - dx[:, :2].mean(0)) ** 2).sum(0)
    r2 = 1 - ((dx[:, :2] - pred) ** 2).sum(0) / np.where(ss > 0, ss, np.nan)
    return dict(coef=coef, degree=degree, r2=r2)


def drift_eval(fit, x, y):
    """Valore del polinomio di deriva nei punti (x, y): (u, v)."""
    F = _poly_features(np.atleast_1d(x).astype(float), np.atleast_1d(y).astype(float), fit["degree"])
    out = F @ fit["coef"]
    return out[:, 0], out[:, 1]


def _jacobian(fit, p, h=1e-4):
    J = np.zeros((2, 2))
    for k in range(2):
        e = np.zeros(2)
        e[k] = h
        up = np.array(drift_eval(fit, *(p + e))).ravel()
        dn = np.array(drift_eval(fit, *(p - e))).ravel()
        J[:, k] = (up - dn) / (2 * h)
    return J


def fixed_points(fit, xlim, ylim, n_start=8, tol=None):
    """
    Zeri del campo di deriva fittato dentro [xlim] x [ylim] (zona coperta dai dati), cercati da una
    griglia di n_start x n_start punti iniziali. Ritorna lista di dict(x, y, mu (autovalori della
    mappa I + J), kind = stable / stable focus / unstable / unstable focus / saddle).
    """
    tol = tol or 0.03 * max(xlim[1] - xlim[0], ylim[1] - ylim[0])
    found = []
    for x0 in np.linspace(*xlim, n_start):
        for y0 in np.linspace(*ylim, n_start):
            sol = root(lambda p: np.array(drift_eval(fit, p[0], p[1])).ravel(), [x0, y0])
            if not sol.success:
                continue
            p = sol.x
            if not (xlim[0] <= p[0] <= xlim[1] and ylim[0] <= p[1] <= ylim[1]):
                continue
            if any(np.hypot(p[0] - f["x"], p[1] - f["y"]) < tol for f in found):
                continue
            mu = np.linalg.eigvals(np.eye(2) + _jacobian(fit, p))
            mod = np.abs(mu)
            cplx = np.any(np.abs(mu.imag) > 1e-9)
            if np.all(mod < 1):
                kind = "stable focus" if cplx else "stable"
            elif np.all(mod > 1):
                kind = "unstable focus" if cplx else "unstable"
            else:
                kind = "saddle"
            found.append(dict(x=float(p[0]), y=float(p[1]), mu=mu, kind=kind))
    return found


def linear_drift(x, dx):
    """
    Fit lineare D1(x) = A x + b (prime 2 componenti). Ritorna dict(A, b, lam_max = autovalore di A
    con parte reale massima, fixed_point = -A^-1 b, r2).
    """
    if len(x) < 10:
        return dict(A=np.full((2, 2), np.nan), b=np.full(2, np.nan), lam_max=np.nan,
                    fixed_point=np.full(2, np.nan), r2=np.full(2, np.nan))
    F = np.column_stack([x[:, :2], np.ones(len(x))])
    coef, *_ = np.linalg.lstsq(F, dx[:, :2], rcond=None)
    A, b = coef[:2].T, coef[2]
    lam = np.linalg.eigvals(A)
    pred = F @ coef
    ss = ((dx[:, :2] - dx[:, :2].mean(0)) ** 2).sum(0)
    try:
        fp = -np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        fp = np.full(2, np.nan)
    return dict(A=A, b=b, lam_max=float(lam.real.max()), fixed_point=fp,
                r2=1 - ((dx[:, :2] - pred) ** 2).sum(0) / np.where(ss > 0, ss, np.nan))
