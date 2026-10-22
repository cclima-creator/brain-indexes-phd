# -*- coding: utf-8 -*-
"""
Detail figures of the edge-of-chaos analysis on the real data of a database (text in Italian):

  dfa_envelope_<band>.png    band signal and amplitude envelope of a representative subject per
                             group, its DFA fluctuation function, DFA alpha of every subject
  fei_<band>.png             fE/I: envelope cut in windows, wAmp vs wnF of a representative subject
                             per group, fE/I distribution over channels and subjects
  aperiodic_spectra.png      PSD of every subject with the aperiodic (1/f^beta) fit, group means,
                             exponent by group
  lambda_max_origin.png      where lambda_max comes from, per band and group: mean(C_ii), std(C_ij),
                             width = std / mean, lambda_max = f(width)
  lambda_max_formula.png     lambda_max(width) curve with every subject and band on it
  lambda_cov_matrices.png    integrated covariance matrix, mean per group, per band

Representative subject = the subject with the median value of the measure in its group; the
single-channel panels use one channel (CHANNEL, occipital by default). Envelopes, windows and
covariances are recomputed from the EEG files with the same settings as the analysis; the PSD and
the aperiodic fits are read from chaos_results.pkl.

    python edge_of_chaos/plot_chaos_detail.py          # IPS database, all channels
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from descriptive_analysis.plots import paper_style, _box_by_group, _label, _p_str, _stars
from descriptive_analysis.summary_stats import compare_groups
from edge_of_chaos import chaos_indexes as ci
from edge_of_chaos.io_chaos import discover_files, load_subject, load_results
from edge_of_chaos.MAIN_chaos import signal_bands

warnings.filterwarnings("ignore")
CHANNEL = "O1"
BAND_IT = {"broadband": "banda larga 1–45 Hz", "delta": "delta 1–4 Hz", "theta": "theta 4–8 Hz",
           "alpha": "alfa 8–13 Hz", "beta": "beta 13–30 Hz", "gamma": "gamma 30–45 Hz"}


def _save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ", path.name)


def _title(fig, title, subtitle):
    h = fig.get_figheight()
    fig.tight_layout()
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1 + 0.55 / h, va="bottom")
    fig.text(0.5, 1 + 0.18 / h, subtitle, ha="center", va="bottom", fontsize=10.5, style="italic", color="0.3")


def _note(ax, text, loc="upper right", size=9):
    x, ha = (0.98, "right") if "right" in loc else (0.02, "left")
    y, va = (0.97, "top") if "upper" in loc else (0.03, "bottom")
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=size,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", alpha=0.92))


def representative(subj, col, group):
    """Subject with the median value of col in group."""
    d = subj[(subj.group == group) & subj[col].notna()]
    if d.empty:
        return None
    return d.iloc[(d[col] - d[col].median()).abs().argsort().iloc[0]]["subject"]


# ================================================================== RECOMPUTATION FROM THE EEG
def collect(groups, cfg, load_kw, file_filter=None, channel=CHANNEL, show_s=30.0):
    """
    One pass over the subjects: for every band, the band signal and envelope of `channel`, its DFA
    and fE/I windows, and the integrated covariance matrix of the whole montage.
    Returns {subject: dict(group, ch_names, sfreq, bands={band: {...}})}.
    """
    out = {}
    for group, folder in groups.items():
        files = discover_files(folder, "*.set")
        if file_filter is not None:
            files = file_filter(group, files)
        for f in files:
            sub = load_subject(f, sfreq=cfg.SFREQ, max_epochs=cfg.MAX_EPOCHS, **load_kw)
            data, sf, chs = sub["data"], sub["sfreq"], sub["ch_names"]
            ic = next((i for i, c in enumerate(chs) if c.upper() == channel.upper()), 0)
            bands = {}
            for bname, band in signal_bands(cfg).items():
                x = ci.broadband(data, sf, band, cfg.FILTER_ORDER, cfg.EDGE_TRIM_S)
                env = ci.band_envelope(data, sf, band, cfg.FILTER_ORDER, cfg.EDGE_TRIM_S)
                e1 = ci.concat_epochs(env)[ic]
                d = ci.compute_dfa(e1, sf, cfg.DFA_FIT_S, cfg.DFA_N_SCALES, cfg.DFA_OVERLAP)
                fe = ci.compute_fei_ratio(e1, sf, cfg.FEI_WINDOW_S, cfg.FEI_OVERLAP)      # without threshold
                cov = ci.compute_covariance_width(x)["cov"]
                n_show = int(show_s * sf)
                bands[bname] = dict(sig=ci.concat_epochs(x)[ic, :n_show].astype(np.float32),
                                    env=e1.astype(np.float32), dfa=d, fei=fe, cov=cov.astype(np.float64))
            out[f.stem] = dict(group=group, ch_names=chs, channel=chs[ic], sfreq=sf, bands=bands)
            print(f"    {f.stem}: ok", flush=True)
    return out


# ================================================================== DFA
def fig_dfa(band, rec, subj, colors, out):
    col = f"dfa_{band}"
    groups = [g for g in colors if g in set(subj.group)]
    fig, ax = plt.subplots(len(groups), 4, figsize=(20, 4.4 * len(groups)),
                           gridspec_kw=dict(width_ratios=[2.2, 1.1, 1.2, 1.0]), squeeze=False)
    for r, g in enumerate(groups):
        sid = representative(subj, col, g)
        R = rec[sid]
        b, sf = R["bands"][band], R["sfreq"]
        t = np.arange(b["sig"].size) / sf
        env = b["env"][:b["sig"].size]
        a = ax[r, 0]
        a.plot(t, b["sig"], color="0.7", lw=0.4, label="segnale filtrato")
        a.plot(t, env, color=colors[g], lw=1.0, label="inviluppo (|Hilbert|)")
        a.set(xlabel="tempo (s, epoche concatenate)", ylabel="µV",
              title=f"{_label(g)} — soggetto {sid}, {R['channel']} (primi {t[-1]:.0f} s)")
        a.legend(loc="upper right", fontsize=8.5, ncol=2)
        a = ax[r, 1]
        m = t < 3
        a.plot(t[m], b["sig"][m], color="0.6", lw=0.7)
        a.plot(t[m], env[m], color=colors[g], lw=1.5)
        a.plot(t[m], -env[m], color=colors[g], lw=0.8, alpha=0.5)
        a.set(xlabel="tempo (s)", ylabel="µV", title="zoom 3 s")
        a = ax[r, 2]
        d = b["dfa"]
        if np.isfinite(d["alpha"]):
            lw, lf = np.log10(d["windows"] / sf), np.log10(d["fluctuations"])
            p = np.polyfit(lw, lf, 1)
            a.plot(lw, lf, "o", color=colors[g], ms=5)
            a.plot(lw, np.polyval(p, lw), "k-", lw=1.5)
            for ref, ls in ((0.5, ":"), (1.0, "--")):
                a.plot(lw, lf[0] + ref * (lw - lw[0]), color="0.5", ls=ls, lw=1, label=f"pendenza {ref}")
            _note(a, f"α = {d['alpha']:.2f}\n(canale {R['channel']})", loc="lower right", size=10)
            a.legend(loc="upper left", fontsize=8.5)
        a.set(xlabel="log10 finestra (s)", ylabel="log10 F(n)", title="DFA dell'inviluppo")
        a = ax[r, 3]
        if r == 0:
            _box_by_group(a, subj, col, colors, "DFA α (media canali)")
            for gg in groups:
                s2 = representative(subj, col, gg)
                v = subj.loc[subj.subject == s2, col].values[0]
                a.scatter(groups.index(gg), v, s=140, facecolor="none", edgecolor="k", lw=2, zorder=4)
            a.axhline(0.5, color="0.5", ls=":", lw=1)
            _note(a, "cerchio = soggetto mostrato\nlinea = 0.5 (nessuna memoria)", loc="lower right", size=8)
        else:
            a.axis("off")
            a.text(0, 0.9, "Come leggere\n\n• l'inviluppo segue i 'burst' del ritmo\n• DFA misura se l'ampiezza ha\n"
                   "  memoria a lungo termine\n• α = 0.5 nessuna memoria,\n  0.5–1 correlazioni a lungo raggio\n\n"
                   f"Fit tra {d['windows'][0] / sf:.0f} e {d['windows'][-1] / sf:.0f} s\n(epoche concatenate)", va="top", fontsize=10)
    _title(fig, f"DFA sull'inviluppo — {BAND_IT[band]}",
           "soggetto rappresentativo = mediana del gruppo per DFA α (media sui canali); colonna 3 = canale mostrato")
    _save(fig, out / f"dfa_envelope_{band}.png")


# ================================================================== fE/I
def fig_fei(band, rec, subj, chan, colors, out, cfg):
    col = f"fei_{band}"
    groups = [g for g in colors if g in set(subj.group)]
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    reps = {g: representative(subj, col, g) or representative(subj, f"dfa_{band}", g) for g in groups}
    R0 = rec[reps[groups[0]]]
    b, sf = R0["bands"][band], R0["sfreq"]
    a = ax[0, 0]
    env = b["env"][:int(60 * sf)]
    t = np.arange(env.size) / sf
    a.plot(t, env, color=colors[groups[0]], lw=0.4)
    top = env.max()
    step = cfg.FEI_WINDOW_S * (1 - cfg.FEI_OVERLAP)
    for k, s0 in enumerate(np.arange(0, 60 - cfg.FEI_WINDOW_S + 1e-9, step)[::5]):
        a.add_patch(plt.Rectangle((s0, top * (1.03 + 0.05 * (k % 2))), cfg.FEI_WINDOW_S, top * 0.04,
                                  fc="0.5", alpha=0.6))
    a.set(xlabel="tempo (s)", ylabel="inviluppo (µV)",
          title=f"Inviluppo in finestre di {cfg.FEI_WINDOW_S:g} s ({cfg.FEI_OVERLAP:.0%} overlap)\n"
                f"{_label(groups[0])}, soggetto {reps[groups[0]]}, {R0['channel']} — una finestra ogni 5 mostrata")
    for k, g in enumerate(groups[:2]):
        a = ax[0, k + 1]
        R = rec[reps[g]]
        fe, d = R["bands"][band]["fei"], R["bands"][band]["dfa"]
        if fe["w_amp"].size:
            a.scatter(fe["w_amp"], fe["w_dnf"], s=10, color=colors[g], alpha=0.55)
            c = np.polyfit(fe["w_amp"], fe["w_dnf"], 1)
            xx = np.linspace(fe["w_amp"].min(), fe["w_amp"].max(), 10)
            a.plot(xx, np.polyval(c, xx), "k-", lw=2)
            rr = np.corrcoef(fe["w_amp"], fe["w_dnf"])[0, 1]
            valid = d["alpha"] > cfg.FEI_DFA_THRESHOLD
            txt = (f"r = {rr:+.2f}\nfE/I = 1 − r = {fe['fei']:.2f}\nDFA α = {d['alpha']:.2f}"
                   + ("" if valid else f"\n(α ≤ {cfg.FEI_DFA_THRESHOLD}: NaN nelle tabelle)"))
            _note(a, txt, size=10)
        a.set(xlabel="ampiezza media della finestra (wAmp, µV)", ylabel="fluttuazione normalizzata (wnF)",
              title=f"{_label(g)} — soggetto {reps[g]}, {R['channel']}")
    a = ax[1, 0]
    allv = chan[col].dropna()
    edges = np.linspace(min(0.6, allv.min()), max(1.4, allv.max()), 41)
    for g in groups:
        v = chan.loc[chan.group == g, col].dropna()
        a.hist(v, edges, density=True, histtype="stepfilled", alpha=0.4,
               color=colors[g], label=f"{_label(g)} (n = {len(v)} canali)")
        a.axvline(v.median(), color=colors[g], lw=2)
    a.axvline(1, color="k", ls="--", lw=1)
    a.set(xlabel="fE/I (canale × soggetto)", ylabel="densità", title="Distribuzione su tutti i canali")
    a.legend(loc="upper left", fontsize=9)
    _note(a, "< 1 inibizione · 1 bilanciato · > 1 eccitazione", loc="lower right", size=8.5)
    _box_by_group(ax[1, 1], subj, col, colors, "fE/I (media canali per soggetto)")
    ax[1, 1].axhline(1, color="k", ls="--", lw=1)
    a = ax[1, 2]
    frac = [chan.loc[chan.group == g, col].isna().mean() for g in groups]
    a.bar(range(len(groups)), frac, color=[colors[g] for g in groups], alpha=0.7)
    a.set_xticks(range(len(groups)))
    a.set_xticklabels([_label(g) for g in groups])
    for i, v in enumerate(frac):
        a.text(i, v, f"{v:.0%}", ha="center", va="bottom")
    a.set(ylabel="frazione di canali", ylim=(0, max(0.1, max(frac) * 1.3)),
          title=f"Canali senza fE/I (DFA α ≤ {cfg.FEI_DFA_THRESHOLD})")
    _title(fig, f"fE/I (Bruining et al. 2020) — {BAND_IT[band]}",
           "fE/I = 1 − corr(wnF, wAmp); soggetto rappresentativo = mediana del gruppo per fE/I (media sui canali)")
    _save(fig, out / f"fei_{band}.png")


# ================================================================== APERIODIC EXPONENT
def fig_aperiodic(res, subj, chan, colors, out, frange=(2.0, 40.0)):
    groups = [g for g in colors if g in set(subj.group)]
    det = res["details"]
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    means = {}
    for k, g in enumerate(groups[:2]):
        a = ax[0, k]
        P, fits = [], []
        for sid in subj.loc[subj.group == g, "subject"]:
            d = det.get(sid, {})
            if "psd" not in d:
                continue
            f, p = d["psd_freqs"], d["psd"].mean(0)
            c = chan[chan.subject == sid]
            beta, off = c.aperiodic_exponent_broadband.mean(), c.aperiodic_offset.mean()
            a.loglog(f, p, color=colors[g], lw=0.7, alpha=0.45)
            P.append(p)
            fits.append((beta, off))
        P = np.array(P)
        means[g] = (f, np.exp(np.log(P).mean(0)), np.median([b for b, _ in fits]), np.median([o for _, o in fits]))
        a.loglog(f, means[g][1], color="k", lw=2, label="media geometrica del gruppo")
        m = (f >= frange[0]) & (f <= frange[1])
        for beta, off in fits:
            a.loglog(f[m], 10 ** off * f[m] ** -beta, color=colors[g], lw=0.6, ls="--", alpha=0.6)
        a.axvspan(*frange, color="0.93", zorder=0)
        a.set(xlabel="frequenza (Hz)", ylabel="PSD (µV²/Hz)", title=f"{_label(g)} (n = {len(P)})")
        a.legend(loc="lower left", fontsize=9)
        _note(a, "linee sottili = PSD di ogni soggetto\n(media sui canali)\ntratteggio = fit aperiodico\n"
                 f"grigio = range del fit {frange[0]:g}–{frange[1]:g} Hz", size=8.5)
    a = ax[0, 2]
    for g in groups:
        f, p, beta, off = means[g]
        m = (f >= frange[0]) & (f <= frange[1])
        a.loglog(f, p, color=colors[g], lw=2, label=f"{_label(g)}: β mediano = {beta:.2f}")
        a.loglog(f[m], 10 ** off * f[m] ** -beta, color=colors[g], lw=1.5, ls="--")
    a.set(xlabel="frequenza (Hz)", ylabel="PSD (µV²/Hz)", title="Confronto tra gruppi")
    a.legend(loc="lower left", fontsize=9)
    _note(a, "pendenza in log-log = −β\nβ più alto = spettro più ripido", size=8.5)
    _box_by_group(ax[1, 0], subj, "aperiodic_exponent_broadband", colors, "esponente aperiodico β")
    a = ax[1, 1]
    for g in groups:
        d = subj[subj.group == g]
        a.scatter(d.aperiodic_offset, d.aperiodic_exponent_broadband, s=50, color=colors[g],
                  edgecolor="k", lw=0.5, label=_label(g))
    a.set(xlabel="offset (log10 µV²/Hz)", ylabel="β", title="Offset e pendenza (per soggetto)")
    a.legend(loc="upper left", fontsize=9)
    _note(a, "offset e β sono correlati: uno spettro più\nripido parte più in alto", loc="lower right", size=8.5)
    a = ax[1, 2]
    edges = np.linspace(chan.aperiodic_r2.min(), 1, 30)
    for g in groups:
        a.hist(chan.loc[chan.group == g, "aperiodic_r2"], edges, alpha=0.45, color=colors[g], label=_label(g))
    a.set(xlabel="R² del fit (per canale)", ylabel="numero di canali", title="Qualità del fit")
    a.legend(loc="upper left", fontsize=9)
    method = next(iter(det.values())).get("aperiodic_method", "")
    _title(fig, "Esponente aperiodico β (pendenza spettrale)",
           f"{method}, range {frange[0]:g}–{frange[1]:g} Hz; PSD di Welch (segmenti 2 s) mediata sulle epoche")
    _save(fig, out / "aperiodic_spectra.png")


# ================================================================== LAMBDA_MAX
def lambda_table(rec, n_units=None):
    rows = []
    for sid, R in rec.items():
        for bname, b in R["bands"].items():
            C = b["cov"]
            n = len(C) if n_units is None else n_units
            md, sd = np.diag(C).mean(), C.std()
            w = sd / md
            rows.append(dict(group=R["group"], subject=sid, band=bname, mean_diag=md, std_all=sd,
                             mean_offdiag=C[~np.eye(len(C), dtype=bool)].mean(), width=w,
                             lambda_max=ci.lambda_from_width(w, n), n_units=n))
    return pd.DataFrame(rows)


def _strip(ax, lt, band, col, colors, ylabel, seed=0):
    rng = np.random.default_rng(seed)
    groups = [g for g in colors if g in set(lt.group)]
    d = lt[lt.band == band]
    data = [d.loc[d.group == g, col].values for g in groups]
    bp = ax.boxplot(data, positions=range(len(groups)), widths=0.5, patch_artist=True, showfliers=False,
                    medianprops=dict(color="k", lw=2))
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor(colors[g])
        patch.set_alpha(0.4)
    for i, (g, v) in enumerate(zip(groups, data)):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=18, color=colors[g], edgecolor="k", lw=0.4, zorder=3)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([_label(g).replace(" ", "\n", 1) for g in groups], fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=9.5)
    t = compare_groups(d, [col])
    if len(t):
        ax.set_title(f"{_p_str(t['p'].iloc[0])} {_stars(t['p'].iloc[0])}", fontsize=9)


def fig_lambda(lt, colors, out):
    bands = list(dict.fromkeys(lt.band))
    rows = [("mean_diag", "1. media(C_ii)\n(varianza integrata)"), ("std_all", "2. SD(C_ij)\n(tutte le coppie)"),
            ("width", "3. w = SD / media"), ("lambda_max", "4. λmax = f(w)")]
    fig, ax = plt.subplots(len(rows), len(bands), figsize=(3.2 * len(bands), 3.1 * len(rows)), squeeze=False)
    for j, b in enumerate(bands):
        for i, (col, lab) in enumerate(rows):
            _strip(ax[i, j], lt, b, col, colors, lab if j == 0 else "")
            if i == 0:
                ax[i, j].text(0.5, 1.22, BAND_IT[b], transform=ax[i, j].transAxes, ha="center",
                              fontsize=12, fontweight="bold")
                ax[i, j].axhline(0, color="0.5", lw=0.8, ls=":")
    _title(fig, "Da dove viene λmax (Dahmen et al. 2019), per banda e gruppo",
           "C = covarianza integrata nel tempo tra canali (edgeofpy); w = SD(C) / media(diag C); λmax = √(1 − √(1/(1 + N·w²))), N = numero di canali")
    _save(fig, out / "lambda_max_origin.png")


def fig_lambda_formula(lt, colors, out):
    n = int(lt.n_units.iloc[0])
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.8))
    a = ax[0]
    w = np.logspace(-2, np.log10(max(50, np.nanmax(np.abs(lt.width)) * 1.2)), 400)
    a.semilogx(w, [ci.lambda_from_width(x, n) for x in w], "k-", lw=2, label=f"λmax(w), N = {n}")
    markers = dict(zip(dict.fromkeys(lt.band), "osD^vP"))
    for (g, b), d in lt.groupby(["group", "band"]):
        a.scatter(np.abs(d.width), d.lambda_max, s=30, color=colors.get(g, "k"), marker=markers[b],
                  edgecolor="k", lw=0.3, alpha=0.8)
    for b, mk in markers.items():
        a.scatter([], [], marker=mk, color="0.6", edgecolor="k", label=BAND_IT[b])
    for g, c in colors.items():
        a.scatter([], [], marker="s", color=c, label=_label(g))
    a.axhline(1, color="0.5", ls=":")
    a.set(xlabel="|w| (scala log)", ylabel="λmax", ylim=(0, 1.05), title="Ogni soggetto e banda sulla curva")
    a.legend(loc="lower right", fontsize=8.5, ncol=2)
    _note(a, "λmax si avvicina a 1 appena w > ~1:\noltre, la curva è piatta e λmax non\ndistingue più i soggetti",
          loc="upper left", size=9)
    a = ax[1]
    for g, c in colors.items():
        d = lt[lt.group == g]
        a.scatter(d.mean_diag, d.std_all, s=26, color=c, edgecolor="k", lw=0.3, label=_label(g))
    a.axvline(0, color="0.5", ls=":")
    a.set_xscale("symlog", linthresh=1.0)
    lim = 10 ** np.ceil(np.log10(np.nanmax(np.abs(lt.mean_diag))))
    a.set_xlim(-lim, lim)
    ticks = [t for k in range(int(np.log10(lim)), 0, -1) for t in (-10 ** k,)] + [0] + \
            [10 ** k for k in range(1, int(np.log10(lim)) + 1)]
    a.set_xticks(ticks)
    a.set_xticklabels([f"{t:g}" for t in ticks], fontsize=9)
    a.set_yscale("log")
    a.set(xlabel="media(C_ii) (scala symlog, lineare tra −1 e 1)", ylabel="SD(C_ij)",
          title="Numeratore e denominatore di w")
    a.legend(loc="lower left", fontsize=9)
    neg = (lt.mean_diag <= 0).mean()
    a.text(0.5, 0.97, f"media(C_ii) ≤ 0 in {neg:.0%} dei casi soggetto×banda.\nIl segnale è filtrato passa-alto: "
           "la varianza integrata\n(potenza a 0 Hz) è vicina a zero, quindi w e λmax sono instabili",
           transform=a.transAxes, ha="center", va="top", fontsize=9,
           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", alpha=0.92))
    _title(fig, "λmax in funzione della larghezza della covarianza", "tutti i soggetti, tutte le bande")
    _save(fig, out / "lambda_max_formula.png")


def fig_cov_matrices(rec, colors, out):
    groups = [g for g in colors if g in {R["group"] for R in rec.values()}]
    bands = list(next(iter(rec.values()))["bands"])
    ch = next(iter(rec.values()))["ch_names"]
    fig, ax = plt.subplots(len(groups), len(bands), figsize=(3.4 * len(bands), 3.5 * len(groups)), squeeze=False)
    for j, b in enumerate(bands):
        Ms = {g: np.mean([R["bands"][b]["cov"] for R in rec.values()
                          if R["group"] == g and R["bands"][b]["cov"].shape == (len(ch), len(ch))], axis=0)
              for g in groups}
        v = max(np.nanpercentile(np.abs(M), 98) for M in Ms.values())
        for i, g in enumerate(groups):
            a = ax[i, j]
            im = a.imshow(Ms[g], cmap="RdBu_r", vmin=-v, vmax=v)
            a.set(xticks=[], yticks=[])
            if j == 0:
                a.set_ylabel(_label(g), fontsize=11)
            if i == 0:
                a.set_title(BAND_IT[b], fontsize=11)
            fig.colorbar(im, ax=a, fraction=0.046)
    _title(fig, "Matrice di covarianza integrata C, media per gruppo",
           f"ordine dei canali: {', '.join(ch)}; la stessa scala di colori per i due gruppi di ogni banda")
    _save(fig, out / "lambda_cov_matrices.png")


# ================================================================== ALL
def make_detail_figures(results_dir, groups, cfg, load_kw, file_filter=None, channel=CHANNEL):
    results_dir = Path(results_dir)
    res = load_results(results_dir / "chaos_results.pkl")
    subj, chan, colors = res["tables"]["subject_metrics"], res["tables"]["channel_metrics"], res["colors"]
    out = results_dir / "figures" / "detail"
    out.mkdir(parents=True, exist_ok=True)
    paper_style(base=11)
    print("Aperiodic spectra (from chaos_results.pkl)")
    fig_aperiodic(res, subj, chan, colors, out, tuple(cfg.APERIODIC_FRANGE))
    print("Rereading the EEG files (envelopes, fE/I windows, covariances)")
    rec = collect(groups, cfg, load_kw, file_filter, channel)
    lt = lambda_table(rec, cfg.LAMBDA_N_UNITS)
    for b in dict.fromkeys(lt.band):
        if f"cov_width_{b}" in subj:
            ref = subj.set_index("subject")[f"cov_width_{b}"]
            diff = np.nanmax(np.abs(lt[lt.band == b].set_index("subject").width - ref))
            print(f"    width {b}: max difference from chaos_results = {diff:.2g}")
    lt.to_csv(out / "lambda_max_origin.csv", index=False)
    for band in signal_bands(cfg):
        fig_dfa(band, rec, subj, colors, out)
        fig_fei(band, rec, subj, chan, colors, out, cfg)
    fig_lambda(lt, colors, out)
    fig_lambda_formula(lt, colors, out)
    fig_cov_matrices(rec, colors, out)
    print(f"Figures in {out}")


if __name__ == "__main__":
    from photosensitive import config_ips as base
    from photosensitive import MAIN_IPS_CHAOS as ips          # applies the IPS overrides to config_chaos
    make_detail_figures(ips.OUT_ROOT, base.GROUPS, ips.cfg,
                        dict(fixed_length_s=base.FIXED_LENGTH_S, window=base.ANALYSIS_WINDOW, channels=None),
                        base.subject_files)
