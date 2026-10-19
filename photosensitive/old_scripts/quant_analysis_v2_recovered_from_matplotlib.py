"""
Analisi quantitativa EEG - stimolazione fotica intermittente (IPS)
Gruppo fotosensibile vs non fotosensibile.

Ogni epoca (11 s) = una frequenza di stimolazione. La frequenza viene letta
dall'etichetta dell'evento dell'epoca (vedi parse_freq / STIM_FREQ_MAP).

Output (in OUT_DIR):
  XLSX: risultati_eeg.xlsx  (un foglio per tabella, colonne separate)
  HTML interattivi (hover = nome soggetto, click in legenda = nascondi/mostra):
        fig1_psd_all_subjects_interactive.html
        fig4_snr_vs_stim_freq_interactive.html
  PNG : fig1_psd_all_subjects, fig2_psd_each_subject, fig3_epoch_maps,
        fig4_snr_vs_stim_freq, fig5_hist_gauss_bands, fig6_relative_power_boxplot

Requisiti: pip install mne numpy scipy pandas matplotlib plotly openpyxl
"""
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import welch
from scipy.integrate import trapezoid
from openpyxl.utils import get_column_letter

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAVE_PLOTLY = True
except ImportError:
    HAVE_PLOTLY = False
    print("plotly non installato: grafici interattivi saltati (pip install plotly)")

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

# ============================================================== CONFIG
PHOTOSENSITIVE_FOLDER = r"C:\Users\Chiara Clima\Desktop\eeg preprocess criticality\eeg photosensitive\removed epo + ICA + removed bad IC"
NOT_PHOTOSENSITIVE_FOLDER = r"C:\Users\Chiara Clima\Desktop\eeg preprocess criticality\eeg not photosensitive\removed epo + ICA + removed bad IC"
GROUPS = {"photosensitive": PHOTOSENSITIVE_FOLDER,
          "not_photosensitive": NOT_PHOTOSENSITIVE_FOLDER}
OUT_DIR = Path(r"C:\Users\Chiara Clima\Desktop\eeg preprocess criticality\quant_analysis_output_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_NAMES = ["fig1_psd_all_subjects.png", "fig2_psd_each_subject.png", "fig3_epoch_maps.png",
                "fig4_snr_vs_stim_freq.png", "fig5_hist_gauss_bands.png",
                "fig6_relative_power_boxplot.png", "fig1_psd_all_subjects_interactive.html",
                "fig4_snr_vs_stim_freq_interactive.html", "risultati_eeg.xlsx"]


def _can_write(folder):
    """Controlla SUBITO di poter scrivere tutti i file di output (prima dell'analisi lunga)."""
    try:
        folder.mkdir(parents=True, exist_ok=True)
        for n in OUTPUT_NAMES:
            t = folder / n
            t.write_bytes(b"test")
            t.unlink()
        return True
    except OSError as e:
        print(f"ATTENZIONE: non riesco a scrivere in {folder}\n  -> {e}")
        return False


if not _can_write(OUT_DIR):
    OUT_DIR = Path.home() / "eeg_quant_output"
    print("Provo la cartella alternativa:", OUT_DIR)
    if not _can_write(OUT_DIR):
        raise SystemExit("Nessuna cartella scrivibile: imposta OUT_DIR a un percorso valido (es. C:/eeg_out)")
print("Cartella di output:", OUT_DIR)

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 45)}
BAND_TOTAL = (1.0, 45.0)        # range per la potenza totale (potenza relativa)
FMIN, FMAX_PSD = 1.0, 70.0      # range PSD (serve per le armoniche)

# finestra analizzata dentro l'epoca (s). None = epoca intera.
# Se l'epoca contiene un pre-stimolo usa es. (1.0, 11.0). Con finestre di
# 10 o 11 s le frequenze intere cadono esattamente su un bin della FFT.
ANALYSIS_WINDOW = None

# se le etichette degli eventi non contengono la frequenza, mappale qui:
# STIM_FREQ_MAP = {"S 11": 10.0, "S 12": 12.0, ...}
STIM_FREQ_MAP = {}

# ROI dove ci si aspetta la risposta alla stimolazione (se nessun canale
# corrisponde, usa tutti i canali)
ROI_CHANNELS = ["O1", "O2", "Oz", "PO3", "PO4", "POz", "PO7", "PO8", "P3", "P4", "Pz"]

# SNR a banda stretta: rapporto tra la potenza al bin di stimolazione e la
# media dei bin vicini (da NB_GAP a NB_WIDTH Hz di distanza, da entrambi i lati)
NB_GAP, NB_WIDTH = 0.3, 1.5
N_HARMONICS = 3

COLORS = {"photosensitive": "tab:red", "not_photosensitive": "tab:blue"}
COMMON_F = np.arange(FMIN, FMAX_PSD, 0.1)
RGBA = {"photosensitive": "rgba(214,39,40,0.45)", "not_photosensitive": "rgba(31,119,180,0.45)"}
RGB = {"photosensitive": "rgb(214,39,40)", "not_photosensitive": "rgb(31,119,180)"}

tables = {}          # DataFrame da scrivere nel file Excel (un foglio ciascuna)
SHEET_ORDER = ["group_summary_bands", "group_tests_mannwhitney", "subject_level_metrics",
               "snr_by_stim_freq_group", "epoch_table_snr", "band_power_per_subject",
               "band_power_per_channel"]


def save_tables(tables, path):
    """Scrive un unico .xlsx con un foglio per tabella (colonne separate)."""
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name in SHEET_ORDER + [k for k in tables if k not in SHEET_ORDER]:
            if name not in tables:
                continue
            df = tables[name]
            sheet = name[:31]
            df.to_excel(xw, sheet_name=sheet, index=False)
            ws = xw.sheets[sheet]
            ws.freeze_panes = "A2"
            for i, col in enumerate(df.columns, start=1):
                widths = [len(str(col))] + [len(str(v)) for v in df[col].head(200)]
                ws.column_dimensions[get_column_letter(i)].width = min(max(widths) + 2, 40)


# ============================================================== UTILITA'
def parse_freq(label):
    """Estrae la frequenza (Hz) da un'etichetta evento, es. '12Hz', 'stim_12', '12'."""
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*hz", str(label), re.I)
    if m:
        return float(m.group(1).replace(",", "."))
    nums = re.findall(r"\d+(?:\.\d+)?", str(label))
    return float(nums[-1]) if nums else np.nan


def band_power(psd, freqs, lo, hi):
    m = (freqs >= lo) & (freqs < hi)
    return trapezoid(psd[..., m], freqs[m], axis=-1)


def snr_at(freqs, psd, f0):
    """SNR (dB) a f0 per ogni canale. psd: (n_ch, n_f)."""
    n_ch = psd.shape[0]
    if not np.isfinite(f0) or f0 > freqs.max():
        return np.full(n_ch, np.nan)
    df = freqs[1] - freqs[0]
    i0 = np.argmin(np.abs(freqs - f0))
    if abs(freqs[i0] - f0) > df:
        return np.full(n_ch, np.nan)
    d = np.abs(freqs - f0)
    nb = (d >= NB_GAP) & (d <= NB_WIDTH)
    if nb.sum() < 4:
        return np.full(n_ch, np.nan)
    return 10 * np.log10(psd[:, i0] / psd[:, nb].mean(1))


# ============================================================== PER SOGGETTO
def analyze_file(path, group):
    sid = Path(path).stem
    ep = mne.io.read_epochs_eeglab(path)
    ep.pick("eeg")
    if ANALYSIS_WINDOW is not None:
        ep.crop(ANALYSIS_WINDOW[0], ANALYSIS_WINDOW[1], include_tmax=False)
    sf = ep.info["sfreq"]
    names = ep.ch_names
    data = ep.get_data() * 1e6                              # (n_ep, n_ch, n_t) in uV
    n_ep, n_ch, n_t = data.shape

    # PSD per epoca: un solo segmento = intera finestra (risoluzione 1/T Hz)
    f, p = welch(data, fs=sf, window="hann", nperseg=n_t, noverlap=0, axis=-1)
    fmax = min(FMAX_PSD, sf / 2 - 1)
    m = (f >= FMIN) & (f <= fmax)
    f, p = f[m], p[..., m]                                  # p: (n_ep, n_ch, n_f) uV^2/Hz

    # frequenza di stimolazione per epoca
    inv = {v: k for k, v in ep.event_id.items()}
    labels = [inv.get(c, str(c)) for c in ep.events[:, 2]]
    fstim = np.array([STIM_FREQ_MAP.get(l, parse_freq(l)) for l in labels], dtype=float)
    print(f"    etichette: {sorted(set(labels))[:8]}{' ...' if len(set(labels)) > 8 else ''}")

    # ROI
    roi_up = {c.upper() for c in ROI_CHANNELS}
    roi = [i for i, c in enumerate(names) if c.upper() in roi_up]
    if not roi:
        print("    ATTENZIONE: nessun canale ROI trovato, uso tutti i canali")
        roi = list(range(n_ch))

    # --- SNR a banda stretta per epoca
    ep_rows = []
    for k in range(n_ep):
        row = dict(group=group, subject=sid, epoch=k, label=labels[k], stim_freq=fstim[k])
        for h in range(1, N_HARMONICS + 1):
            s = snr_at(f, p[k], fstim[k] * h)
            row[f"snr_h{h}_all_dB"] = np.nanmean(s)
            row[f"snr_h{h}_roi_dB"] = np.nanmean(s[roi])
        ep_rows.append(row)

    # --- PSD media del soggetto (media lineare su epoche)
    psd_sub = p.mean(0)                                     # (n_ch, n_f)
    tot = band_power(psd_sub, f, BAND_TOTAL[0], BAND_TOTAL[1] + 1e-6)
    long_rows = []
    for b, (lo, hi) in BANDS.items():
        bp = band_power(psd_sub, f, lo, hi)
        for i, ch in enumerate(names):
            long_rows.append(dict(group=group, subject=sid, channel=ch, band=b,
                                  abs_uV2=bp[i], rel=bp[i] / tot[i],
                                  log10_abs=np.log10(bp[i])))

    # --- metriche di soggetto
    x = data.transpose(1, 0, 2).reshape(n_ch, -1)
    sig = (f >= 1) & (f <= 30)
    noi = (f >= 40) & (f <= 45)
    snr_bb = (10 * np.log10(psd_sub[:, sig].mean(1) / psd_sub[:, noi].mean(1)).mean()
              if noi.sum() else np.nan)
    summary = dict(group=group, subject=sid, sfreq=sf, n_channels=n_ch, n_epochs=n_ep,
                   n_roi_channels=len(roi), epoch_len_s=n_t / sf,
                   rms_uV=np.sqrt((x ** 2).mean(1)).mean(), std_uV=x.std(1).mean(),
                   median_abs_uV=np.median(np.abs(x), axis=1).mean(),
                   skew_time=stats.skew(x, axis=1).mean(),
                   kurtosis_time=stats.kurtosis(x, axis=1).mean(),
                   snr_broadband_dB=snr_bb)

    store = dict(group=group, subject=sid, f=f, psd=psd_sub.mean(0),
                 heat=np.log10(p[:, roi, :].mean(1)), fstim=fstim)
    return long_rows, ep_rows, summary, store


# ============================================================== LOOP
long_rows, epoch_rows, subj_rows, store = [], [], [], []
for g, folder in GROUPS.items():
    files = sorted(Path(folder).glob("*.set"))
    print(f"{g}: {len(files)} file")
    for fpath in files:
        print("  ", fpath.name)
        try:
            lr, er, sm, st = analyze_file(str(fpath), g)
            long_rows += lr; epoch_rows += er; subj_rows.append(sm); store.append(st)
        except Exception as e:
            print("    ERRORE ->", e)

long = pd.DataFrame(long_rows)
epoch_df = pd.DataFrame(epoch_rows)
subj_meta = pd.DataFrame(subj_rows)
tables["band_power_per_channel"] = long
tables["epoch_table_snr"] = epoch_df

# SNR medio per soggetto (media sulle epoche) e per frequenza di stimolo
snr_cols = [c for c in epoch_df.columns if c.startswith("snr_")]
snr_subj = epoch_df.groupby(["group", "subject"])[snr_cols].mean().reset_index()
subj_meta = subj_meta.merge(snr_subj, on=["group", "subject"], how="left")
tables["subject_level_metrics"] = subj_meta

snr_freq = (epoch_df.groupby(["group", "subject", "stim_freq"])[snr_cols].mean().reset_index())
snr_freq_group = (snr_freq.groupby(["group", "stim_freq"])[snr_cols]
                  .agg(["mean", "median", "count"]))
snr_freq_group.columns = ["_".join(c) for c in snr_freq_group.columns]
tables["snr_by_stim_freq_group"] = snr_freq_group.reset_index()

subj = (long.groupby(["group", "subject", "band"])[["abs_uV2", "rel", "log10_abs"]]
        .mean().reset_index())
tables["band_power_per_subject"] = subj


# ============================================================== STATISTICHE
def safe_shapiro(x):
    x = np.asarray(x.dropna())
    return stats.shapiro(x).pvalue if len(x) >= 3 else np.nan


summ = []
for (g, b), s in subj.groupby(["group", "band"]):
    ch = long[(long.group == g) & (long.band == b)]
    mu, sd = stats.norm.fit(ch.log10_abs)
    summ.append(dict(
        group=g, band=b, n_subjects=len(s),
        abs_mean=s.abs_uV2.mean(), abs_median=s.abs_uV2.median(), abs_sd=s.abs_uV2.std(),
        rel_mean=s.rel.mean(), rel_median=s.rel.median(), rel_sd=s.rel.std(),
        log10_mean=s.log10_abs.mean(), log10_median=s.log10_abs.median(),
        log10_sd=s.log10_abs.std(),
        gauss_mu_log10_channels=mu, gauss_sigma_log10_channels=sd,
        skew_channels_linear=stats.skew(ch.abs_uV2),
        skew_channels_log10=stats.skew(ch.log10_abs),
        shapiro_p_subjects_linear=safe_shapiro(s.abs_uV2),
        shapiro_p_subjects_log10=safe_shapiro(s.log10_abs)))
tables["group_summary_bands"] = pd.DataFrame(summ)

tests = []
g1, g2 = list(GROUPS)
for b in BANDS:
    for metric in ["log10_abs", "rel"]:
        a = subj[(subj.group == g1) & (subj.band == b)][metric].dropna()
        c = subj[(subj.group == g2) & (subj.band == b)][metric].dropna()
        if len(a) > 1 and len(c) > 1:
            tests.append(dict(metric=f"{metric}_{b}", n1=len(a), n2=len(c),
                              p_mannwhitney=stats.mannwhitneyu(a, c).pvalue))
for m in ["snr_broadband_dB", "rms_uV", "std_uV", "skew_time", "kurtosis_time"] + snr_cols:
    a = subj_meta[subj_meta.group == g1][m].dropna()
    c = subj_meta[subj_meta.group == g2][m].dropna()
    if len(a) > 1 and len(c) > 1:
        tests.append(dict(metric=m, n1=len(a), n2=len(c),
                          p_mannwhitney=stats.mannwhitneyu(a, c).pvalue))
tables["group_tests_mannwhitney"] = pd.DataFrame(tests)

# ============================================================== FIGURE
# 1) PSD di tutti i soggetti + media di gruppo (±SEM tra soggetti)
fig, ax = plt.subplots(figsize=(9, 6))
for s in store:
    ax.loglog(s["f"], s["psd"], color=COLORS[s["group"]], lw=0.6, alpha=0.4)
for g in GROUPS:
    A = np.array([np.interp(COMMON_F, s["f"], s["psd"]) for s in store if s["group"] == g])
    if len(A) == 0:
        continue
    mean = A.mean(0)
    sem = A.std(0, ddof=1) / np.sqrt(len(A)) if len(A) > 1 else 0 * mean
    ax.loglog(COMMON_F, mean, color=COLORS[g], lw=2.5, label=f"{g} (n={len(A)})")
    ax.fill_between(COMMON_F, np.clip(mean - sem, 1e-12, None), mean + sem,
                    color=COLORS[g], alpha=0.2)
ax.set(xlabel="Frequenza (Hz)", ylabel="PSD (uV²/Hz)",
       title="PSD per soggetto (linee sottili) e media di gruppo ± SEM")
ax.legend()
fig.tight_layout(); fig.savefig(OUT_DIR / "fig1_psd_all_subjects.png", dpi=200); plt.close(fig)


def grid(n, ncols=5, h=2.6, w=3.4):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    return fig, axes.ravel()


# 2) PSD di ogni soggetto (small multiples)
fig, axes = grid(len(store))
for ax, s in zip(axes, store):
    ax.loglog(s["f"], s["psd"], color=COLORS[s["group"]], lw=1)
    ax.set_title(s["subject"][:22], fontsize=7, color=COLORS[s["group"]])
    ax.tick_params(labelsize=6)
fig.suptitle("PSD per soggetto (media su epoche e canali) - rosso: fotosensibili, blu: non fotosensibili")
fig.tight_layout(); fig.savefig(OUT_DIR / "fig2_psd_each_subject.png", dpi=200); plt.close(fig)

# 3) Mappe epoca x frequenza (ROI), epoche ordinate per frequenza di stimolo
fig, axes = grid(len(store))
for ax, s in zip(axes, store):
    order = np.argsort(np.where(np.isfinite(s["fstim"]), s["fstim"], np.inf))
    H = s["heat"][order]
    fm = s["f"] <= 60
    ax.imshow(H[:, fm], aspect="auto", origin="upper", cmap="viridis",
              extent=[s["f"][fm][0], s["f"][fm][-1], len(order), 0],
              vmin=np.percentile(H[:, fm], 5), vmax=np.percentile(H[:, fm], 99.5))
    ys = np.arange(len(order)) + 0.5
    ax.scatter(s["fstim"][order], ys, s=6, c="white", marker="|", linewidths=0.8)
    ax.set_yticks(ys[::2])
    ax.set_yticklabels([f"{v:g}" for v in s["fstim"][order][::2]], fontsize=5)
    ax.set_title(s["subject"][:22], fontsize=7, color=COLORS[s["group"]])
    ax.tick_params(axis="x", labelsize=6)
fig.suptitle("log10 PSD (ROI): asse y = frequenza di stimolo (tacca bianca = frequenza attesa), asse x = Hz")
fig.tight_layout(); fig.savefig(OUT_DIR / "fig3_epoch_maps.png", dpi=200); plt.close(fig)

# 4) SNR alla frequenza di stimolo (e 2a armonica) in funzione della frequenza
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for ax, col, ttl in zip(axes, ["snr_h1_roi_dB", "snr_h2_roi_dB"],
                        ["Fondamentale (f)", "Seconda armonica (2f)"]):
    for g in GROUPS:
        piv = snr_freq[snr_freq.group == g].pivot(index="subject", columns="stim_freq", values=col)
        for _, r in piv.iterrows():
            ax.plot(piv.columns, r.values, color=COLORS[g], lw=0.6, alpha=0.3)
        n = piv.count()
        mean = piv.mean()
        sem = piv.std(ddof=1) / np.sqrt(n)
        ok = n >= 2
        ax.errorbar(piv.columns[ok], mean[ok], yerr=sem[ok], color=COLORS[g], lw=2,
                    marker="o", ms=3, capsize=2, label=f"{g} (n={piv.shape[0]})")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set(title=ttl, xlabel="Frequenza di stimolazione (Hz)")
axes[0].set_ylabel("SNR ROI (dB)")
axes[0].legend()
fig.suptitle("SNR a banda stretta: potenza al bin di stimolo / media dei bin vicini")
fig.tight_layout(); fig.savefig(OUT_DIR / "fig4_snr_vs_stim_freq.png", dpi=200); plt.close(fig)

# ---- GRAFICI INTERATTIVI (plotly): hover = soggetto, click in legenda = nascondi/mostra
if HAVE_PLOTLY:
    # PSD di tutti i soggetti
    fi = go.Figure()
    for s in store:
        fi.add_trace(go.Scatter(
            x=s["f"], y=s["psd"], mode="lines", name=s["subject"], legendgroup=s["group"],
            line=dict(color=RGBA[s["group"]], width=1.3),
            hovertemplate="%{x:.2f} Hz<br>%{y:.3g} uV²/Hz<extra>" + s["subject"] + " | " + s["group"] + "</extra>"))
    for g in GROUPS:
        A = np.array([np.interp(COMMON_F, s["f"], s["psd"]) for s in store if s["group"] == g])
        if len(A):
            fi.add_trace(go.Scatter(x=COMMON_F, y=A.mean(0), mode="lines", name=f"MEDIA {g}",
                                    line=dict(color=RGB[g], width=4),
                                    hovertemplate="%{x:.2f} Hz<br>%{y:.3g}<extra>media " + g + "</extra>"))
    fi.update_layout(template="plotly_white", hovermode="closest",
                     xaxis=dict(type="log", title="Frequenza (Hz)"),
                     yaxis=dict(type="log", title="PSD (uV²/Hz)"),
                     title="PSD per soggetto (doppio click in legenda = isola un soggetto)")
    fi.write_html(OUT_DIR / "fig1_psd_all_subjects_interactive.html", include_plotlyjs=True)

    # SNR vs frequenza di stimolo (f e 2f), un trace per soggetto collegato tra i due pannelli
    cols = ["snr_h1_roi_dB", "snr_h2_roi_dB"]
    fs = make_subplots(rows=1, cols=2, shared_yaxes=True,
                       subplot_titles=("Fondamentale (f)", "Seconda armonica (2f)"))
    for j, col in enumerate(cols, start=1):
        for (g, sid), d in snr_freq.groupby(["group", "subject"]):
            d = d.sort_values("stim_freq")
            fs.add_trace(go.Scatter(
                x=d["stim_freq"], y=d[col], mode="lines+markers", name=sid, legendgroup=sid,
                showlegend=(j == 1), marker=dict(size=5), line=dict(color=RGBA[g], width=1.3),
                hovertemplate="stim %{x:g} Hz<br>SNR %{y:.1f} dB<extra>" + sid + " | " + g + "</extra>"),
                row=1, col=j)
        for g in GROUPS:
            mm = snr_freq[snr_freq.group == g].groupby("stim_freq")[col].mean()
            fs.add_trace(go.Scatter(
                x=mm.index, y=mm.values, mode="lines+markers", name=f"MEDIA {g}",
                legendgroup=f"MEDIA {g}", showlegend=(j == 1),
                line=dict(color=RGB[g], width=4), marker=dict(size=7),
                hovertemplate="stim %{x:g} Hz<br>SNR medio %{y:.1f} dB<extra>media " + g + "</extra>"),
                row=1, col=j)
    fs.update_layout(template="plotly_white", hovermode="closest",
                     title="SNR ROI alla frequenza di stimolazione (click in legenda = nascondi/mostra)")
    fs.update_xaxes(title_text="Frequenza di stimolazione (Hz)")
    fs.update_yaxes(title_text="SNR (dB)", row=1, col=1)
    fs.write_html(OUT_DIR / "fig4_snr_vs_stim_freq_interactive.html", include_plotlyjs=True)

# 5) Istogrammi log10 potenza per banda + fit gaussiano
fig, axes = plt.subplots(1, len(BANDS), figsize=(4 * len(BANDS), 3.8))
for ax, b in zip(axes, BANDS):
    for g in GROUPS:
        v = long[(long.group == g) & (long.band == b)].log10_abs.dropna()
        if len(v) < 3:
            continue
        mu, sd = stats.norm.fit(v)
        xs = np.linspace(v.min(), v.max(), 200)
        ax.hist(v, bins=30, density=True, alpha=0.4, color=COLORS[g])
        ax.plot(xs, stats.norm.pdf(xs, mu, sd), color=COLORS[g], lw=2,
                label=f"{g[:5]}.. μ={mu:.2f} σ={sd:.2f}")
    ax.set(title=b, xlabel="log10 potenza (uV²)")
    ax.legend(fontsize=7)
axes[0].set_ylabel("Densità")
fig.suptitle("Distribuzione della potenza per banda (canali x soggetti) con fit gaussiano")
fig.tight_layout(); fig.savefig(OUT_DIR / "fig5_hist_gauss_bands.png", dpi=200); plt.close(fig)

# 6) Potenza relativa per banda
fig, ax = plt.subplots(figsize=(9, 5))
w = 0.35
for k, g in enumerate(GROUPS):
    data_b = [subj[(subj.group == g) & (subj.band == b)].rel.values for b in BANDS]
    pos = np.arange(len(BANDS)) + (k - 0.5) * w
    bp = ax.boxplot(data_b, positions=pos, widths=w * 0.9, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS[g]); patch.set_alpha(0.5)
    ax.plot([], [], color=COLORS[g], label=g)
ax.set_xticks(range(len(BANDS))); ax.set_xticklabels(BANDS.keys())
ax.set(ylabel="Potenza relativa", title="Potenza relativa per banda (soggetti)")
ax.legend()
fig.tight_layout(); fig.savefig(OUT_DIR / "fig6_relative_power_boxplot.png", dpi=200); plt.close(fig)

save_tables(tables, OUT_DIR / "risultati_eeg.xlsx")
print("\nFatto. Output in:", OUT_DIR)