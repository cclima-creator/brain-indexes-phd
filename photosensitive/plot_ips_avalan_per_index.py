# -*- coding: utf-8 -*-
"""
One image per avalanche index, database IPS: reads avalanche_results.pkl (MAIN_IPS_AVALAN.py, no
re-analysis) and writes <results>/per_index/<index>.png, one panel per band (broadband first),
subjects by group, Mann-Whitney p and FDR across the bands of that index
(descriptive_analysis/index_figures.py). The thematic figures A1-A15 stay in plot_ips_avalan.py.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from photosensitive import config_ips as base
from photosensitive import config_ips_avalan as cfg
from photosensitive.report import load_results
from photosensitive.plot_ips_avalan import find_results
from descriptive_analysis import plots
from descriptive_analysis.index_figures import plot_index_bands

INDEX_LABELS = {
    "n_avalanches": "number of avalanches", "avalanche_rate_hz": "avalanche rate (Hz)",
    "mean_size": "mean size S", "mean_duration_bins": "mean duration T (bins)",
    "tau": "τ (size exponent)", "alpha": "α (duration exponent)", "third": "1/σνz (fitted)",
    "dcc": "DCC", "dcc_clauset": "DCC (Clauset xmin)", "tau_range": "τ (power-law range)",
    "alpha_range": "α (power-law range)", "dcc_range": "DCC (power-law range)",
    "tau_rigorous": "τ (rigorous)", "alpha_rigorous": "α (rigorous)", "third_rigorous": "1/σνz (rigorous)",
    "dcc_rigorous": "DCC (rigorous)",
    "size_p_gof": "power-law p (S)", "duration_p_gof": "power-law p (T)",
    "size_decades": "power-law range, decades (S)", "duration_decades": "power-law range, decades (T)",
    "size_xmax_cutoff": "upper cutoff (S)", "duration_xmax_cutoff": "upper cutoff (T)",
    "size_pct_in_range": "% avalanches in power-law range (S)",
    "duration_pct_in_range": "% avalanches in power-law range (T)",
    "size_mse": "log-log MSE (S)", "duration_mse": "log-log MSE (T)",
    "sigma_bins": "branching ratio σ", "sigma_geometric": "σ geometric", "sigma_ancestors": "σ first generation",
    "repertoire_rate": "repertoire (unique patterns / s)", "repertoire_per_avalanche": "unique patterns / avalanches",
    "diversity_mean": "pattern diversity (mean Hamming)", "ff": "Fano factor", "ff_beta": "Fano factor slope β",
}


def main():
    channel_set, _, out_root = base.choose_channel_set()      # same choice as MAIN_IPS_AVALAN
    out = find_results(cfg.results_dir(channel_set, out_root))
    if out is None:
        raise SystemExit("avalanche_results.pkl not found: run MAIN_IPS_AVALAN.py first")
    res = load_results(out / "avalanche_results.pkl")
    metrics, bands = res["metrics"], list(res["bands"])
    colors = {g: c for g, c in cfg.COLORS.items() if g in set(metrics.group)}
    plots.paper_style()
    d = out / "per_index"
    d.mkdir(exist_ok=True)
    for col, lab in INDEX_LABELS.items():
        if col in metrics:
            plot_index_bands(metrics, col, lab, bands, colors, d / f"{col}.png",
                             title=f"{lab} per band (Mann-Whitney; FDR across bands)")
    print("Figures saved in:", d)


if __name__ == "__main__":
    main()
