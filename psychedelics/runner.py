# -*- coding: utf-8 -*-
"""
Shared machinery of the Ayahuasca MAINs: per subject x condition cache, parallel processes,
output folders.

  <OUT_DIR>/cache/<module>/<condition>/<subject>.pkl   one result per subject and condition,
                                                        reused by both analyses (and re-used on
                                                        re-runs if the parameters are the same)
  <OUT_DIR>/<analysis>/<module folder>/                tables and figures of one analysis
"""
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from psychedelics import config_aya as cfg
from psychedelics import io_aya


def cache_path(module, cond, sid):
    return cfg.OUT_DIR / "cache" / module / cond / f"{sid}.pkl"


def analysis_dir(analysis, module):
    d = cfg.OUT_DIR / analysis / cfg.MODULE_DIRS[module]
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def unit_key(cond, sid):
    """
    What identifies the data of one subject x condition: the definition of that condition only
    (folder, file pattern, time window; the other conditions can be added without invalidating
    it), the file itself (path, size, modification time), MAX_EPOCHS and the number of epochs the
    N-dependent indices are equalised to (it depends on the other conditions of the subject).
    """
    real = io_aya.resolve(cond, sid)
    path = io_aya.condition_file(real, sid)
    st = path.stat() if path is not None else None
    return dict(condition=real, definition=repr(cfg.CONDITIONS[real]), file=str(path),
                size=st.st_size if st else None, mtime=round(st.st_mtime) if st else None,
                MAX_EPOCHS=cfg.MAX_EPOCHS, n_equal=io_aya.n_equal(real, sid))


def cached(module, cond, sid, params):
    p = cache_path(module, cond, sid)
    if not p.exists():
        return None
    try:
        res = load_pickle(p)
    except Exception:
        return None
    strip = lambda d: {k: v for k, v in (d or {}).items() if k != "CONDITIONS"}
    if strip(res.get("params")) != strip(params) or res.get("unit") != unit_key(cond, sid):
        return None
    return res


def _job(args):
    """Worker: imports the module MAIN, analyses one subject x condition and caches it."""
    mod_name, module, cond, sid, params = args
    import importlib
    import warnings
    warnings.filterwarnings("ignore")
    mod = importlib.import_module(mod_name)
    t0 = time.time()
    try:
        res = mod.analyze(cond, sid)
    except Exception as e:                      # one bad file must not stop the others
        import traceback
        print(f"  ERROR {module} {cond} {sid}: {e}\n{traceback.format_exc()}", flush=True)
        return None
    res["params"] = params
    res["unit"] = unit_key(cond, sid)
    save_pickle(res, cache_path(module, cond, sid))
    print(f"  {module} {cond} {sid} done in {time.time() - t0:.0f} s", flush=True)
    return sid


def needed_units():
    """Every (condition, subject) used by at least one analysis."""
    units = []
    for analysis, conds in cfg.ANALYSES.items():
        for sid in io_aya.analysis_subjects(analysis):
            for c in conds:
                u = (io_aya.resolve(c, sid), sid)          # virtual conditions -> the real one
                if u not in units:
                    units.append(u)
    return units


def run_all(mod_name, module, params, n_jobs=None, overwrite=False):
    """Analyses every needed subject x condition not yet cached. Returns {(cond, sid): result}."""
    n_jobs = n_jobs or cfg.N_JOBS
    units = needed_units()
    todo = [(c, s) for c, s in units if overwrite or cached(module, c, s, params) is None]
    print(f"{module}: {len(units)} subject x condition, {len(todo)} to analyse, {n_jobs} processes", flush=True)
    jobs = [(mod_name, module, c, s, params) for c, s in todo]
    # biggest files first so the long jobs do not end up last
    if n_jobs > 1 and len(jobs) > 1:
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[v] = "1"
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            list(ex.map(_job, jobs))
    else:
        for j in jobs:
            _job(j)
    out = {}
    for c, s in units:
        r = cached(module, c, s, params)
        if r is not None:
            out[(c, s)] = r
        else:
            print(f"  missing result: {module} {c} {s}")
    return out


def analysis_results(results, analysis):
    """Subset of run_all results for one analysis: list of (cond, sid, result), paired subjects only."""
    conds = cfg.ANALYSES[analysis]
    return _subset(results, conds, io_aya.analysis_subjects(analysis))


def comparison_results(results, comp):
    """Same for a drug x time comparison: the six conditions (Ayahuasca then placebo)."""
    aya, pla, _ = cfg.COMPARISONS[comp]
    return _subset(results, aya + pla, io_aya.comparison_subjects(comp))


def _subset(results, conds, subjects):
    """(cond, sid, result) for the subjects with every condition; virtual conditions keep their name."""
    key = lambda c, s: (io_aya.resolve(c, s), s)
    subs = [s for s in subjects if all(key(c, s) in results for c in conds)]
    return [(c, s, results[key(c, s)]) for s in subs for c in conds], subs


def save_tables_excel(tables, path, order=()):
    from photosensitive.report import save_tables_excel as _save
    tables = {k: v for k, v in tables.items() if v is not None and len(v)}
    try:
        _save(tables, path, order)
    except PermissionError:
        alt = Path(path).with_name(Path(path).stem + "_new.xlsx")
        print(f"WARNING: {Path(path).name} is open in another program, saved as {alt.name}")
        _save(tables, alt, order)
