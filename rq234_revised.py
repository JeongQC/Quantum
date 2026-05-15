"""rq234_revised.py — RQ2 / RQ3 / RQ4 QA experiment driver (revised).

Phase-1 of paper revision. Extracted from RQ234.ipynb and updated per
TASK A–D:

  TASK A  Endpoint / token / solver come from environment variables only.
          No real token is ever embedded in code.
  TASK B  SAMPLER_MODE env var ("fixed_embedding" | "clique_sampler") chooses
          between FixedEmbeddingComposite + minorminer and DWaveCliqueSampler.
          sampler_mode is recorded in CSV rows, output directory name, and
          a JSON metadata file (Phase 4 will flesh this out further — see
          TODO comments).
  TASK C  CBF and energy stats are weighted by num_occurrences via the
          summarize_cbf / summarize_energy helpers in qa_utils.
  TASK D  QPU timing is captured per-replicate via extract_qpu_timing and
          aggregated in the per-L summary CSV.

Layout (top-to-bottom):
  1. Imports & env-var configuration
  2. ICE / CBF model + grid-search fitter
  3. Sampling primitives (FixedEmbedding / CliqueSampler) — return timing
  4. Per-L driver (one BQM, R replicates)
  5. Experiment driver (sweep over L, fit ICE, save CSV/NPZ/PDF)
  6. Chain-strength sweep helper
  7. Plotting (PDF, no inline magic)
  8. Phase-1 sanity checks
  9. CLI entrypoint
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# 1. Imports & configuration
# -----------------------------------------------------------------------------
import argparse
import itertools
import json
import os
import sys
import time
from math import erf, sqrt
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless safe — no inline magic
import matplotlib.pyplot as plt

from qa_utils import (
    ENDPOINT, TOKEN, SOLVER, MASTER_SEED,
    DWAVE_AVAILABLE,
    apply_anneal_params,
    chain_lengths_from_embedding,
    chain_lengths_from_sampleset,
    expand_by_occurrences,
    extract_qpu_timing,
    find_bqm_embedding,
    find_clique_embedding,
    get_clique_sampler,
    get_L_list_from_clique,
    get_qpu,
    log,
    make_random_bqm,
    sample_with_retry,
    seed_everything,
    summarize_cbf,
    summarize_energy,
    tqdm,
    TIMING_FIELDS,
)

# Optional Ocean imports for direct sampler use
if DWAVE_AVAILABLE:
    from dwave.embedding.chain_strength import uniform_torque_compensation
    from dwave.system import FixedEmbeddingComposite
else:
    uniform_torque_compensation = None
    FixedEmbeddingComposite = None

# Reproducibility
seed_everything(MASTER_SEED)

# TASK B: explicit sampler-mode env var. Default matches the manuscript.
SAMPLER_MODE = os.getenv("SAMPLER_MODE", "fixed_embedding")
ALLOWED_SAMPLER_MODES = ("fixed_embedding", "clique_sampler")
# Internal: "bqm_embedding" is preserved from the original notebook for
# backward compatibility but is intentionally not in ALLOWED_SAMPLER_MODES.
_LEGACY_SAMPLER_MODES = ("bqm_embedding",)


# -----------------------------------------------------------------------------
# 2. ICE / CBF model + grid-search fitter
# -----------------------------------------------------------------------------
def sigma_eff_sq(ell, sigma_h, sigma_c):
    return ell * (sigma_h ** 2) + max(0, ell - 1) * (sigma_c ** 2)


def p_break_exact(ell, sigma_h, sigma_c, kappa):
    s2 = sigma_eff_sq(int(ell), sigma_h, sigma_c)
    if s2 <= 0:
        return 0.0
    x = kappa / (sqrt(2.0) * sqrt(s2))
    return 1.0 - erf(x)


def cbf_from_lengths(lengths, sigma_h, sigma_c, kappa):
    if len(lengths) == 0:
        return 0.0
    return float(np.mean([p_break_exact(ell, sigma_h, sigma_c, kappa)
                          for ell in lengths]))


def refine_sigma_params(L_list, cbf_obs_list, chain_lengths_dict,
                        sh0, sc0, kappa0,
                        sh_bounds=(0.001, 0.15),
                        sc_bounds=(0.001, 0.15),
                        kappa_bounds=(0.05, 1.5),
                        max_iter=500, verbose=False):
    """Local refinement around (sh0, sc0, kappa0) via Nelder-Mead.

    Objective: SSE between observed mean CBF and predicted mean CBF, where
    the prediction at each L is the mean of CBP(ell_i; params) over the
    chain-length vector for that L (same model as ``fit_sigma_params``;
    no Jensen approximation).

    Bounds enforced by clipping the parameter vector inside the objective,
    so the optimizer always sees a finite SSE on its (possibly out-of-bound)
    proposed steps.

    Note: the function takes ``chain_lengths_dict`` rather than the
    mean-chain-length-per-L list, because using only the mean would change
    the model from "mean of CBP over chains" to "CBP at mean chain length"
    (Jensen-different) and make this refinement inconsistent with the grid
    fit. The argument name is corrected accordingly.

    Returns
    -------
    dict with keys sigma_h, sigma_c, kappa, sse, n_iter, converged.
    """
    from scipy.optimize import minimize

    L_arr = [int(L) for L in L_list]
    cbf_arr = np.asarray(cbf_obs_list, dtype=float)
    bounds = [sh_bounds, sc_bounds, kappa_bounds]

    def _clip(x):
        return np.array([np.clip(v, lo, hi)
                         for v, (lo, hi) in zip(x, bounds)], dtype=float)

    def loss(x):
        sh, sc, k = _clip(x)
        sse = 0.0
        for i, L in enumerate(L_arr):
            ell_vec = chain_lengths_dict[int(L)]
            pred = cbf_from_lengths(ell_vec, sh, sc, k)
            sse += (pred - cbf_arr[i]) ** 2
        return sse

    x0 = np.array([float(sh0), float(sc0), float(kappa0)])
    res = minimize(loss, x0, method="Nelder-Mead",
                   options={"maxiter": max_iter,
                            "xatol": 1e-8, "fatol": 1e-12,
                            "adaptive": True})
    sh_f, sc_f, k_f = _clip(res.x).tolist()
    sse_f = float(loss([sh_f, sc_f, k_f]))
    if verbose:
        log(f"[refine] start ({sh0:.5f}, {sc0:.5f}, {kappa0:.5f}) "
            f"-> ({sh_f:.5f}, {sc_f:.5f}, {k_f:.5f})  "
            f"SSE {loss(x0):.4e} -> {sse_f:.4e}  "
            f"iters={res.nit} success={res.success}")
    return dict(sigma_h=float(sh_f), sigma_c=float(sc_f),
                kappa=float(k_f), sse=sse_f,
                n_iter=int(res.nit), converged=bool(res.success))


def fit_sigma_params(calib_rows,
                     sh_grid=None, sc_grid=None, k_grid=None,
                     verbose=True):
    if sh_grid is None:
        sh_grid = np.linspace(0.005, 0.08, 16)
    if sc_grid is None:
        sc_grid = np.linspace(0.005, 0.08, 16)
    if k_grid is None:
        k_grid = np.linspace(0.10, 1.00, 19)

    best, best_err = None, float("inf")
    for sh, sc, k in itertools.product(sh_grid, sc_grid, k_grid):
        err = 0.0
        for row in calib_rows:
            pred = cbf_from_lengths(row["lengths"], sh, sc, k)
            err += (pred - row["cbf_obs"]) ** 2
        if err < best_err:
            best_err = err
            best = (float(sh), float(sc), float(k))

    if verbose and best is not None:
        log(f"[fit] sigma_h={best[0]:.4f}, sigma_c={best[1]:.4f}, "
            f"kappa={best[2]:.4f} | SSE={best_err:.6f}")

    table = []
    for row in calib_rows:
        pred = cbf_from_lengths(row["lengths"], best[0], best[1], best[2])
        table.append(dict(L=row["L"], cbf_obs=row["cbf_obs"], cbf_pred=float(pred)))
    return dict(sigma_h=best[0], sigma_c=best[1], kappa=best[2],
                sse=best_err, table=table)


# -----------------------------------------------------------------------------
# 3. Sampling primitives  (TASK D: return timing dict)
# -----------------------------------------------------------------------------
def run_qa_with_embedding(bqm, emb, num_reads=2000, chain_strength=None,
                          anneal_time_us=None, anneal_schedule=None,
                          verbose=False):
    """FixedEmbeddingComposite + QPU sample.

    Returns
    -------
    sampleset, cbf_stats, energy_stats, client_wall_time_sec, timing
    """
    qpu = get_qpu()
    comp = FixedEmbeddingComposite(qpu, emb)

    comp_kwargs = dict(
        num_reads=num_reads,
        chain_strength=chain_strength,
        chain_break_fraction=True,
    )
    comp_kwargs.update(apply_anneal_params(
        qpu, {}, anneal_time_us, anneal_schedule, verbose))

    if verbose:
        log(f"[sample kwargs] {comp_kwargs}")

    t0 = time.time()
    ss = sample_with_retry(comp.sample, bqm, **comp_kwargs)
    dt = time.time() - t0

    cbf = summarize_cbf(ss)
    e_stats = summarize_energy(ss)
    timing = extract_qpu_timing(ss)
    return ss, cbf, e_stats, dt, timing


def run_qa_via_clique_sampler(bqm, num_reads=2000, chain_strength=None,
                              anneal_time_us=None, anneal_schedule=None,
                              verbose=False):
    """DWaveCliqueSampler sample.

    Returns
    -------
    sampleset, cbf_stats, energy_stats, client_wall_time_sec, timing,
    chain_lengths
    """
    qpu = get_qpu()  # only used to negotiate solver params
    cs = get_clique_sampler()

    solver_kwargs = apply_anneal_params(
        qpu, {}, anneal_time_us, anneal_schedule, verbose)

    t0 = time.time()
    ss = sample_with_retry(
        cs.sample, bqm,
        num_reads=num_reads,
        chain_strength=chain_strength,
        chain_break_fraction=True,
        **solver_kwargs,
    )
    dt = time.time() - t0

    cbf = summarize_cbf(ss)
    e_stats = summarize_energy(ss)
    timing = extract_qpu_timing(ss)
    lengths = chain_lengths_from_sampleset(ss)

    if verbose:
        if lengths.size:
            log(f"[CliqueSampler] meanCL={np.mean(lengths):.2f}, "
                f"maxCL={int(np.max(lengths))}")
        log(f"[sample kwargs] {dict(num_reads=num_reads, chain_strength=chain_strength, **solver_kwargs)}")

    return ss, cbf, e_stats, dt, timing, lengths


# -----------------------------------------------------------------------------
# 4. Per-L driver — one BQM, R replicates
# -----------------------------------------------------------------------------
def _per_rep_row(L, replicate, sampler_mode, qubo_seed, emb_seed,
                 chain_strength, cbf, e_stats, dt_wall, n_reads,
                 anneal_time_us, timing, mean_chainlen, max_chainlen):
    """Build a per-replicate CSV row (TASK B: includes sampler_mode + seeds;
    TASK D: includes timing)."""
    row = dict(
        L=L,
        replicate=replicate,
        sampler_mode=sampler_mode,
        qubo_seed=int(qubo_seed),
        emb_seed=int(emb_seed),
        chain_strength=float(chain_strength),
        mean_cbf=cbf["mean"],
        std_cbf=cbf["std"],
        prob_break=cbf["prob_break"],
        n_reads=int(n_reads),
        mean_energy=e_stats["mean"],
        std_energy=e_stats["std"],
        best_energy=e_stats["best"],
        mean_chainlen=float(mean_chainlen) if mean_chainlen is not None else np.nan,
        max_chainlen=int(max_chainlen) if max_chainlen is not None else -1,
        client_wall_time_sec=float(dt_wall),
        anneal_time_us=float(anneal_time_us) if anneal_time_us is not None else np.nan,
    )
    row.update({k: timing.get(k, np.nan) for k in TIMING_FIELDS})
    return row


def _append_progress_row(path, row_dict):
    df = pd.DataFrame([row_dict])
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)


def measure_L_oneQUBO(L, *,
                      qubo_seed=0, emb_seed=0,
                      n_reps=10, num_reads=2000, density=1.0,
                      chain_strength_mode="utc",
                      anneal_time_us=None, anneal_schedule=None,
                      sampler_mode=None,
                      out_dir=".",
                      verbose=True,
                      show_progress=False,
                      progress_csv=None,
                      checkpoint_every=1):
    """Run R replicates on a single (L, BQM) pair. Returns aggregated dict."""
    if sampler_mode is None:
        sampler_mode = SAMPLER_MODE
    if (sampler_mode not in ALLOWED_SAMPLER_MODES
            and sampler_mode not in _LEGACY_SAMPLER_MODES):
        raise ValueError(
            f"sampler_mode={sampler_mode!r} not in {ALLOWED_SAMPLER_MODES}")

    tL0 = time.time()
    bqm = make_random_bqm(L, density=density, seed=qubo_seed)

    # Chain-strength resolution
    if isinstance(chain_strength_mode, (int, float)):
        cs_value = float(chain_strength_mode)
    elif chain_strength_mode == "utc":
        cs_value = float(uniform_torque_compensation(bqm))
    else:
        raise ValueError("chain_strength_mode must be float or 'utc'")

    per_rep = []
    cbf_vecs = []      # read-level expanded CBF per replicate
    energy_vecs = []   # read-level expanded energy per replicate
    lengths_per_rep = []  # chain-length vector per replicate

    if sampler_mode == "clique_sampler":
        it = tqdm(range(n_reps), disable=not show_progress, desc=f"reps L={L}")
        for r in it:
            ss, cbf, e_stats, dt, timing, lengths = run_qa_via_clique_sampler(
                bqm, num_reads=num_reads, chain_strength=cs_value,
                anneal_time_us=anneal_time_us, anneal_schedule=anneal_schedule,
                verbose=(verbose and r == 0),
            )
            cbf_vecs.append(cbf["vec"])
            energy_vecs.append(e_stats["vec"])
            lengths_per_rep.append(lengths)
            mcl = float(np.mean(lengths)) if lengths.size else None
            xcl = int(np.max(lengths)) if lengths.size else None

            row = _per_rep_row(L, r, sampler_mode, qubo_seed, emb_seed,
                               cs_value, cbf, e_stats, dt, cbf["n"],
                               anneal_time_us, timing, mcl, xcl)
            per_rep.append(row)

            if progress_csv and checkpoint_every and ((r + 1) % checkpoint_every == 0):
                row_with_elapsed = dict(row, elapsed_sec=time.time() - tL0)
                _append_progress_row(progress_csv, row_with_elapsed)

            try:
                it.set_postfix(meanCBF=f"{cbf['mean']:.3f}",
                               pBreak=f"{cbf['prob_break']:.3f}",
                               bestE=f"{e_stats['best']:.3f}")
            except Exception:
                pass

    elif sampler_mode in ("fixed_embedding", "bqm_embedding"):
        qpu = get_qpu()
        if sampler_mode == "fixed_embedding":
            emb = find_clique_embedding(qpu, L, seed=emb_seed, verbose=verbose)
        else:
            emb = find_bqm_embedding(qpu, bqm, seed=emb_seed, verbose=verbose)
        shared_lengths = chain_lengths_from_embedding(emb)

        it = tqdm(range(n_reps), disable=not show_progress, desc=f"reps L={L}")
        for r in it:
            ss, cbf, e_stats, dt, timing = run_qa_with_embedding(
                bqm, emb, num_reads=num_reads, chain_strength=cs_value,
                anneal_time_us=anneal_time_us, anneal_schedule=anneal_schedule,
                verbose=(verbose and r == 0),
            )
            cbf_vecs.append(cbf["vec"])
            energy_vecs.append(e_stats["vec"])
            lengths_per_rep.append(shared_lengths)
            mcl = float(np.mean(shared_lengths))
            xcl = int(np.max(shared_lengths))

            row = _per_rep_row(L, r, sampler_mode, qubo_seed, emb_seed,
                               cs_value, cbf, e_stats, dt, cbf["n"],
                               anneal_time_us, timing, mcl, xcl)
            per_rep.append(row)

            if progress_csv and checkpoint_every and ((r + 1) % checkpoint_every == 0):
                row_with_elapsed = dict(row, elapsed_sec=time.time() - tL0)
                _append_progress_row(progress_csv, row_with_elapsed)

            try:
                it.set_postfix(meanCBF=f"{cbf['mean']:.3f}",
                               pBreak=f"{cbf['prob_break']:.3f}",
                               bestE=f"{e_stats['best']:.3f}")
            except Exception:
                pass

    # Determine canonical chain-length vector (TASK B).
    canonical_lengths, lengths_consistent = _resolve_canonical_lengths(
        lengths_per_rep)

    # Pooled stats over all replicates (occurrence-expanded).
    cbf_all = np.concatenate(cbf_vecs) if cbf_vecs else np.array([], dtype=float)
    e_all = np.concatenate(energy_vecs) if energy_vecs else np.array([], dtype=float)

    pooled_cbf = dict(
        mean=float(np.mean(cbf_all)) if cbf_all.size else 0.0,
        std=float(np.std(cbf_all, ddof=1)) if cbf_all.size > 1 else 0.0,
        prob_break=float(np.mean(cbf_all > 0.0)) if cbf_all.size else 0.0,
        n=int(cbf_all.size),
    )
    pooled_e = dict(
        mean=float(np.mean(e_all)) if e_all.size else 0.0,
        std=float(np.std(e_all, ddof=1)) if e_all.size > 1 else 0.0,
        # Best across replicates: min over per-replicate bests
        best=float(min((r["best_energy"] for r in per_rep), default=0.0)),
    )

    if verbose:
        atxt = f" | AT={anneal_time_us}us" if anneal_time_us is not None else ""
        mcl = (float(np.mean(canonical_lengths))
               if canonical_lengths.size else float("nan"))
        xcl = (int(np.max(canonical_lengths))
               if canonical_lengths.size else -1)
        log(f"[L={L}] pooled_meanCBF={pooled_cbf['mean']:.4f}, "
            f"pooled_pBreak={pooled_cbf['prob_break']:.3f}, "
            f"meanE={pooled_e['mean']:.4f}, bestE={pooled_e['best']:.4f} "
            f"| meanCL={mcl:.2f}, maxCL={xcl}{atxt} "
            f"| elapsed={time.time() - tL0:.2f}s "
            f"| sampler_mode={sampler_mode}")

    return dict(
        L=L, qubo_seed=qubo_seed, emb_seed=emb_seed,
        sampler_mode=sampler_mode,
        chain_strength=cs_value,
        per_rep=per_rep,
        cbf_vecs=cbf_vecs,
        energy_vecs=energy_vecs,
        lengths=canonical_lengths,
        lengths_per_rep=lengths_per_rep,
        lengths_consistent=lengths_consistent,
        pooled_cbf=pooled_cbf,
        pooled_energy=pooled_e,
        anneal_time_us=anneal_time_us,
    )


def _resolve_canonical_lengths(lengths_per_rep):
    """Pick a canonical chain-length vector across replicates and report
    whether all replicates agreed (sorted-vector equality)."""
    if not lengths_per_rep:
        return np.array([], dtype=int), True
    first_sorted = np.sort(lengths_per_rep[0])
    consistent = all(
        len(L) == len(first_sorted) and np.array_equal(np.sort(L), first_sorted)
        for L in lengths_per_rep
    )
    if not consistent:
        log("[chain-len] WARNING: chain lengths differ across replicates "
            "(common with DWaveCliqueSampler re-embedding). "
            "Saving rep-0 as canonical and full per-rep set in NPZ.")
    return np.asarray(lengths_per_rep[0], dtype=int), consistent


# -----------------------------------------------------------------------------
# 5. Experiment driver — sweep over L, fit ICE, save artifacts
# -----------------------------------------------------------------------------
def _ensure_sampler_mode_in_dirname(out_dir, sampler_mode):
    base = os.path.basename(os.path.normpath(out_dir))
    if sampler_mode in base:
        return out_dir
    return f"{out_dir.rstrip('/')}_{sampler_mode}"


def _atomic_write_csv(path, df):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _atomic_savez(path, npz_dict):
    # np.savez(file, ...) auto-appends ".npz" if missing, so the tmp path
    # must already end in .npz to land where we expect.
    base, _ext = os.path.splitext(path)
    tmp = base + ".tmp.npz"
    np.savez(tmp, **npz_dict)
    os.replace(tmp, path)


def run_experiment_and_fit(L_list,
                           n_reps=10, num_reads=2000,
                           qubo_seed_base=0, emb_seed_base=0,
                           density=1.0, chain_strength_mode="utc",
                           out_dir="qa_ice_fit_hybrid_out",
                           sh_grid=None, sc_grid=None, k_grid=None,
                           make_plots=True, verbose=True,
                           anneal_time_us=None, anneal_schedule=None,
                           sampler_mode=None,
                           show_progress=True,
                           checkpoint_every=1,
                           label=None,
                           exact_out_dir=False,
                           progress_csv_override=None,
                           resume=False):
    if sampler_mode is None:
        sampler_mode = SAMPLER_MODE

    if not exact_out_dir:
        out_dir = _ensure_sampler_mode_in_dirname(out_dir, sampler_mode)
    os.makedirs(out_dir, exist_ok=True)

    tag = label if label is not None else str(int(time.time()))

    perL_rows, raw_npz = [], {}

    progress_csv = (progress_csv_override if progress_csv_override is not None
                    else os.path.join(out_dir, f"per_replicate_{tag}.csv"))
    perL_live_csv = os.path.join(out_dir, f"summary_per_L_live_{tag}.csv")
    # Compute canonical output paths up front so we can write an atomic
    # per-L checkpoint after every L (Task 1 — protects against subprocess
    # SIGKILL mid-sweep).
    perL_csv = os.path.join(out_dir, f"summary_per_L_{tag}.csv")
    raw_npz_path = os.path.join(out_dir, f"raw_vectors_{tag}.npz")

    # Resume support: preload completed L data so we don't re-run them.
    completed_L_set: set[int] = set()
    if resume:
        if os.path.isfile(perL_csv):
            try:
                df_prev = pd.read_csv(perL_csv)
                if {"L", "mean_cbf_obs"}.issubset(df_prev.columns):
                    for _, r in df_prev.iterrows():
                        v = r.get("mean_cbf_obs")
                        if pd.notna(v):
                            perL_rows.append(r.to_dict())
                            completed_L_set.add(int(r["L"]))
            except Exception as e:
                log(f"[resume] failed to read {perL_csv}: "
                    f"{e.__class__.__name__}")
        if os.path.isfile(raw_npz_path):
            try:
                with np.load(raw_npz_path, allow_pickle=False) as z:
                    for k in z.files:
                        raw_npz[k] = z[k]
            except Exception as e:
                log(f"[resume] failed to read {raw_npz_path}: "
                    f"{e.__class__.__name__}")
        if completed_L_set:
            log(f"[resume] preloaded {len(completed_L_set)} completed L: "
                f"{sorted(completed_L_set)}")

    iterable = tqdm(list(enumerate(L_list, 1)), total=len(L_list),
                    disable=not show_progress, desc="L sweep")

    for idx, L in iterable:
        if int(L) in completed_L_set:
            log(f"[resume] L={L} already complete (from prior checkpoint); "
                "skipping")
            continue
        log(f"\n=== [{idx}/{len(L_list)}] L={L} (sampler_mode={sampler_mode}) ===")
        try:
            res = measure_L_oneQUBO(
                L, qubo_seed=qubo_seed_base + L, emb_seed=emb_seed_base + L,
                n_reps=n_reps, num_reads=num_reads, density=density,
                chain_strength_mode=chain_strength_mode,
                out_dir=out_dir, verbose=verbose,
                anneal_time_us=anneal_time_us, anneal_schedule=anneal_schedule,
                sampler_mode=sampler_mode,
                show_progress=show_progress,
                progress_csv=progress_csv,
                checkpoint_every=checkpoint_every,
            )
        except Exception as e:
            log(f"[skip] L={L} failed: {e.__class__.__name__}: {e}")
            continue

        # Stash raw vectors. CBF and energy are read-level expanded (TASK C).
        for r, vec in enumerate(res["cbf_vecs"]):
            raw_npz[f"CBF_vec_L{L}_rep{r}"] = vec
        raw_npz[f"CBF_vec_L{L}_ALL"] = (
            np.concatenate(res["cbf_vecs"]) if res["cbf_vecs"]
            else np.array([], dtype=float))
        for r, vec in enumerate(res["energy_vecs"]):
            raw_npz[f"E_vec_L{L}_rep{r}"] = vec
        raw_npz[f"E_vec_L{L}_ALL"] = (
            np.concatenate(res["energy_vecs"]) if res["energy_vecs"]
            else np.array([], dtype=float))

        # TASK B: chain length vector(s) saved per L.
        raw_npz[f"chain_lengths_L{L}"] = res["lengths"]
        if not res["lengths_consistent"]:
            for r, vec in enumerate(res["lengths_per_rep"]):
                raw_npz[f"chain_lengths_L{L}_rep{r}"] = np.asarray(vec, dtype=int)

        # Per-L summary row (with timing aggregates — TASK D).
        rep_df = pd.DataFrame(res["per_rep"])
        mcl = (float(np.mean(res["lengths"])) if res["lengths"].size
               else float("nan"))
        xcl = (int(np.max(res["lengths"])) if res["lengths"].size else -1)
        row = dict(
            L=L,
            sampler_mode=sampler_mode,
            qubo_seed=res["qubo_seed"], emb_seed=res["emb_seed"],
            mean_cbf_obs=res["pooled_cbf"]["mean"],
            std_cbf_obs=res["pooled_cbf"]["std"],
            prob_break_obs=res["pooled_cbf"]["prob_break"],
            n_total=res["pooled_cbf"]["n"],
            chain_strength=res["chain_strength"],
            mean_chainlen=mcl, max_chainlen=xcl,
            chain_lengths_consistent=bool(res["lengths_consistent"]),
            mean_energy_obs=res["pooled_energy"]["mean"],
            best_energy_obs=res["pooled_energy"]["best"],
            anneal_time_us=(float(res["anneal_time_us"])
                            if res["anneal_time_us"] is not None else np.nan),
        )
        for col in ("qpu_sampling_time_us", "qpu_access_time_us",
                    "client_wall_time_sec",
                    "qpu_programming_time_us",
                    "qpu_anneal_time_per_sample_us",
                    "qpu_readout_time_per_sample_us",
                    "qpu_delay_time_per_sample_us",
                    "total_post_processing_time_us",
                    "post_processing_overhead_time_us"):
            if col in rep_df.columns:
                row[f"{col}_mean"] = float(np.nanmean(rep_df[col]))
                row[f"{col}_std"] = (float(np.nanstd(rep_df[col], ddof=1))
                                     if len(rep_df) > 1 else 0.0)
            else:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_std"] = np.nan
        perL_rows.append(row)

        df_live = pd.DataFrame(perL_rows).sort_values("L").reset_index(drop=True)
        # Atomic per-L checkpoint — write canonical CSV + master NPZ now so
        # a SIGKILL between L values keeps the partial data on disk.
        _atomic_write_csv(perL_csv, df_live)
        _atomic_write_csv(perL_live_csv, df_live)
        _atomic_savez(raw_npz_path, raw_npz)

        if make_plots:
            pdf = os.path.join(out_dir, f"cbf_hist_L{L}_R{n_reps}.pdf")
            plot_histogram(
                raw_npz[f"CBF_vec_L{L}_ALL"],
                f"CBF Histogram (L={L}, R={n_reps}, mode={sampler_mode})",
                out_path=pdf,
            )

    # Guard: if every L raised at the QPU level (e.g. rate limit), perL_rows
    # is empty. Build an all-NaN placeholder DataFrame with the expected
    # columns so downstream phases can detect the failure (instead of
    # crashing on `sort_values("L")` against an empty frame).
    if not perL_rows:
        log("[fail] all L values failed at QPU level; producing all-NaN "
            "summary so downstream phases can detect the failure")
        for L_nan in L_list:
            row_nan = dict(
                L=int(L_nan),
                sampler_mode=sampler_mode,
                qubo_seed=qubo_seed_base + L_nan,
                emb_seed=emb_seed_base + L_nan,
                mean_cbf_obs=np.nan,
                std_cbf_obs=np.nan,
                prob_break_obs=np.nan,
                n_total=0,
                chain_strength=np.nan,
                mean_chainlen=np.nan,
                max_chainlen=-1,
                chain_lengths_consistent=False,
                mean_energy_obs=np.nan,
                best_energy_obs=np.nan,
                anneal_time_us=(float(anneal_time_us)
                                if anneal_time_us is not None else np.nan),
            )
            for col in ("qpu_sampling_time_us", "qpu_access_time_us",
                        "client_wall_time_sec",
                        "qpu_programming_time_us",
                        "qpu_anneal_time_per_sample_us",
                        "qpu_readout_time_per_sample_us",
                        "qpu_delay_time_per_sample_us",
                        "total_post_processing_time_us",
                        "post_processing_overhead_time_us"):
                row_nan[f"{col}_mean"] = np.nan
                row_nan[f"{col}_std"] = np.nan
            perL_rows.append(row_nan)

    df_L = pd.DataFrame(perL_rows).sort_values("L").reset_index(drop=True)
    # Final write (idempotent — same content as the latest per-L checkpoint
    # unless the empty-rows guard above appended NaN placeholders).
    _atomic_write_csv(perL_csv, df_L)
    log(f"[saved] {perL_csv}")

    # Fit ICE
    calib = [
        {"L": int(r.L), "cbf_obs": float(r.mean_cbf_obs),
         "lengths": raw_npz[f"chain_lengths_L{int(r.L)}"]}
        for _, r in df_L.iterrows()
        if f"chain_lengths_L{int(r.L)}" in raw_npz
    ]
    fit = fit_sigma_params(calib, sh_grid, sc_grid, k_grid, verbose=True) \
        if calib else dict(sigma_h=np.nan, sigma_c=np.nan, kappa=np.nan,
                           sse=np.nan, table=[])

    df_fit = pd.DataFrame(fit["table"]).sort_values("L").reset_index(drop=True) \
        if fit["table"] else pd.DataFrame(columns=["L", "cbf_obs", "cbf_pred"])
    if not df_fit.empty:
        df_fit["cbf_err"] = (df_fit["cbf_pred"] - df_fit["cbf_obs"]).abs()
    fit_csv = os.path.join(out_dir, f"fit_obs_vs_pred_{tag}.csv")
    df_fit.to_csv(fit_csv, index=False)
    log(f"[saved] {fit_csv}")
    log(f"[fit] sigma_h={fit['sigma_h']}  sigma_c={fit['sigma_c']}  "
        f"kappa={fit['kappa']}  SSE={fit['sse']}")

    overlay_pdf = os.path.join(out_dir, f"cbf_obs_pred_vs_L_{tag}.pdf")
    if make_plots and not df_L.empty and not df_fit.empty:
        plot_obs_vs_pred(df_L, df_fit, overlay_pdf, sampler_mode)

    # Final NPZ write (idempotent w.r.t. per-L checkpoints).
    _atomic_savez(raw_npz_path, raw_npz)
    log(f"[saved] {raw_npz_path}")

    # Fit-params JSON (Phase 4 will extend with full provenance metadata).
    fit_json_path = os.path.join(out_dir, f"fit_params_{tag}.json")
    with open(fit_json_path, "w") as f:
        json.dump({
            "sigma_h": fit["sigma_h"],
            "sigma_c": fit["sigma_c"],
            "kappa":   fit["kappa"],
            "sse":     fit["sse"],
            "sampler_mode": sampler_mode,
            # TODO(Phase 4): add solver, embedding seeds, qubo seeds,
            # anneal_time, chain_strength_mode, num_reads, n_reps, git SHA, etc.
        }, f, indent=2, default=_json_default)

    return dict(df_L=df_L, df_fit=df_fit, fit=fit,
                perL_csv=perL_csv, fit_csv=fit_csv,
                overlay_pdf=overlay_pdf, raw_npz=raw_npz_path,
                out_dir=out_dir, sampler_mode=sampler_mode)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# -----------------------------------------------------------------------------
# 6. Chain-strength sweep helper
# -----------------------------------------------------------------------------
def run_and_fit_chain_strength_sweep_flat(
    L_list, cs_values, *,
    n_reps=10, num_reads=2000,
    qubo_seed_base=0, emb_seed_base=0,
    density=1.0,
    anneal_time_us=20.0,
    out_root="qa_ice_fit_hybrid_CSsweep_AT20us_flat",
    sh_grid=None, sc_grid=None, k_grid=None,
    make_plots=True, verbose=True,
    sampler_mode=None,
    show_progress=True, checkpoint_every=1,
    label=None,
    exact_out_dir=False,
):
    if sampler_mode is None:
        sampler_mode = SAMPLER_MODE
    if not exact_out_dir:
        out_root = _ensure_sampler_mode_in_dirname(out_root, sampler_mode)
    os.makedirs(out_root, exist_ok=True)

    sweep_tag = label if label is not None else "sweep"
    sweep_progress_csv = os.path.join(out_root, f"per_replicate_{sweep_tag}.csv")

    combined_rows = []
    sub_npz_files = []
    for cs in cs_values:
        cs_tag = f"cs{str(round(cs, 2)).replace('.', 'p')}"
        sub_label = f"{sweep_tag}_{cs_tag}"
        log(f"=== [SWEEP] chain_strength={cs:.2f}, "
            f"anneal_time={anneal_time_us}us, sampler_mode={sampler_mode} ===")

        res = run_experiment_and_fit(
            L_list=L_list,
            n_reps=n_reps, num_reads=num_reads,
            qubo_seed_base=qubo_seed_base, emb_seed_base=emb_seed_base,
            density=density, chain_strength_mode=cs,
            out_dir=out_root,
            sh_grid=sh_grid, sc_grid=sc_grid, k_grid=k_grid,
            make_plots=make_plots, verbose=verbose,
            anneal_time_us=anneal_time_us, anneal_schedule=None,
            sampler_mode=sampler_mode,
            show_progress=show_progress,
            checkpoint_every=checkpoint_every,
            label=sub_label,
            exact_out_dir=True,
            progress_csv_override=sweep_progress_csv,
        )

        df_one = pd.read_csv(res["perL_csv"])
        df_one["chain_strength"] = cs
        df_one["anneal_time_us"] = anneal_time_us
        df_one["sampler_mode"] = sampler_mode
        combined_rows.append(df_one)
        sub_npz_files.append((cs_tag, res["raw_npz"]))

    df_all = pd.concat(combined_rows, ignore_index=True) \
        if combined_rows else pd.DataFrame()
    combined_csv = os.path.join(out_root, "sweep_summary_per_L.csv")
    df_all.to_csv(combined_csv, index=False)
    log(f"[saved] {combined_csv}")

    # Also write a labeled alias matching the manuscript file naming.
    alias_csv = os.path.join(out_root, f"summary_per_L_{sweep_tag}.csv")
    df_all.to_csv(alias_csv, index=False)
    log(f"[saved] {alias_csv}")

    # Merge per-cs raw NPZs into a single labeled NPZ with cs-prefixed keys.
    if sub_npz_files:
        merged = {}
        for cs_tag, path in sub_npz_files:
            try:
                with np.load(path) as data:
                    for k in data.files:
                        merged[f"{cs_tag}__{k}"] = data[k]
            except Exception as e:
                log(f"[merge-npz] WARNING: failed to merge {path}: "
                    f"{e.__class__.__name__}")
        merged_path = os.path.join(out_root, f"raw_vectors_{sweep_tag}.npz")
        np.savez(merged_path, **merged)
        log(f"[saved] {merged_path}")

    return df_all


# -----------------------------------------------------------------------------
# 7. Plotting (PDFs, headless)
# -----------------------------------------------------------------------------
def plot_histogram(vec, title, out_path, bins=25):
    if vec is None or len(vec) == 0:
        return
    fig = plt.figure(figsize=(6, 4))
    plt.hist(vec, bins=bins, alpha=0.7, density=True)
    plt.title(title)
    plt.xlabel("chain_break_fraction")
    plt.ylabel("density")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    log(f"[saved] {out_path}")


def plot_obs_vs_pred(df_L, df_fit, out_path, sampler_mode):
    fig = plt.figure(figsize=(6, 4))
    plt.plot(df_L["L"], df_L["mean_cbf_obs"], marker="o", label="Observed")
    plt.plot(df_fit["L"], df_fit["cbf_pred"], marker="o", label="Predicted")
    plt.xlabel("L (logical variables / chains)")
    plt.ylabel("mean CBF")
    plt.title(f"Observed vs Predicted mean CBF (mode={sampler_mode})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    log(f"[saved] {out_path}")


# -----------------------------------------------------------------------------
# 8. Phase-1 sanity checks
# -----------------------------------------------------------------------------
def _make_fake_sampleset(num_occ, cbf, energies=None, timing=None):
    """Build a minimal SampleSet-like object for unit tests."""
    n = len(num_occ)
    if energies is None:
        energies = np.linspace(-1.0, -0.5, n)
    dtype = [
        ("chain_break_fraction", "f8"),
        ("num_occurrences", "i8"),
        ("energy", "f8"),
    ]
    rec = np.zeros(n, dtype=dtype)
    rec["chain_break_fraction"] = cbf
    rec["num_occurrences"] = num_occ
    rec["energy"] = energies
    ss = SimpleNamespace()
    ss.record = rec
    ss.info = {"timing": timing} if timing is not None else {}
    return ss


# Files this revision is responsible for. Pre-existing notebooks like
# Experiment*.ipynb / Distribution*.ipynb are out of scope (they predate
# Phase 1) but still get scanned and reported as a separate WARNING — see
# _sanity_check_token_scan.
IN_SCOPE_FILES = (
    "RQ1.ipynb",
    "RQ234.ipynb",
    "RQ5.ipynb",
    "qa_utils.py",
    "rq234_revised.py",
    # Phase 2: analysis_validation_and_kstar.py
    # Phase 3: rq5_time_matched.py
)


def _scan_for_hardcoded_tokens(paths):
    """Return a list of (file, snippet) for files containing a TOKEN
    assignment to a long literal string."""
    import re
    pattern = re.compile(r"""TOKEN\s*=\s*['"]([A-Za-z0-9_\-]{16,})['"]""")
    findings = []
    for p in paths:
        if not os.path.isfile(p):
            continue
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()
        except Exception:
            continue
        for m in pattern.finditer(txt):
            # Truncate to avoid logging the full token even in failure msgs.
            snippet = m.group(0)[:24] + "…(redacted)…"
            findings.append((p, snippet))
    return findings


def _sanity_check_token_scan(repo_root):
    """Check 1: in-scope files assign TOKEN only via env-var read.

    Strict pass requires no hard-coded token in the files this revision
    owns. Any leaks in legacy notebooks are reported as a separate
    WARNING block but do not fail the check.
    """
    in_scope_paths = [os.path.join(repo_root, fn) for fn in IN_SCOPE_FILES]
    in_scope_bad = _scan_for_hardcoded_tokens(in_scope_paths)

    # Walk the rest of the repo for an informational warning.
    other_paths = []
    for root, _, files in os.walk(repo_root):
        if any(s in root for s in (".git", "__pycache__",
                                   ".ipynb_checkpoints", ".venv")):
            continue
        for fn in files:
            if not fn.endswith((".py", ".ipynb")):
                continue
            full = os.path.join(root, fn)
            if os.path.basename(full) in IN_SCOPE_FILES:
                continue
            other_paths.append(full)
    legacy_bad = _scan_for_hardcoded_tokens(other_paths)
    return in_scope_bad, legacy_bad


def _sanity_check_summarize_cbf():
    """Check 2: weighted mean of CBF with num_occurrences=[3,1,2],
    cbf=[0.0,0.5,1.0] equals 0.4167."""
    ss = _make_fake_sampleset([3, 1, 2], [0.0, 0.5, 1.0])
    stats = summarize_cbf(ss)
    expected = (0.0 * 3 + 0.5 * 1 + 1.0 * 2) / 6.0
    assert abs(stats["mean"] - expected) < 1e-4, (
        f"weighted mean mismatch: got {stats['mean']!r}, expected {expected!r}")
    assert stats["n"] == 6, f"expected 6 expanded reads, got {stats['n']}"
    assert stats["vec"].size == 6
    return expected, stats["mean"]


def _sanity_check_extract_qpu_timing_missing():
    """Check 3: extract_qpu_timing returns NaN dict when timing is missing
    or non-numeric. Must not raise."""
    ss_empty = SimpleNamespace(info={})
    t1 = extract_qpu_timing(ss_empty)
    assert all(np.isnan(v) for v in t1.values()), (
        f"expected all NaN, got {t1}")

    ss_missing_block = SimpleNamespace(info={"timing": None})
    t2 = extract_qpu_timing(ss_missing_block)
    assert all(np.isnan(v) for v in t2.values())

    ss_bad = SimpleNamespace(info={"timing": {"qpu_sampling_time": "n/a"}})
    t3 = extract_qpu_timing(ss_bad)
    assert np.isnan(t3["qpu_sampling_time_us"])
    return t1, t2, t3


def _sanity_check_sampler_mode_threading():
    """Check 4: SAMPLER_MODE env var is read and threaded through to
    output paths and per-replicate CSV columns."""
    # Env var is read at module top
    import rq234_revised as mod
    assert hasattr(mod, "SAMPLER_MODE")

    # CSV row template includes sampler_mode
    fake_cbf = dict(mean=0.1, std=0.0, prob_break=0.05, n=10,
                    vec=np.zeros(10))
    fake_e = dict(mean=-1.0, std=0.1, best=-2.0, n=10, vec=-np.ones(10))
    fake_timing = {k: 0.0 for k in TIMING_FIELDS}
    row = _per_rep_row(L=8, replicate=0, sampler_mode="fixed_embedding",
                       qubo_seed=8, emb_seed=8, chain_strength=1.0,
                       cbf=fake_cbf, e_stats=fake_e, dt_wall=0.5, n_reads=10,
                       anneal_time_us=20.0, timing=fake_timing,
                       mean_chainlen=2.0, max_chainlen=3)
    assert "sampler_mode" in row
    assert row["sampler_mode"] == "fixed_embedding"
    for col in ("client_wall_time_sec", "qpu_sampling_time_us",
                "qpu_access_time_us", "qpu_programming_time_us",
                "qpu_anneal_time_per_sample_us",
                "qpu_readout_time_per_sample_us",
                "qpu_delay_time_per_sample_us"):
        assert col in row, f"missing CSV col: {col}"

    # Output directory name carries sampler_mode
    p = _ensure_sampler_mode_in_dirname("qa_ice_fit_hybrid_out",
                                        "fixed_embedding")
    assert "fixed_embedding" in p
    p2 = _ensure_sampler_mode_in_dirname("Test_clique_sampler",
                                         "clique_sampler")
    assert p2 == "Test_clique_sampler"  # already present, no double-suffix
    return row, p, p2


def _sanity_check_chain_lengths_in_npz():
    """Check 5: chain_lengths_L{L} is saved in the NPZ for at least one L.

    We exercise this without a QPU by writing a small NPZ with the same key
    convention used by run_experiment_and_fit, then re-reading it. Also
    statically verifies the line is present in the source.
    """
    import tempfile

    src = os.path.abspath(__file__)
    with open(src, "r") as f:
        source = f.read()
    assert 'raw_npz[f"chain_lengths_L{L}"] = res["lengths"]' in source, (
        "chain_lengths_L{L} write site missing from run_experiment_and_fit")

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "raw_vectors_test.npz")
        L = 8
        chain_lengths = np.array([1, 2, 2, 3], dtype=int)
        np.savez(path, **{f"chain_lengths_L{L}": chain_lengths,
                          f"CBF_vec_L{L}_ALL": np.zeros(10)})
        with np.load(path) as data:
            assert f"chain_lengths_L{L}" in data.files
            assert np.array_equal(data[f"chain_lengths_L{L}"], chain_lengths)
    return True


def run_phase1_sanity_checks(repo_root=None):
    if repo_root is None:
        repo_root = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("Phase-1 sanity checks")
    print("=" * 60)

    # 1
    in_scope_bad, legacy_bad = _sanity_check_token_scan(repo_root)
    if legacy_bad:
        print("[WARN] check 1: legacy notebooks (out of Phase-1 scope) "
              "contain hard-coded TOKEN strings. ROTATE THE D-WAVE TOKEN "
              "and clean these files manually:")
        for path, snippet in legacy_bad:
            print(f"    {path}: {snippet}")
    if in_scope_bad:
        print("[FAIL] check 1: in-scope files contain hard-coded TOKEN:")
        for path, snippet in in_scope_bad:
            print(f"    {path}: {snippet}")
        raise AssertionError("hard-coded token strings detected in in-scope files")
    print(f"[ OK ] check 1: in-scope files ({', '.join(IN_SCOPE_FILES)}) "
          "use env-var reads only.")

    # 2
    expected, got = _sanity_check_summarize_cbf()
    print(f"[ OK ] check 2: summarize_cbf weighted mean = {got:.6f} "
          f"(expected {expected:.6f}).")

    # 3
    _sanity_check_extract_qpu_timing_missing()
    print("[ OK ] check 3: extract_qpu_timing returns NaN (no raise) "
          "when timing is missing/malformed.")

    # 4
    row, p1, p2 = _sanity_check_sampler_mode_threading()
    print(f"[ OK ] check 4: SAMPLER_MODE threaded — row['sampler_mode']="
          f"{row['sampler_mode']!r}, dirname1={p1!r}, dirname2={p2!r}.")

    # 5
    _sanity_check_chain_lengths_in_npz()
    print("[ OK ] check 5: chain_lengths_L{L} round-trips through NPZ; "
          "write site present in source.")

    print("=" * 60)
    print("All Phase-1 sanity checks passed.")
    print("=" * 60)


# -----------------------------------------------------------------------------
# 9. CLI entrypoint
# -----------------------------------------------------------------------------
def _build_cli():
    p = argparse.ArgumentParser(
        description="RQ2/RQ3/RQ4 QA driver (revised)."
    )
    p.add_argument("--sanity-check", action="store_true",
                   help="Run Phase-1 sanity checks and exit.")
    p.add_argument("--sampler-mode",
                   choices=list(ALLOWED_SAMPLER_MODES),
                   default=None,
                   help=f"Override SAMPLER_MODE env var "
                        f"(default: {SAMPLER_MODE!r}).")
    p.add_argument("--L-min", type=int, default=5)
    p.add_argument("--L-step", type=int, default=5)
    p.add_argument("--n-reps", type=int, default=10)
    p.add_argument("--num-reads", type=int, default=2000)
    p.add_argument("--anneal-time-us", type=float, default=20.0)
    p.add_argument("--cs-min", type=float, default=0.1)
    p.add_argument("--cs-max", type=float, default=2.5)
    p.add_argument("--cs-step", type=float, default=0.1)
    p.add_argument("--out-root", type=str, default=None,
                   help="Output directory root. Default includes sampler_mode.")
    return p


def main(argv=None):
    args = _build_cli().parse_args(argv)

    if args.sanity_check:
        run_phase1_sanity_checks()
        return 0

    sampler_mode = args.sampler_mode or SAMPLER_MODE
    out_root = args.out_root or f"qa_ice_fit_hybrid_CSsweep_AT{int(args.anneal_time_us)}us_flat"

    L_list, _ = get_L_list_from_clique(L_min=args.L_min, step=args.L_step,
                                       solver=SOLVER)
    cs_values = np.round(np.arange(args.cs_min,
                                   args.cs_max + args.cs_step / 2,
                                   args.cs_step), 2)

    run_and_fit_chain_strength_sweep_flat(
        L_list=L_list, cs_values=cs_values,
        n_reps=args.n_reps, num_reads=args.num_reads,
        density=1.0, anneal_time_us=args.anneal_time_us,
        out_root=out_root,
        sampler_mode=sampler_mode,
        show_progress=True, checkpoint_every=1,
        make_plots=True, verbose=True,
    )
    log("[done] chain-strength sweep finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
