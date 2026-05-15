"""analysis_kstar_correlated_followup.py — exploratory follow-up to
analysis_model_comparison.py.

Two tasks (no QPU; pure CPU on existing artifacts).

Task 1
    Run the existing independent-vs-correlated fit on the RQ2 CBF data
    across multiple optimizer seeds {1, 2, 3, 4, 5, 42, 100, 1000} to
    check whether the rho_corr=0 collapse is seed-stable.

Task 2
    Fit two power-law models to the k*(L) data at tau in {0.01, 0.02,
    0.05}, on log-log space:
        Independent reference (1 param): log(k*) = log(a) + 0.5 * log(L)
        Correlated extension (2 params): log(k*) = log(a) + alpha * log(L)
    The "correlated extension" lets alpha float; the "independent
    reference" pins alpha to 0.5. Compare in-sample fit (log-space SSE,
    linear-space SSE/MAE) and AIC/BIC.

Outputs
    out/<ts>/phase2/rq6_correlated_seed_robustness.csv     (Task 1)
    out/<ts>/phase2/rq6_kstar_model_comparison.csv         (Task 2)
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import erfc


_TOK = os.getenv("DWAVE_API_TOKEN", "")


def safe_print(msg):
    s = str(msg)
    if _TOK and _TOK in s:
        s = s.replace(_TOK, "***REDACTED***")
    print(s, flush=True)


# ----------------------------------------------------------------------------
# Task 1 — CBF seed robustness
# ----------------------------------------------------------------------------
def task1_cbf_seed_robustness(run_dir, seeds):
    from analysis_model_comparison import (
        load_rq2, fit_model, compute_metrics, aic_bic,
    )
    summary_csv = os.path.join(run_dir, "rq2_at5us_cs1p0",
                                "summary_per_L_RQ2.csv")
    raw_npz = os.path.join(run_dir, "rq2_at5us_cs1p0", "raw_vectors_RQ2.npz")
    L_full, cbf_full, chain_dict, _std = load_rq2(summary_csv, raw_npz)
    n = len(L_full)
    safe_print(f"[task1] CBF data: N={n} L values, L={L_full[0]}..{L_full[-1]}")

    rows = []
    for seed in seeds:
        safe_print(f"  fitting both models at seed={seed} ...")
        # Independent (3 params)
        p_i, _ = fit_model("independent", L_full, chain_dict, cbf_full,
                           n_restarts=8, seed=seed)
        sse_i = compute_metrics("independent", p_i, L_full,
                                chain_dict, cbf_full)["sse"]
        aic_i, bic_i = aic_bic(sse_i, n, k=3)

        # Correlated (5 params)
        p_c, _ = fit_model("correlated", L_full, chain_dict, cbf_full,
                           n_restarts=10, seed=seed)
        sse_c = compute_metrics("correlated", p_c, L_full,
                                chain_dict, cbf_full)["sse"]
        aic_c, bic_c = aic_bic(sse_c, n, k=5)

        rows.append({
            "seed": seed,
            "indep_sigma_h": p_i["sigma_h"],
            "indep_sigma_c": p_i["sigma_c"],
            "indep_kappa": p_i["kappa"],
            "indep_sse": sse_i,
            "indep_aic": aic_i,
            "indep_bic": bic_i,
            "corr_sigma_h": p_c["sigma_h"],
            "corr_sigma_c": p_c["sigma_c"],
            "corr_kappa": p_c["kappa"],
            "corr_rho_corr": p_c["rho_corr"],
            "corr_gamma": p_c["gamma"],
            "corr_sse": sse_c,
            "corr_aic": aic_c,
            "corr_bic": bic_c,
            "delta_aic": aic_c - aic_i,
            "delta_bic": bic_c - bic_i,
        })
    return pd.DataFrame(rows)


def print_task1_table(df):
    safe_print("\n" + "=" * 92)
    safe_print("Task 1 — CBF seed-robustness summary")
    safe_print("=" * 92)
    safe_print(
        f"  {'seed':>5}  {'sigma_h':>7}  {'sigma_c':>7}  {'kappa':>6}  "
        f"{'rho_corr':>8}  {'gamma':>6}  {'SSE_c':>10}  "
        f"{'dAIC':>7}  {'dBIC':>7}")
    for _, r in df.sort_values("seed").iterrows():
        safe_print(
            f"  {int(r['seed']):>5}  "
            f"{r['corr_sigma_h']:>7.4f}  {r['corr_sigma_c']:>7.4f}  "
            f"{r['corr_kappa']:>6.3f}  "
            f"{r['corr_rho_corr']:>8.4f}  {r['corr_gamma']:>6.3f}  "
            f"{r['corr_sse']:>10.3e}  "
            f"{r['delta_aic']:>+7.2f}  {r['delta_bic']:>+7.2f}")
    safe_print("-" * 92)
    safe_print(
        f"  rho_corr range across seeds: "
        f"[{df['corr_rho_corr'].min():.4f}, {df['corr_rho_corr'].max():.4f}] "
        f"   mean={df['corr_rho_corr'].mean():.4f}  "
        f"  max|rho_corr|={df['corr_rho_corr'].abs().max():.4f}")
    safe_print(
        f"  delta_AIC range: "
        f"[{df['delta_aic'].min():+.2f}, {df['delta_aic'].max():+.2f}]  "
        f"(positive => correlated penalized)")


# ----------------------------------------------------------------------------
# Task 2 — k* power-law model comparison
# ----------------------------------------------------------------------------
def fit_kstar_models(L_arr, k_arr):
    """Fit independent (alpha=1/2 fixed) and correlated (alpha free)
    power-law models on log-log space.

    Returns dict with both fits + their metrics.
    """
    x = np.log(L_arr.astype(float))
    y = np.log(k_arr.astype(float))
    n = int(y.size)

    # Independent reference: alpha = 0.5 fixed; only intercept (log a) free.
    # log(k*) = log(a) + 0.5 * log(L)   =>   log(a) = mean(y - 0.5 x)
    log_a_indep = float(np.mean(y - 0.5 * x))
    pred_log_indep = log_a_indep + 0.5 * x
    sse_log_indep = float(np.sum((y - pred_log_indep) ** 2))
    pred_lin_indep = np.exp(pred_log_indep)
    sse_lin_indep = float(np.sum((k_arr - pred_lin_indep) ** 2))
    mae_lin_indep = float(np.mean(np.abs(k_arr - pred_lin_indep)))
    aic_indep, bic_indep = _aic_bic(sse_log_indep, n, k=1)

    # Correlated extension: log(a) and alpha both free => ordinary
    # least-squares on (log L, log k*).
    coef = np.polyfit(x, y, 1)
    alpha_corr = float(coef[0])
    log_a_corr = float(coef[1])
    pred_log_corr = log_a_corr + alpha_corr * x
    sse_log_corr = float(np.sum((y - pred_log_corr) ** 2))
    pred_lin_corr = np.exp(pred_log_corr)
    sse_lin_corr = float(np.sum((k_arr - pred_lin_corr) ** 2))
    mae_lin_corr = float(np.mean(np.abs(k_arr - pred_lin_corr)))
    aic_corr, bic_corr = _aic_bic(sse_log_corr, n, k=2)

    return {
        "n": n,
        "independent": dict(
            a=float(np.exp(log_a_indep)),
            log_a=log_a_indep,
            alpha=0.5,  # fixed
            alpha_fixed=True,
            sse_log=sse_log_indep,
            sse_linear=sse_lin_indep,
            mae_linear=mae_lin_indep,
            aic=aic_indep,
            bic=bic_indep,
        ),
        "correlated": dict(
            a=float(np.exp(log_a_corr)),
            log_a=log_a_corr,
            alpha=alpha_corr,
            alpha_fixed=False,
            sse_log=sse_log_corr,
            sse_linear=sse_lin_corr,
            mae_linear=mae_lin_corr,
            aic=aic_corr,
            bic=bic_corr,
        ),
        "delta_aic": aic_corr - aic_indep,
        "delta_bic": bic_corr - bic_indep,
    }


def _aic_bic(sse, n, k):
    """AIC/BIC under Gaussian-noise assumption.

    Standard form: AIC = n ln(SSE/n) + 2k, BIC = n ln(SSE/n) + k ln(n).
    Computed on log-space SSE (the fit's natural loss), so the AIC values
    are absolute-comparable between the two models on the same threshold.
    """
    if n <= 0 or not np.isfinite(sse) or sse <= 0:
        return float("nan"), float("nan")
    aic = n * np.log(sse / n) + 2 * k
    bic = n * np.log(sse / n) + k * np.log(n)
    return float(aic), float(bic)


def task2_kstar_comparison(run_dir, taus=(0.01, 0.02, 0.05)):
    rows = []
    fits_by_tau = {}
    for tau in taus:
        tau_tag = f"{tau:.2f}".replace(".", "p")
        csv_path = os.path.join(run_dir, "phase2",
                                 f"kstar_points_tau{tau_tag}.csv")
        df = pd.read_csv(csv_path)
        # Keep only finite, positive points (log requires positive).
        df = df[(df["L"] > 0) & (df["kstar"] > 0)].copy()
        L_arr = df["L"].to_numpy(dtype=float)
        k_arr = df["kstar"].to_numpy(dtype=float)
        if len(L_arr) < 3:
            safe_print(f"[task2] tau={tau}: only {len(L_arr)} usable points; "
                       "skipping (need >=3 for a 2-param fit)")
            continue
        fit = fit_kstar_models(L_arr, k_arr)
        fits_by_tau[tau] = fit
        for model in ("independent", "correlated"):
            m = fit[model]
            rows.append({
                "tau": tau,
                "model": model,
                "n_points": fit["n"],
                "a": m["a"],
                "alpha": m["alpha"],
                "alpha_fixed": m["alpha_fixed"],
                "sse_log": m["sse_log"],
                "sse_linear": m["sse_linear"],
                "mae_linear": m["mae_linear"],
                "aic": m["aic"],
                "bic": m["bic"],
                "delta_aic_corr_minus_indep": fit["delta_aic"],
                "delta_bic_corr_minus_indep": fit["delta_bic"],
            })
    return pd.DataFrame(rows), fits_by_tau


def print_task2_table(df, fits_by_tau):
    safe_print("\n" + "=" * 100)
    safe_print("Task 2 — k* power-law model comparison "
               "(independent alpha=0.5 vs correlated alpha free)")
    safe_print("=" * 100)
    safe_print(
        f"  {'tau':>5}  {'model':>11}  {'N':>2}  "
        f"{'a':>7}  {'alpha':>6}  "
        f"{'SSE_log':>10}  {'SSE_lin':>10}  {'MAE_lin':>8}  "
        f"{'AIC':>8}  {'BIC':>8}")
    for tau, fit in fits_by_tau.items():
        for model in ("independent", "correlated"):
            m = fit[model]
            alpha_str = f"{m['alpha']:.3f}{'*' if m['alpha_fixed'] else ' '}"
            safe_print(
                f"  {tau:>5.2f}  {model:>11s}  {fit['n']:>2d}  "
                f"{m['a']:>7.3f}  {alpha_str:>6s}  "
                f"{m['sse_log']:>10.4f}  {m['sse_linear']:>10.4f}  "
                f"{m['mae_linear']:>8.4f}  "
                f"{m['aic']:>8.2f}  {m['bic']:>8.2f}")
        winner_aic = ("correlated" if fit["delta_aic"] < 0 else "independent")
        winner_bic = ("correlated" if fit["delta_bic"] < 0 else "independent")
        safe_print(
            f"        {'':>11}      "
            f"delta_AIC={fit['delta_aic']:+.2f} (winner: {winner_aic})  "
            f"delta_BIC={fit['delta_bic']:+.2f} (winner: {winner_bic})")
        safe_print("-" * 100)
    safe_print("  * = alpha pinned to 0.5 in the independent reference model.")


# ----------------------------------------------------------------------------
# 3-line interpretation
# ----------------------------------------------------------------------------
def print_interpretation(task1_df, fits_by_tau):
    safe_print("\n" + "=" * 80)
    safe_print("Interpretation")
    safe_print("=" * 80)

    # (a) rho_corr robust?
    rho_max = float(task1_df["corr_rho_corr"].abs().max())
    daic_pos = (task1_df["delta_aic"] > 0).all()
    if rho_max < 1e-3 and daic_pos:
        line_a = (f"(a) rho_corr=0 is ROBUST on CBF data: max|rho_corr| = "
                  f"{rho_max:.2e} across all "
                  f"{len(task1_df)} seeds; delta_AIC always > 0 "
                  f"(correlated extension penalized).")
    elif rho_max < 1e-3:
        line_a = (f"(a) rho_corr=0 is ROBUST on CBF data: max|rho_corr| = "
                  f"{rho_max:.2e}, but at least one seed gives delta_AIC <= 0.")
    else:
        line_a = (f"(a) rho_corr=0 is NOT robust on CBF data: max|rho_corr| = "
                  f"{rho_max:.4f} across seeds (some seeds found nonzero rho).")

    # (b) correlated wins on k*?
    parts_b = []
    all_corr_win = True
    for tau, fit in fits_by_tau.items():
        a_corr = fit["correlated"]["alpha"]
        parts_b.append(f"tau={tau}: alpha_fit={a_corr:.3f}, "
                       f"dAIC={fit['delta_aic']:+.2f}")
        if fit["delta_aic"] >= 0:
            all_corr_win = False
    if all_corr_win:
        line_b = ("(b) Correlated extension JUSTIFIED on k* data: "
                  "delta_AIC < 0 at all thresholds. "
                  + "; ".join(parts_b))
    else:
        line_b = ("(b) Correlated extension NOT uniformly justified on k* "
                  "data: at least one threshold gives delta_AIC >= 0. "
                  + "; ".join(parts_b))

    # (c) overall narrative support
    if rho_max < 1e-3 and all_corr_win:
        line_c = ("(c) Paper narrative SUPPORTED: independent baseline is "
                  "sufficient for the fixed-k CBF marginal, AND the "
                  "correlated extension is uniquely justified on the k* "
                  "scaling diagnostic. The two diagnostics tell consistent, "
                  "complementary stories.")
    elif rho_max < 1e-3 and not all_corr_win:
        line_c = ("(c) Paper narrative MIXED: CBF rho_corr=0 is robust, but "
                  "the correlated power-law extension does not uniformly "
                  "beat alpha=0.5 on k* data. The k* deviation may require "
                  "a different parameterization than the one tested here.")
    elif rho_max >= 1e-3 and all_corr_win:
        line_c = ("(c) Paper narrative PARTIALLY supported: correlated wins "
                  "on k* but rho_corr is not strictly zero across seeds on "
                  "CBF data. Manuscript claim of rho_corr=0 may need a "
                  "softer wording (rho_corr <= eps under most seeds).")
    else:
        line_c = ("(c) Paper narrative NEEDS REVISION: rho_corr=0 not robust "
                  "AND correlated does not uniformly win on k* data.")

    safe_print(line_a)
    safe_print(line_b)
    safe_print(line_c)


# ----------------------------------------------------------------------------
# Task 3 — full noise-model fit on k*(L) with numerical CBF crossing
# ----------------------------------------------------------------------------
#
# Model: predicted CBF at logical size L, chain strength cs, with parameters
# (sigma_h, sigma_c, kappa[, rho_corr, gamma]) is
#
#     CBF(L, cs) = (1 / K_L) sum_i  erfc(kappa * cs / sqrt(2 * var_i))
#
# where the sum is over the K_L chains for logical size L and
#
#     var_i = ell_i * sigma_h^2 + max(ell_i - 1, 0) * sigma_c^2          (indep)
#           + rho_corr * max(ell_i, 1)^gamma                            (corr extra)
#
# kappa here is a coupling scale that maps chain strength cs to the
# argument of erfc (the noise margin in the manuscript's Theorem 2 form,
# parameterized so that per-cs CBF varies through cs * kappa).
#
# Predicted k*(L) = the cs at which CBF(L, cs) = tau, found via bisection.

KSTAR_LO = 1e-4
KSTAR_HI = 20.0


def _var_array(ell, sigma_h, sigma_c, rho_corr=0.0, gamma=1.0):
    ell = np.asarray(ell, dtype=float)
    v = ell * sigma_h ** 2 + np.maximum(ell - 1, 0) * sigma_c ** 2
    if rho_corr != 0.0:
        v = v + rho_corr * np.power(np.maximum(ell, 1.0), gamma)
    return np.maximum(v, 1e-12)


def predicted_cbf_at_cs(cs, ell_vec, sigma_h, sigma_c, kappa,
                        rho_corr=0.0, gamma=1.0):
    var = _var_array(ell_vec, sigma_h, sigma_c, rho_corr, gamma)
    return float(np.mean(erfc(kappa * cs / np.sqrt(2.0 * var))))


def predicted_kstar(L, ell_vec, sigma_h, sigma_c, kappa, tau,
                    rho_corr=0.0, gamma=1.0,
                    lo=KSTAR_LO, hi=KSTAR_HI):
    """Solve for cs s.t. CBF(L, cs; params) == tau. Returns NaN if the
    target tau is not bracketed (CBF stays above tau even at hi, or below
    tau already at lo)."""
    def f(cs):
        return predicted_cbf_at_cs(cs, ell_vec, sigma_h, sigma_c, kappa,
                                    rho_corr, gamma) - tau
    f_lo, f_hi = f(lo), f(hi)
    if not np.isfinite(f_lo) or not np.isfinite(f_hi):
        return float("nan")
    if f_lo < 0:
        # Already below tau at lo — predicted k* < lo. Return lo as a stub
        # (loss function will see (lo - obs)^2; sufficient gradient signal).
        return lo
    if f_hi > 0:
        # CBF still above tau at hi — predicted k* > hi. Return hi stub.
        return hi
    try:
        return float(brentq(f, lo, hi, xtol=1e-6, rtol=1e-6, maxiter=80))
    except Exception:
        return float("nan")


# Parameter packing
INDEP_BOUNDS = [(0.001, 0.15), (0.001, 0.15), (0.05, 5.0)]
CORR_BOUNDS = INDEP_BOUNDS + [(0.0, 1.0), (1.0, 3.0)]

INDEP_STARTS = [
    np.array([0.05, 0.05, 0.5]),
    np.array([0.02, 0.02, 1.0]),
    np.array([0.08, 0.08, 0.3]),
    np.array([0.03, 0.05, 0.6]),
]
CORR_STARTS = [
    np.array([0.05, 0.05, 0.5, 0.0,  1.0]),     # collapse-to-indep
    np.array([0.05, 0.01, 0.4, 0.05, 1.8]),
    np.array([0.03, 0.02, 0.3, 0.20, 1.6]),
    np.array([0.02, 0.01, 0.8, 0.10, 2.0]),
    np.array([0.04, 0.04, 0.6, 0.30, 1.4]),
    np.array([0.01, 0.01, 1.2, 0.40, 2.4]),
]


def _clip(x, bounds):
    return np.array([np.clip(v, lo, hi)
                     for v, (lo, hi) in zip(x, bounds)], dtype=float)


def _unpack(x, model):
    if model == "independent":
        return dict(sigma_h=float(x[0]), sigma_c=float(x[1]),
                    kappa=float(x[2]), rho_corr=0.0, gamma=1.0)
    return dict(sigma_h=float(x[0]), sigma_c=float(x[1]),
                kappa=float(x[2]),
                rho_corr=float(x[3]), gamma=float(x[4]))


def fit_kstar_full(L_obs, kstar_obs, chain_lengths_dict, tau, model,
                   seed=42, verbose=False):
    """Fit (sigma_h, sigma_c, kappa[, rho_corr, gamma]) by minimizing
    sum_i (predicted_kstar(L_i) - kstar_obs_i)^2 (linear-space SSE)."""
    bounds = INDEP_BOUNDS if model == "independent" else CORR_BOUNDS
    starts = list(INDEP_STARTS if model == "independent" else CORR_STARTS)
    rng = np.random.default_rng(seed)
    # Round out the starting grid to at least 8 restarts
    while len(starts) < 8:
        x = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
        starts.append(x)

    L_obs = list(L_obs)
    kstar_obs = np.asarray(kstar_obs, dtype=float)
    # Cache chain-length vectors as float for speed
    ell_cache = {int(L): np.asarray(chain_lengths_dict[int(L)], dtype=float)
                 for L in L_obs if int(L) in chain_lengths_dict}

    def loss(x):
        xc = _clip(x, bounds)
        p = _unpack(xc, model)
        sse = 0.0
        for L, k_obs in zip(L_obs, kstar_obs):
            ell = ell_cache.get(int(L))
            if ell is None:
                continue
            k_pred = predicted_kstar(L, ell, p["sigma_h"], p["sigma_c"],
                                      p["kappa"], tau,
                                      p["rho_corr"], p["gamma"])
            sse += (k_pred - k_obs) ** 2
        return sse + _bound_penalty(x, bounds)

    best_x, best_loss = None, float("inf")
    for x0 in starts:
        try:
            res = minimize(loss, x0, method="Nelder-Mead",
                           options=dict(maxiter=2000, xatol=1e-7, fatol=1e-10,
                                         adaptive=True))
        except Exception as e:
            if verbose:
                safe_print(f"    restart failed: {e}")
            continue
        xc = _clip(res.x, bounds)
        f_final = float(loss(xc))
        if f_final < best_loss:
            best_loss = f_final
            best_x = xc
    if best_x is None:
        raise RuntimeError(f"all restarts failed for {model}/tau={tau}")

    params = _unpack(best_x, model)
    # Pure linear-space SSE without the boundary penalty
    sse_lin = 0.0
    abs_err_sum = 0.0
    n = 0
    for L, k_obs in zip(L_obs, kstar_obs):
        ell = ell_cache.get(int(L))
        if ell is None:
            continue
        k_pred = predicted_kstar(L, ell, params["sigma_h"], params["sigma_c"],
                                  params["kappa"], tau,
                                  params["rho_corr"], params["gamma"])
        err = k_pred - k_obs
        sse_lin += err * err
        abs_err_sum += abs(err)
        n += 1
    mae_lin = abs_err_sum / max(n, 1)
    return dict(params=params, sse_linear=float(sse_lin),
                mae_linear=float(mae_lin), n=int(n))


def _bound_penalty(x, bounds):
    pen = 0.0
    for v, (lo, hi) in zip(x, bounds):
        if v < lo:
            pen += 1e3 * (lo - v) ** 2
        elif v > hi:
            pen += 1e3 * (v - hi) ** 2
    return pen


def task3_kstar_full_model_comparison(run_dir, taus=(0.01, 0.02, 0.05)):
    """Fit the 3-param independent and 5-param correlated noise models
    directly on the k*(L) data at each threshold."""
    from analysis_model_comparison import load_rq2
    rq2_summary = os.path.join(run_dir, "rq2_at5us_cs1p0",
                                "summary_per_L_RQ2.csv")
    rq2_npz = os.path.join(run_dir, "rq2_at5us_cs1p0", "raw_vectors_RQ2.npz")
    _Lall, _cbf, chain_lengths_dict, _std = load_rq2(rq2_summary, rq2_npz)

    rows = []
    by_tau = {}
    for tau in taus:
        tau_tag = f"{tau:.2f}".replace(".", "p")
        csv_path = os.path.join(run_dir, "phase2",
                                 f"kstar_points_tau{tau_tag}.csv")
        df = pd.read_csv(csv_path)
        df = df[(df["L"] > 0) & (df["kstar"] > 0)].copy()
        L_obs = df["L"].astype(int).tolist()
        k_obs = df["kstar"].astype(float).tolist()
        safe_print(f"  fitting tau={tau}: {len(L_obs)} (L, k*) points")

        indep = fit_kstar_full(L_obs, k_obs, chain_lengths_dict, tau,
                                model="independent", seed=42)
        corr = fit_kstar_full(L_obs, k_obs, chain_lengths_dict, tau,
                               model="correlated", seed=42)

        # Sanity: 5-param should not be worse than 3-param (set rho=0,
        # gamma=1 is a valid sub-fit). If it is, log a warning.
        if corr["sse_linear"] > indep["sse_linear"] + 1e-9:
            safe_print(f"    [warn] tau={tau}: correlated SSE "
                       f"({corr['sse_linear']:.4e}) > independent "
                       f"({indep['sse_linear']:.4e}) — likely local minimum; "
                       "increasing restarts could help")

        n = indep["n"]
        aic_i, bic_i = _aic_bic_lin(indep["sse_linear"], n, k=3)
        aic_c, bic_c = _aic_bic_lin(corr["sse_linear"], n, k=5)
        d_aic = aic_c - aic_i
        d_bic = bic_c - bic_i

        for model_name, fit_res, n_params, aic, bic in (
            ("independent", indep, 3, aic_i, bic_i),
            ("correlated",  corr,  5, aic_c, bic_c),
        ):
            p = fit_res["params"]
            rows.append({
                "tau": tau,
                "model": model_name,
                "n_points": n,
                "n_params": n_params,
                "sigma_h": p["sigma_h"],
                "sigma_c": p["sigma_c"],
                "kappa":   p["kappa"],
                "rho_corr": (p["rho_corr"] if model_name == "correlated"
                             else float("nan")),
                "gamma":    (p["gamma"]    if model_name == "correlated"
                             else float("nan")),
                "sse_linear": fit_res["sse_linear"],
                "mae_linear": fit_res["mae_linear"],
                "aic": aic,
                "bic": bic,
                "delta_aic_corr_minus_indep": d_aic,
                "delta_bic_corr_minus_indep": d_bic,
            })
        by_tau[tau] = dict(indep=indep, corr=corr,
                           aic=(aic_i, aic_c), bic=(bic_i, bic_c),
                           delta_aic=d_aic, delta_bic=d_bic, n=n)
    return pd.DataFrame(rows), by_tau


def _aic_bic_lin(sse, n, k):
    if n <= 0 or not np.isfinite(sse) or sse <= 0:
        return float("nan"), float("nan")
    aic = n * np.log(sse / n) + 2 * k
    bic = n * np.log(sse / n) + k * np.log(n)
    return float(aic), float(bic)


def print_task3_table(df, by_tau):
    safe_print("\n" + "=" * 116)
    safe_print("Task 3 — Full noise-model comparison on k*(L) "
               "(3-param independent vs 5-param correlated, linear-space SSE)")
    safe_print("=" * 116)
    safe_print(
        f"  {'tau':>5}  {'model':>11}  {'N':>2}  "
        f"{'σ_h':>7}  {'σ_c':>7}  {'κ':>6}  "
        f"{'ρ_corr':>8}  {'γ':>6}  "
        f"{'SSE_lin':>10}  {'MAE_lin':>8}  {'AIC':>8}  {'BIC':>8}")
    for tau, info in by_tau.items():
        for model_name, fit_res, aic, bic in (
            ("independent", info["indep"], info["aic"][0], info["bic"][0]),
            ("correlated",  info["corr"],  info["aic"][1], info["bic"][1]),
        ):
            p = fit_res["params"]
            rho_s = (f"{p['rho_corr']:.4f}" if model_name == "correlated"
                     else "  —   ")
            gamma_s = (f"{p['gamma']:.3f}" if model_name == "correlated"
                       else "  —  ")
            safe_print(
                f"  {tau:>5.2f}  {model_name:>11s}  {info['n']:>2d}  "
                f"{p['sigma_h']:>7.4f}  {p['sigma_c']:>7.4f}  "
                f"{p['kappa']:>6.3f}  "
                f"{rho_s:>8s}  {gamma_s:>6s}  "
                f"{fit_res['sse_linear']:>10.4e}  "
                f"{fit_res['mae_linear']:>8.4f}  "
                f"{aic:>8.2f}  {bic:>8.2f}")
        win_aic = "correlated" if info["delta_aic"] < 0 else "independent"
        win_bic = "correlated" if info["delta_bic"] < 0 else "independent"
        safe_print(
            f"        {'':>11}      "
            f"ΔAIC={info['delta_aic']:+.2f} (winner: {win_aic})  "
            f"ΔBIC={info['delta_bic']:+.2f} (winner: {win_bic})")
        safe_print("-" * 116)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
SEEDS = (1, 2, 3, 4, 5, 42, 100, 1000)


def build_parser():
    p = argparse.ArgumentParser(
        description="Followup model-comparison analyses (Tasks 1 + 2).")
    p.add_argument("--run-dir", default="out/2026-05-06T23-07-42Z")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir
    if not os.path.isdir(run_dir):
        safe_print(f"ERROR: run-dir {run_dir!r} not found.")
        return 1
    phase2 = os.path.join(run_dir, "phase2")
    os.makedirs(phase2, exist_ok=True)

    # Task 1
    safe_print("\n=== Task 1 — CBF seed robustness ===")
    task1_df = task1_cbf_seed_robustness(run_dir, seeds=SEEDS)
    task1_csv = os.path.join(phase2, "rq6_correlated_seed_robustness.csv")
    task1_df.to_csv(task1_csv, index=False)
    safe_print(f"[saved] {task1_csv}")
    print_task1_table(task1_df)

    # Task 2
    safe_print("\n=== Task 2 — k* power-law model comparison ===")
    task2_df, fits_by_tau = task2_kstar_comparison(run_dir)
    task2_csv = os.path.join(phase2, "rq6_kstar_model_comparison.csv")
    task2_df.to_csv(task2_csv, index=False)
    safe_print(f"[saved] {task2_csv}")
    print_task2_table(task2_df, fits_by_tau)

    # Task 3
    safe_print("\n=== Task 3 — Full 5-param noise-model fit on k*(L) ===")
    task3_df, by_tau3 = task3_kstar_full_model_comparison(run_dir)
    task3_csv = os.path.join(phase2, "rq6_kstar_5param_comparison.csv")
    task3_df.to_csv(task3_csv, index=False)
    safe_print(f"[saved] {task3_csv}")
    print_task3_table(task3_df, by_tau3)

    # Interpretation
    print_interpretation(task1_df, fits_by_tau)
    return 0


if __name__ == "__main__":
    sys.exit(main())
