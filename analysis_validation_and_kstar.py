"""analysis_validation_and_kstar.py — Phase 2 (TASK E + TASK F).

TASK E — Calibration / validation splits
    Fit ICE on a *training* subset of L values and evaluate CBF prediction
    error on the held-out *test* L values, under two split strategies:

        size_extrapolation  train on L <= L_cut, test on L > L_cut
        interleaved         train on every-other L, test on remainder

TASK F — k* bootstrap CIs
    For each tau in {0.01, 0.02, 0.05}:
        - Per L, compute k*(L) = chain-strength at which observed mean CBF
          first crosses tau (linear interpolation in cs).
        - Fit power law:  log k* = log a + alpha * log L
        - Bootstrap (n_iters resamples, default 1000) over L points -> 95% CI
          on alpha. Plot log-log with shaded CI band.

Synthetic alpha=0.5 recovery sanity check
    Constructs synthetic CBF(L, cs) curves with closed-form k*(L) = a * L^0.5,
    runs the full extraction + bootstrap pipeline, and asserts the recovered
    alpha is within tolerance of 0.5.

CLI
    --selftest                 run only the synthetic recovery; exit 0 on PASS
    --summary <csv>            Phase 1.5 Exp 1 per-L summary
    --raw <npz>                Phase 1.5 Exp 1 raw NPZ (for chain_lengths)
    --sweep <csv>              Phase 1.5 Exp 3 sweep_summary_per_L.csv
    --output-dir <dir>         where to write CSVs/PDFs/TeX
    --bootstrap-iters N        default 1000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# Token-aware print
# ----------------------------------------------------------------------------
_TOK = os.getenv("DWAVE_API_TOKEN", "")


def safe_print(msg):
    s = str(msg)
    if _TOK and _TOK in s:
        s = s.replace(_TOK, "***REDACTED***")
    print(s, flush=True)


def utc_now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


# ----------------------------------------------------------------------------
# ICE fitting (small grid search; mirrors rq234_revised.fit_sigma_params)
# ----------------------------------------------------------------------------
from math import erf, sqrt
import itertools


def _sigma_eff_sq(ell, sh, sc):
    return ell * (sh ** 2) + max(0, ell - 1) * (sc ** 2)


def _p_break(ell, sh, sc, kappa):
    s2 = _sigma_eff_sq(int(ell), sh, sc)
    if s2 <= 0:
        return 0.0
    return 1.0 - erf(kappa / (sqrt(2.0) * sqrt(s2)))


def _cbf_from_lengths(lengths, sh, sc, kappa):
    if len(lengths) == 0:
        return 0.0
    return float(np.mean([_p_break(ell, sh, sc, kappa) for ell in lengths]))


def _fit_ice(rows,
             sh_grid=None, sc_grid=None, k_grid=None):
    if sh_grid is None:
        sh_grid = np.linspace(0.005, 0.08, 16)
    if sc_grid is None:
        sc_grid = np.linspace(0.005, 0.08, 16)
    if k_grid is None:
        k_grid = np.linspace(0.10, 1.00, 19)
    best, best_err = None, float("inf")
    for sh, sc, k in itertools.product(sh_grid, sc_grid, k_grid):
        err = 0.0
        for r in rows:
            err += (_cbf_from_lengths(r["lengths"], sh, sc, k) - r["cbf"]) ** 2
        if err < best_err:
            best_err = err
            best = (float(sh), float(sc), float(k))
    return dict(sigma_h=best[0], sigma_c=best[1], kappa=best[2], sse=best_err)


# ----------------------------------------------------------------------------
# TASK E — calibration/validation splits
# ----------------------------------------------------------------------------
def load_summary_and_raw(summary_csv, raw_npz):
    df = pd.read_csv(summary_csv)
    df = df.dropna(subset=["L", "mean_cbf_obs"]).sort_values("L").reset_index(drop=True)
    npz = np.load(raw_npz, allow_pickle=False)
    rows = []
    for _, r in df.iterrows():
        L = int(r["L"])
        key = f"chain_lengths_L{L}"
        if key not in npz.files:
            continue
        rows.append(dict(L=L, cbf=float(r["mean_cbf_obs"]),
                         lengths=np.asarray(npz[key], dtype=int)))
    npz.close()
    return df, rows


def split_size_extrapolation(rows, train_frac=0.5):
    n = len(rows)
    cut = max(1, int(round(train_frac * n)))
    train = rows[:cut]
    test = rows[cut:]
    return train, test


def split_interleaved(rows):
    train = [r for i, r in enumerate(rows) if i % 2 == 0]
    test = [r for i, r in enumerate(rows) if i % 2 == 1]
    return train, test


def evaluate_split(train, test, label):
    """Fit on train, predict on test; return per-L observation table."""
    assert len(set(r["L"] for r in train) & set(r["L"] for r in test)) == 0, \
        f"split '{label}' has overlapping L between train and test"
    if not train or not test:
        return [], dict(sigma_h=np.nan, sigma_c=np.nan, kappa=np.nan, sse=np.nan)
    fit = _fit_ice(train)
    out = []
    for r in train:
        pred = _cbf_from_lengths(r["lengths"], fit["sigma_h"], fit["sigma_c"], fit["kappa"])
        out.append(dict(split=label, set="train", L=r["L"],
                        cbf_obs=r["cbf"], cbf_pred=pred,
                        abs_err=abs(pred - r["cbf"])))
    for r in test:
        pred = _cbf_from_lengths(r["lengths"], fit["sigma_h"], fit["sigma_c"], fit["kappa"])
        out.append(dict(split=label, set="test", L=r["L"],
                        cbf_obs=r["cbf"], cbf_pred=pred,
                        abs_err=abs(pred - r["cbf"])))
    return out, fit


def run_validation_splits(rows, out_dir):
    parts = []
    fits = {}
    for label, splitter in (("size_extrapolation", split_size_extrapolation),
                            ("interleaved", split_interleaved)):
        if label == "size_extrapolation":
            train, test = splitter(rows, train_frac=0.5)
        else:
            train, test = splitter(rows)
        rows_eval, fit = evaluate_split(train, test, label)
        fits[label] = fit
        parts.extend(rows_eval)
    df = pd.DataFrame(parts)
    out_csv = os.path.join(out_dir, "validation_splits_summary.csv")
    df.to_csv(out_csv, index=False)
    safe_print(f"[saved] {out_csv}")

    # Per-set MAE table
    if not df.empty:
        agg = df.groupby(["split", "set"])["abs_err"].agg(
            ["count", "mean", "max"]).reset_index()
        tex_path = os.path.join(out_dir, "validation_splits_table.tex")
        _write_validation_tex(agg, fits, tex_path)
        safe_print(f"[saved] {tex_path}")
        return df, agg, fits, out_csv
    return df, pd.DataFrame(), fits, out_csv


def _write_validation_tex(agg_df, fits, path):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Calibration / validation splits — observed vs predicted "
        r"mean CBF.}",
        r"\begin{tabular}{lllrrr}",
        r"\hline",
        r"Split & Set & N & MAE & Max abs.\ err.\ & ICE $(\sigma_h, \sigma_c, \kappa)$ \\",
        r"\hline",
    ]
    for _, r in agg_df.iterrows():
        f = fits.get(r["split"], {})
        ice = f"$({f.get('sigma_h', float('nan')):.3f}, {f.get('sigma_c', float('nan')):.3f}, {f.get('kappa', float('nan')):.3f})$"
        lines.append(
            f"{r['split'].replace('_', r'\_')} & {r['set']} & "
            f"{int(r['count'])} & {r['mean']:.4f} & {r['max']:.4f} & {ice} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------
# TASK F — k* bootstrap
# ----------------------------------------------------------------------------
TAUS = [0.01, 0.02, 0.05]


def _tau_tag(tau):
    return f"{tau:.2f}".replace(".", "p")


def kstar_from_sweep_per_L(df_sweep, tau):
    """For each L, find k*(L) = cs at first crossing of CBF(cs) <= tau.

    Returns a list of (L, kstar) tuples. L values where the CBF curve never
    crosses tau within the swept cs range are skipped (no extrapolation
    outside CBF crossing — sanity guarantee).
    """
    pts = []
    for L, sub in df_sweep.dropna(subset=["mean_cbf_obs"]).groupby("L"):
        sub = sub.sort_values("chain_strength")
        cs = sub["chain_strength"].to_numpy()
        cb = sub["mean_cbf_obs"].to_numpy()
        if cs.size < 2:
            continue
        # CBF should be monotonically decreasing in cs (broadly). Find first
        # crossing from above to below tau.
        cmin, cmax = float(np.min(cb)), float(np.max(cb))
        if not (cmin <= tau <= cmax):
            # CBF curve doesn't bracket tau -> no crossing -> skip this L.
            continue
        # Linear interpolation for the bracketed crossing.
        kstar = None
        for i in range(len(cs) - 1):
            a, b = cb[i], cb[i + 1]
            if (a - tau) * (b - tau) <= 0 and a != b:
                t = (tau - a) / (b - a)
                kstar = float(cs[i] + t * (cs[i + 1] - cs[i]))
                break
        if kstar is None:
            continue
        pts.append((int(L), kstar))
    return pts


def fit_powerlaw(points):
    """Fit log k* = log a + alpha * log L. Returns (alpha, log_a)."""
    if len(points) < 2:
        return float("nan"), float("nan")
    L = np.array([p[0] for p in points], dtype=float)
    k = np.array([p[1] for p in points], dtype=float)
    pos = (L > 0) & (k > 0)
    if pos.sum() < 2:
        return float("nan"), float("nan")
    x = np.log(L[pos])
    y = np.log(k[pos])
    coeffs = np.polyfit(x, y, 1)
    alpha = float(coeffs[0])
    log_a = float(coeffs[1])
    return alpha, log_a


def bootstrap_alpha(points, n_iters=1000, rng=None):
    if rng is None:
        rng = np.random.default_rng(int(os.getenv("MASTER_SEED", "42")))
    if len(points) < 2:
        return np.array([np.nan])
    n = len(points)
    alphas = np.empty(n_iters, dtype=float)
    for i in range(n_iters):
        idx = rng.integers(0, n, size=n)
        resamp = [points[j] for j in idx]
        alpha, _ = fit_powerlaw(resamp)
        alphas[i] = alpha
    return alphas


def run_kstar_bootstrap(df_sweep, out_dir, n_iters):
    rows = []
    fig_paths = []
    for tau in TAUS:
        pts = kstar_from_sweep_per_L(df_sweep, tau)
        if len(pts) < 2:
            safe_print(f"[kstar] tau={tau}: <2 crossing points, skipping")
            rows.append(dict(tau=tau, n_points=len(pts),
                             alpha=np.nan, alpha_low=np.nan, alpha_high=np.nan,
                             a=np.nan))
            continue
        # Save raw points
        df_pts = pd.DataFrame(pts, columns=["L", "kstar"])
        df_pts["tau"] = tau
        pts_csv = os.path.join(out_dir, f"kstar_points_tau{_tau_tag(tau)}.csv")
        df_pts.to_csv(pts_csv, index=False)
        safe_print(f"[saved] {pts_csv}")

        alpha, log_a = fit_powerlaw(pts)
        alphas = bootstrap_alpha(pts, n_iters=n_iters)
        valid = alphas[~np.isnan(alphas)]
        if len(valid) >= 2:
            lo, hi = np.quantile(valid, [0.025, 0.975])
        else:
            lo, hi = np.nan, np.nan
        rows.append(dict(tau=tau, n_points=len(pts),
                         alpha=alpha, alpha_low=float(lo), alpha_high=float(hi),
                         a=float(np.exp(log_a)) if not np.isnan(log_a) else np.nan))
        fig_paths.append(_plot_kstar_loglog(pts, tau, alpha, log_a, lo, hi, out_dir))

    df_boot = pd.DataFrame(rows)
    boot_csv = os.path.join(out_dir, "kstar_powerlaw_bootstrap.csv")
    df_boot.to_csv(boot_csv, index=False)
    safe_print(f"[saved] {boot_csv}")

    tex_path = os.path.join(out_dir, "kstar_powerlaw_bootstrap.tex")
    _write_kstar_tex(df_boot, tex_path)
    safe_print(f"[saved] {tex_path}")

    return df_boot, fig_paths


def _plot_kstar_loglog(points, tau, alpha, log_a, alpha_low, alpha_high, out_dir):
    fig = plt.figure(figsize=(6, 4))
    L = np.array([p[0] for p in points], dtype=float)
    k = np.array([p[1] for p in points], dtype=float)
    plt.loglog(L, k, "o", label=f"observed k* (tau={tau})")
    if not np.isnan(alpha):
        Lg = np.linspace(L.min(), L.max(), 50)
        plt.loglog(Lg, np.exp(log_a) * Lg ** alpha, "-",
                   label=fr"fit: $\alpha={alpha:.3f}$")
        if not np.isnan(alpha_low):
            plt.fill_between(
                Lg,
                np.exp(log_a) * Lg ** alpha_low,
                np.exp(log_a) * Lg ** alpha_high,
                alpha=0.2, label=f"95% CI [{alpha_low:.3f}, {alpha_high:.3f}]")
    plt.xlabel("L (logical variables)")
    plt.ylabel("k* (chain strength at CBF crossing)")
    plt.title(f"k* scaling, tau={tau}")
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    out = os.path.join(out_dir, f"fig_kstar_scaling_loglog_tau{_tau_tag(tau)}_with_CI.pdf")
    plt.savefig(out, dpi=160)
    plt.close(fig)
    safe_print(f"[saved] {out}")
    return out


def _write_kstar_tex(df, path):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Power-law fit $k^* = a L^{\alpha}$ with bootstrap 95\% "
        r"confidence interval on $\alpha$ (1000 resamples).}",
        r"\begin{tabular}{rrrrr}",
        r"\hline",
        r"$\tau$ & $N$ & $\alpha$ & 95\% CI & $a$ \\",
        r"\hline",
    ]
    for _, r in df.iterrows():
        if np.isnan(r["alpha"]):
            lines.append(f"{r['tau']:.2f} & {int(r['n_points'])} & --- & --- & --- \\\\")
        else:
            lines.append(
                f"{r['tau']:.2f} & {int(r['n_points'])} & "
                f"{r['alpha']:.3f} & "
                f"[{r['alpha_low']:.3f}, {r['alpha_high']:.3f}] & "
                f"{r['a']:.3f} \\\\"
            )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------
# Synthetic alpha=0.5 recovery sanity check
# ----------------------------------------------------------------------------
def synthetic_sweep(alpha=0.5, a=0.3, L_values=None, cs_grid=None, noise_sd=0.003,
                    seed=42):
    """Build a synthetic sweep dataframe with closed-form scaling
    ``kstar(L) = a * L**alpha`` and a linear CBF model
    ``CBF = clip(1 - cs / kstar, 0, 1)``.

    With this model, CBF crosses any tau at ``cs = (1 - tau) * kstar(L)``.
    Default parameters put crossings inside ``[0.1, 2.5]`` for L ∈ [4, 64]:
        kstar(4) = 0.60, kstar(64) = 2.40.
    """
    rng = np.random.default_rng(seed)
    if L_values is None:
        # Squares of small integers => kstar evenly spaced in [0.6, 2.4]
        L_values = [4, 9, 16, 25, 36, 49, 64]
    if cs_grid is None:
        cs_grid = list(np.round(np.arange(0.1, 2.51, 0.1), 2))
    rows = []
    for L in L_values:
        kstar = a * (L ** alpha)
        for cs in cs_grid:
            cbf = max(0.0, 1.0 - cs / kstar)
            cbf = float(np.clip(cbf + rng.normal(0, noise_sd), 0.0, 1.0))
            rows.append(dict(L=L, chain_strength=cs, mean_cbf_obs=cbf,
                             anneal_time_us=20.0, sampler_mode="synthetic"))
    return pd.DataFrame(rows)


def run_synthetic_recovery(out_dir=None, n_iters=300, tolerance=0.10):
    """Run the full pipeline against synthetic alpha=0.5 data. Asserts
    recovered alpha is within ``tolerance`` of 0.5 for tau=0.05."""
    safe_print("\n[selftest] synthetic alpha=0.5 recovery")
    df = synthetic_sweep(alpha=0.5, a=0.3, noise_sd=0.003, seed=42)

    # Sub-sanity: train/test disjoint when split.
    rows = [dict(L=L, cbf=0.0, lengths=np.array([1] * 1)) for L in sorted(df["L"].unique())]
    tr1, te1 = split_size_extrapolation(rows, 0.5)
    tr2, te2 = split_interleaved(rows)
    assert not (set(r["L"] for r in tr1) & set(r["L"] for r in te1)), \
        "size_extrapolation split has overlap"
    assert not (set(r["L"] for r in tr2) & set(r["L"] for r in te2)), \
        "interleaved split has overlap"
    safe_print("[selftest] train/test disjoint guarantee: PASS")

    # Sub-sanity: no extrapolation outside CBF crossing.
    # Pick a tau that some L values bracket and others don't, then verify
    # every returned k* is from a CBF curve that does bracket it.
    for tau_check in (0.5, 0.2, 0.05):
        pts_check = kstar_from_sweep_per_L(df, tau=tau_check)
        for L, _k in pts_check:
            sub = df[df["L"] == L]
            cmin = float(sub["mean_cbf_obs"].min())
            cmax = float(sub["mean_cbf_obs"].max())
            assert cmin <= tau_check <= cmax, (
                f"no-extrapolation guarantee broken: L={L} doesn't bracket "
                f"tau={tau_check} (range [{cmin:.4f}, {cmax:.4f}])")
    safe_print("[selftest] no-extrapolation guarantee: PASS")

    # Recovery test at tau=0.05
    pts = kstar_from_sweep_per_L(df, tau=0.05)
    if len(pts) < 3:
        raise AssertionError(
            f"synthetic recovery: too few crossing points ({len(pts)}) for tau=0.05")
    alpha, log_a = fit_powerlaw(pts)
    alphas = bootstrap_alpha(pts, n_iters=n_iters)
    lo, hi = np.quantile(alphas[~np.isnan(alphas)], [0.025, 0.975])
    safe_print(f"[selftest] recovered alpha = {alpha:.3f}  "
               f"(95% CI [{lo:.3f}, {hi:.3f}])  expected 0.5  "
               f"tolerance ±{tolerance}")
    if abs(alpha - 0.5) > tolerance:
        raise AssertionError(
            f"synthetic alpha recovery FAILED: got {alpha:.3f} expected 0.5 "
            f"(|err|={abs(alpha-0.5):.3f} > {tolerance})")
    safe_print("[selftest] synthetic alpha=0.5 recovery: PASS")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "selftest_synthetic.json"), "w") as f:
            json.dump({"alpha": alpha, "alpha_low": float(lo),
                       "alpha_high": float(hi),
                       "n_points": len(pts), "expected": 0.5,
                       "tolerance": tolerance,
                       "result": "PASS"}, f, indent=2)
    return dict(alpha=alpha, alpha_low=float(lo), alpha_high=float(hi),
                n_points=len(pts), tolerance=tolerance)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="Phase 2: validation splits + k* bootstrap.")
    p.add_argument("--summary", default=None, help="Phase 1.5 RQ2 per-L CSV")
    p.add_argument("--raw", default=None, help="Phase 1.5 RQ2 raw NPZ")
    p.add_argument("--sweep", default=None, help="Phase 1.5 RQ4 sweep CSV")
    p.add_argument("--output-dir", default=None, help="Phase 2 output dir")
    p.add_argument("--bootstrap-iters", type=int, default=1000)
    p.add_argument("--selftest", action="store_true",
                   help="Run synthetic alpha=0.5 recovery only; exit 0 on PASS.")
    return p


def write_summary_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    safe_print(f"[saved] {path}")


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.selftest:
        try:
            res = run_synthetic_recovery(
                out_dir=args.output_dir, n_iters=300, tolerance=0.10)
            safe_print("\n[Phase 2 selftest] PASS")
            return 0
        except AssertionError as e:
            safe_print(f"\n[Phase 2 selftest] FAIL: {e}")
            return 1
        except Exception as e:
            safe_print(f"\n[Phase 2 selftest] ERROR: {e.__class__.__name__}: {e}")
            return 1

    # Real-data path
    if not (args.summary and args.raw and args.sweep and args.output_dir):
        safe_print("ERROR: --summary, --raw, --sweep, and --output-dir are "
                   "all required for the real-data run.")
        return 1

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    log_dir = os.path.join(os.path.dirname(out_dir.rstrip("/")), "logs")
    os.makedirs(log_dir, exist_ok=True)

    started_iso = utc_now_iso()
    t0 = time.time()
    summary = dict(phase="2", started=started_iso, status="UNKNOWN",
                   sanity_checks={}, real_data={})

    # 1) Sanity checks on synthetic data BEFORE real-data analysis.
    safe_print("=== Phase 2: sanity checks ===")
    try:
        s = run_synthetic_recovery(out_dir=out_dir, n_iters=300, tolerance=0.10)
        summary["sanity_checks"] = dict(status="PASS", **s)
    except AssertionError as e:
        summary["sanity_checks"] = dict(status="FAIL", reason=str(e))
        summary["status"] = "FAILED"
        summary["reason"] = "synthetic α recovery sanity check failed"
        write_summary_json(os.path.join(log_dir, "phase2_summary.json"), summary)
        safe_print(f"[Phase 2] FAILED: {e}")
        return 1
    except Exception as e:
        summary["sanity_checks"] = dict(status="ERROR",
                                        reason=f"{e.__class__.__name__}: {e}")
        summary["status"] = "FAILED"
        summary["reason"] = "sanity check raised unexpected error"
        write_summary_json(os.path.join(log_dir, "phase2_summary.json"), summary)
        safe_print(f"[Phase 2] FAILED: {e}")
        return 1

    # 2) Real-data analysis.
    safe_print("\n=== Phase 2: real-data analysis ===")
    try:
        df_summary, rows = load_summary_and_raw(args.summary, args.raw)
        if len(rows) < 4:
            summary["real_data"] = dict(
                status="SKIPPED",
                reason=f"only {len(rows)} L rows with chain_lengths_L*; need >=4")
            summary["status"] = "PASS"  # sanity passed; data insufficient
            write_summary_json(os.path.join(log_dir, "phase2_summary.json"), summary)
            return 0

        df_val, agg, fits, val_csv = run_validation_splits(rows, out_dir)

        df_sweep = pd.read_csv(args.sweep)
        df_boot, fig_paths = run_kstar_bootstrap(
            df_sweep, out_dir, n_iters=args.bootstrap_iters)

        summary["real_data"] = dict(
            status="PASS",
            n_L_rq2=len(rows),
            n_sweep_rows=len(df_sweep),
            ice_fits=fits,
            kstar=df_boot.to_dict(orient="records"),
            figures=fig_paths,
        )
        summary["status"] = "PASS"
    except Exception as e:
        summary["real_data"] = dict(status="FAILED",
                                    reason=f"{e.__class__.__name__}: {e}")
        summary["status"] = "FAILED"
    finally:
        summary["finished"] = utc_now_iso()
        summary["wall_sec"] = time.time() - t0

    write_summary_json(os.path.join(log_dir, "phase2_summary.json"), summary)
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
