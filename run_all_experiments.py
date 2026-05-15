"""run_all_experiments.py — Phase 1.5 orchestrator (six QPU experiments).

Imports ``run_experiment_and_fit`` and ``run_and_fit_chain_strength_sweep_flat``
from rq234_revised and dispatches the six experiments specified in the
manuscript revision plan:

  Exp 1   RQ2  AT=5us   cs=1.0           -> rq2_at5us_cs1p0      label=RQ2
  Exp 2a  RQ3  AT=5us   cs=1.5           -> rq3_at5us_cs1p5      label=RQ3_AT5us
  Exp 2b  RQ3  AT=20us  cs=1.5           -> rq3_at20us_cs1p5     label=RQ3_AT20us
  Exp 2c  RQ3  AT=100us cs=1.5           -> rq3_at100us_cs1p5    label=RQ3_AT100us
  Exp 2d  RQ3  AT=200us cs=1.5           -> rq3_at200us_cs1p5    label=RQ3_AT200us
  Exp 3   RQ4  AT=20us  cs sweep [.1..2.5] -> rq4_at20us_cs_sweep  label=RQ4

Each experiment uses ``exact_out_dir=True`` so the manuscript directory
names are preserved verbatim. Idempotent: if the expected
``summary_per_L_<label>.csv`` already exists with at least the configured
number of L rows, the experiment is logged as SKIPPED and the orchestrator
moves on.

Failure isolation: per-experiment exceptions are caught, redacted via
safe_print, and the experiment is marked FAILED. The orchestrator continues
through the remaining experiments. Exit code 1 only if at least one
experiment failed (skipped/passed do not fail).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Token redaction
# ----------------------------------------------------------------------------
_TOK = os.getenv("DWAVE_API_TOKEN", "")


def safe_print(msg, file=None):
    s = str(msg)
    if _TOK and _TOK in s:
        s = s.replace(_TOK, "***REDACTED***")
    print(s, file=file or sys.stdout, flush=True)


def safe_str(msg):
    s = str(msg)
    if _TOK and _TOK in s:
        s = s.replace(_TOK, "***REDACTED***")
    return s


# ----------------------------------------------------------------------------
# Experiment plan
# ----------------------------------------------------------------------------
EXPERIMENTS = [
    dict(key="exp1",  rq="RQ2", at_us=5.0,   cs=1.0,    sweep=False,
         subdir="rq2_at5us_cs1p0",     label="RQ2"),
    dict(key="exp2a", rq="RQ3", at_us=5.0,   cs=1.5,    sweep=False,
         subdir="rq3_at5us_cs1p5",     label="RQ3_AT5us"),
    dict(key="exp2b", rq="RQ3", at_us=20.0,  cs=1.5,    sweep=False,
         subdir="rq3_at20us_cs1p5",    label="RQ3_AT20us"),
    dict(key="exp2c", rq="RQ3", at_us=100.0, cs=1.5,    sweep=False,
         subdir="rq3_at100us_cs1p5",   label="RQ3_AT100us"),
    dict(key="exp2d", rq="RQ3", at_us=200.0, cs=1.5,    sweep=False,
         subdir="rq3_at200us_cs1p5",   label="RQ3_AT200us"),
    dict(key="exp3",  rq="RQ4", at_us=20.0,  cs=None,   sweep=True,
         subdir="rq4_at20us_cs_sweep", label="RQ4"),
]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def utc_now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def build_L_list(L_min, L_step, dry_run=False, fallback_max=100):
    """Determine L_list from the solver's largest_clique_size.

    Under --dry-run we don't hit the network; use the user's documented
    expectation (5..100 step 5).
    """
    if dry_run:
        L_list = list(range(L_min, fallback_max + 1, L_step))
        return L_list, fallback_max, "dry-run-fallback"
    from qa_utils import get_clique_sampler
    cs_sampler = get_clique_sampler()
    L_max = getattr(cs_sampler, "largest_clique_size", None)
    if not isinstance(L_max, int) or L_max <= 0:
        raise RuntimeError("solver.largest_clique_size unavailable")
    if L_min > L_max:
        L_min = L_max
    L_list = list(range(L_min, L_max + 1, L_step)) or [L_max]
    return L_list, L_max, "solver"


def is_idempotent_skip(out_dir, label, n_expected, cs_grid=None):
    """Return True if the experiment looks complete enough to skip.

    Non-sweep experiments: ``summary_per_L_<label>.csv`` must exist with at
    least ``n_expected`` rows and at least one non-NaN ``mean_cbf_obs``.

    Sweep experiments (``cs_grid`` provided): in addition to the above,
    every ``chain_strength`` value in ``cs_grid`` must have at least one
    row with non-NaN ``mean_cbf_obs``. If any cs is fully NaN, the
    experiment is NOT complete.

    This stricter rule fixes the previous bypass where a single complete cs
    (cs=0.1) made the aggregate ``df["mean_cbf_obs"].notna().any()`` return
    True even though 23 of 25 cs values were empty.
    """
    p = os.path.join(out_dir, f"summary_per_L_{label}.csv")
    if not os.path.isfile(p):
        return False
    try:
        df = pd.read_csv(p)
    except Exception:
        return False
    if "mean_cbf_obs" not in df.columns:
        return False

    if cs_grid is not None:
        if "chain_strength" not in df.columns:
            return False
        for cs in cs_grid:
            sub = df[(df["chain_strength"] - float(cs)).abs() < 1e-6]
            if sub.empty or sub["mean_cbf_obs"].isna().all():
                return False
        return True

    if len(df) < n_expected:
        return False
    return bool(df["mean_cbf_obs"].notna().any())


def add_nan_rows_for_missing_L(csv_path, L_list, sampler_mode):
    """If summary_per_L_<label>.csv is missing rows for some L (because the
    QPU call failed), append NaN rows so Phase 2 can detect them."""
    if not os.path.isfile(csv_path):
        return [], []
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return [], []
    have = set(df["L"].astype(int).tolist()) if "L" in df.columns else set()
    missing = [int(L) for L in L_list if int(L) not in have]
    if not missing:
        return [], []
    nan_rows = []
    for L in missing:
        row = {c: np.nan for c in df.columns}
        row["L"] = L
        row["sampler_mode"] = sampler_mode
        nan_rows.append(row)
    df_out = pd.concat([df, pd.DataFrame(nan_rows)],
                       ignore_index=True).sort_values("L")
    df_out.to_csv(csv_path, index=False)
    return missing, list(df_out["L"].astype(int))


def per_experiment_qpu_access_sec(csv_path):
    """Sum qpu_access_time_us_mean over per-L rows × n_reps if present."""
    if not os.path.isfile(csv_path):
        return float("nan")
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return float("nan")
    col = "qpu_access_time_us_mean"
    if col not in df.columns:
        return float("nan")
    # mean per L × number of reps (we'd need n_reps from somewhere; the
    # CSV has n_total which is reads-summed, but qpu_access_time_us_mean
    # is per-replicate. For an estimate we sum (mean × replicate_count_estimate).
    # n_total ≈ n_reps × num_reads, so n_reps ≈ n_total / num_reads.
    if "n_total" in df.columns:
        # rough: use known num_reads default
        rep_est = df["n_total"].fillna(0).astype(float) / 2000.0
    else:
        rep_est = 1.0
    total_us = (df[col].fillna(0).astype(float) * rep_est).sum()
    return float(total_us / 1e6)


# ----------------------------------------------------------------------------
# Single-experiment dispatcher
# ----------------------------------------------------------------------------
def run_one(exp, out_parent, L_list, n_reps, num_reads, sampler_mode,
            dry_run, cs_grid_for_sweep):
    out_dir = os.path.join(out_parent, exp["subdir"])
    label = exp["label"]
    summary_csv = os.path.join(out_dir, f"summary_per_L_{label}.csv")

    started = time.time()
    started_iso = utc_now_iso()
    cs_str = "cs sweep" if exp["sweep"] else f"cs={exp['cs']}"
    safe_print(f"\n=== {exp['key']} {exp['rq']}  AT={exp['at_us']}us  "
               f"{cs_str}  -> {out_dir} ===")

    if dry_run:
        safe_print(f"  [DRY] would call "
                   f"{'run_and_fit_chain_strength_sweep_flat' if exp['sweep'] else 'run_experiment_and_fit'} "
                   f"with label={label!r} exact_out_dir=True L={L_list[0]}..{L_list[-1]}")
        return dict(key=exp["key"], status="DRYRUN", wall_sec=0.0,
                    qpu_access_sec=float("nan"), failed_L=[],
                    started=started_iso, finished=utc_now_iso(),
                    out_dir=out_dir, label=label)

    # Idempotency (sweep-aware)
    cs_grid_for_check = cs_grid_for_sweep if exp["sweep"] else None
    if is_idempotent_skip(out_dir, label, len(L_list),
                          cs_grid=cs_grid_for_check):
        if cs_grid_for_check is not None:
            safe_print(f"  [SKIP] {summary_csv} has non-NaN data for every "
                       f"cs in cs_grid ({len(cs_grid_for_check)} values)")
        else:
            safe_print(f"  [SKIP] {summary_csv} already has >= "
                       f"{len(L_list)} L rows with non-NaN data")
        return dict(key=exp["key"], status="SKIPPED",
                    wall_sec=time.time() - started,
                    qpu_access_sec=per_experiment_qpu_access_sec(summary_csv),
                    failed_L=[],
                    started=started_iso, finished=utc_now_iso(),
                    out_dir=out_dir, label=label)

    os.makedirs(out_dir, exist_ok=True)
    try:
        # Local import so --dry-run doesn't pull in dwave-system
        from rq234_revised import (
            run_experiment_and_fit,
            run_and_fit_chain_strength_sweep_flat,
        )
        if exp["sweep"]:
            run_and_fit_chain_strength_sweep_flat(
                L_list=L_list, cs_values=cs_grid_for_sweep,
                n_reps=n_reps, num_reads=num_reads,
                anneal_time_us=exp["at_us"],
                out_root=out_dir,
                sampler_mode=sampler_mode,
                show_progress=False, checkpoint_every=1,
                make_plots=True, verbose=True,
                label=label,
                exact_out_dir=True,
            )
        else:
            run_experiment_and_fit(
                L_list=L_list,
                n_reps=n_reps, num_reads=num_reads,
                chain_strength_mode=float(exp["cs"]),
                anneal_time_us=exp["at_us"],
                out_dir=out_dir,
                sampler_mode=sampler_mode,
                show_progress=False, checkpoint_every=1,
                make_plots=True, verbose=True,
                label=label,
                exact_out_dir=True,
            )
        status = "PASS"
        reason = None
    except Exception as e:
        tb = traceback.format_exc()
        safe_print(f"  [FAIL] {exp['key']}: {safe_str(e)}")
        safe_print(safe_str(tb))
        status = "FAIL"
        reason = safe_str(e)

    failed_L, _ = add_nan_rows_for_missing_L(summary_csv, L_list, sampler_mode)
    qpu_sec = per_experiment_qpu_access_sec(summary_csv)
    return dict(
        key=exp["key"],
        status=status,
        wall_sec=time.time() - started,
        qpu_access_sec=qpu_sec,
        failed_L=failed_L,
        started=started_iso,
        finished=utc_now_iso(),
        out_dir=out_dir,
        label=label,
        reason=reason,
    )


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="Phase 1.5 orchestrator — six QPU experiments.")
    p.add_argument("--output-parent", required=False, default=None,
                   help="Parent output dir, e.g. out/<ts>/")
    p.add_argument("--dry-run", action="store_true",
                   help="Print planned dispatch without calling QPU.")
    p.add_argument("--n-reps", type=int, default=10)
    p.add_argument("--num-reads", type=int, default=2000)
    p.add_argument("--L-min", type=int, default=5)
    p.add_argument("--L-step", type=int, default=5)
    p.add_argument("--cs-min", type=float, default=0.1)
    p.add_argument("--cs-max", type=float, default=2.5)
    p.add_argument("--cs-step", type=float, default=0.1)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    sampler_mode = os.getenv("SAMPLER_MODE", "fixed_embedding")

    out_parent = args.output_parent
    if out_parent is None and not args.dry_run:
        out_parent = f"out/{utc_now_iso()}"
    elif out_parent is None:
        out_parent = "out/<ts>"

    safe_print(f"orchestrator: out_parent={out_parent} "
               f"sampler_mode={sampler_mode} n_reps={args.n_reps} "
               f"num_reads={args.num_reads}")

    L_list, L_max, src = build_L_list(args.L_min, args.L_step, dry_run=args.dry_run)
    safe_print(f"L_list (from {src}, L_max={L_max}): {L_list}")

    cs_grid = list(np.round(
        np.arange(args.cs_min, args.cs_max + args.cs_step / 2, args.cs_step), 2))
    safe_print(f"cs sweep grid: {cs_grid}")

    if args.dry_run:
        safe_print("\nPlanned dispatch:")
        for exp in EXPERIMENTS:
            kind = "sweep" if exp["sweep"] else f"cs={exp['cs']}"
            safe_print(f"  {exp['key']:5s}  {exp['rq']}  AT={exp['at_us']:>6.1f}us  "
                       f"{kind:>14s}  -> {os.path.join(out_parent, exp['subdir'])}  "
                       f"label={exp['label']}")
        safe_print("\n--dry-run: no QPU calls made. Exiting 0.")
        return 0

    if not out_parent:
        safe_print("ERROR: --output-parent is required for non-dry-run.")
        return 1

    os.makedirs(out_parent, exist_ok=True)
    log_dir = os.path.join(out_parent, "logs")
    os.makedirs(log_dir, exist_ok=True)

    cooldown_sec = float(os.getenv("INTER_EXPERIMENT_COOLDOWN_SEC", "60"))

    started = time.time()
    started_iso = utc_now_iso()
    results = {}
    for i, exp in enumerate(EXPERIMENTS):
        if i > 0 and cooldown_sec > 0:
            safe_print(f"\n[cooldown] sleeping {cooldown_sec:.0f}s between "
                       "experiments to avoid rate limit")
            time.sleep(cooldown_sec)
        try:
            results[exp["key"]] = run_one(
                exp, out_parent, L_list, args.n_reps, args.num_reads,
                sampler_mode, dry_run=False, cs_grid_for_sweep=cs_grid)
        except Exception as e:
            safe_print(f"  [BUG] orchestrator-level error in {exp['key']}: "
                       f"{safe_str(e)}")
            results[exp["key"]] = dict(
                key=exp["key"], status="FAIL", wall_sec=0.0,
                qpu_access_sec=float("nan"), failed_L=[],
                reason=f"orchestrator-bug: {safe_str(e)}",
            )

    # Write summary JSON
    finished = time.time()
    summary = dict(
        phase="1.5",
        started=started_iso,
        finished=utc_now_iso(),
        total_wall_sec=finished - started,
        total_qpu_access_sec=float(sum(
            r.get("qpu_access_sec", 0.0) or 0.0 for r in results.values()
            if not (isinstance(r.get("qpu_access_sec"), float)
                    and np.isnan(r.get("qpu_access_sec", float("nan"))))
        )),
        sampler_mode=sampler_mode,
        L_list=L_list,
        L_max=L_max,
        n_reps=args.n_reps,
        num_reads=args.num_reads,
        cs_grid=cs_grid,
        experiments=results,
    )
    summary_path = os.path.join(log_dir, "phase1_5_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    safe_print(f"\n[saved] {summary_path}")

    any_fail = any(r["status"] == "FAIL" for r in results.values())
    safe_print(f"\nphase 1.5 final: "
               f"{sum(1 for r in results.values() if r['status']=='PASS')} PASS, "
               f"{sum(1 for r in results.values() if r['status']=='SKIPPED')} SKIPPED, "
               f"{sum(1 for r in results.values() if r['status']=='FAIL')} FAIL")
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
