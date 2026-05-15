"""rerun_failed_experiments.py — re-execute Phase 1.5 experiments that
failed (or partially failed) in an earlier run, then update the existing
``out/<ts>/logs/phase1_5_summary.json`` in place.

Workflow:

  1. Read ``<run-dir>/logs/phase1_5_summary.json``.
  2. Decide per experiment:
        FAIL                 -> full re-run (delete original CSV/NPZ first)
        PASS + failed_L != []-> partial re-run of failed_L only, then merge
                                with existing successful rows in the same
                                ``summary_per_L_<label>.csv`` and NPZ.
        PASS + failed_L = [] -> skip
  3. Use ``sample_with_retry`` (already wired into the runners) and the
     ``INTER_EXPERIMENT_COOLDOWN_SEC`` cooldown between experiments.
  4. Update phase1_5_summary.json with new per-experiment stats and append
     to a ``rerun_history`` list.

Phase 2 / Phase 3 / Phase 4 are NOT triggered by this script — that's the
job of the bash launcher (``rerun_overnight.sh``).

CLI
    python rerun_failed_experiments.py --run-dir out/<ts>
    python rerun_failed_experiments.py --run-dir out/<ts> --dry-run
    python rerun_failed_experiments.py --run-dir out/<ts> --selftest
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from run_all_experiments import EXPERIMENTS, build_L_list, safe_print, safe_str


_TOK = os.getenv("DWAVE_API_TOKEN", "")


def utc_now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


# ----------------------------------------------------------------------------
# Plan building
# ----------------------------------------------------------------------------
def _experiment_by_key(key):
    for exp in EXPERIMENTS:
        if exp["key"] == key:
            return exp
    return None


def build_plan(run_dir):
    """Return a list of plan entries describing what needs to be re-run."""
    sj_path = os.path.join(run_dir, "logs", "phase1_5_summary.json")
    if not os.path.isfile(sj_path):
        raise FileNotFoundError(
            f"phase1_5_summary.json not found at {sj_path}; "
            "rerun requires a prior run.")
    with open(sj_path) as f:
        sj = json.load(f)
    L_list = sj.get("L_list") or []
    plan = []
    for key, info in (sj.get("experiments") or {}).items():
        exp = _experiment_by_key(key)
        if exp is None:
            safe_print(f"[plan] WARNING: unknown experiment key {key!r}; skipping")
            continue
        status = info.get("status")
        failed_L = info.get("failed_L") or []
        if status == "FAIL":
            action = "full_rerun"
            L_to_run = list(L_list)
        elif status == "PASS" and failed_L:
            action = "partial_rerun"
            L_to_run = [int(L) for L in failed_L]
        elif status == "PASS":
            action = "skip"
            L_to_run = []
        elif status == "SKIPPED":
            action = "full_rerun"
            L_to_run = list(L_list)
        else:
            action = "full_rerun"
            L_to_run = list(L_list)
        plan.append(dict(
            key=key,
            exp=exp,
            action=action,
            L_to_run=L_to_run,
            prior_status=status,
            prior_failed_L=failed_L,
            prior_wall_sec=info.get("wall_sec"),
            prior_qpu_access_sec=info.get("qpu_access_sec"),
        ))
    return plan, sj, L_list


def estimate_qpu_seconds(plan, n_reps, num_reads, cs_grid_size,
                         per_call_sec_at_us=None):
    """Rough estimate. Assumes ~0.5s per call at AT=20us, scaled linearly
    with anneal time vs the 20us baseline (programming dominates)."""
    if per_call_sec_at_us is None:
        per_call_sec_at_us = {5: 0.49, 20: 0.50, 100: 0.66, 200: 0.86}

    total = 0.0
    for entry in plan:
        if entry["action"] == "skip":
            continue
        exp = entry["exp"]
        per_call = per_call_sec_at_us.get(int(exp["at_us"]), 0.5)
        if exp["sweep"]:
            total += cs_grid_size * len(entry["L_to_run"]) * n_reps * per_call
        else:
            total += len(entry["L_to_run"]) * n_reps * per_call
    return total


# ----------------------------------------------------------------------------
# Re-run primitives (full and partial)
# ----------------------------------------------------------------------------
def _key_belongs_to_L(key, L):
    return key.endswith(f"_L{L}") or f"_L{L}_" in key


def _filter_npz_keys_by_L_set(keys, L_set):
    """Return the subset of `keys` whose L-suffix is in L_set (any L).

    Handles both `chain_lengths_L8` and `cs1p0__chain_lengths_L8` (cs-sweep
    namespacing).
    """
    out = []
    for k in keys:
        for L in L_set:
            if _key_belongs_to_L(k, L):
                out.append(k)
                break
    return out


def _backup(src, dst):
    if os.path.isfile(src):
        shutil.copy2(src, dst)


def _delete_if_exists(*paths):
    for p in paths:
        try:
            if os.path.isfile(p):
                os.remove(p)
        except Exception as e:
            safe_print(f"[cleanup] failed to remove {p}: {safe_str(e)}")


def full_rerun_experiment(entry, run_dir, n_reps, num_reads, cs_grid):
    """Re-run an entire experiment from scratch. Removes prior CSV/NPZ."""
    exp = entry["exp"]
    out_dir = os.path.join(run_dir, exp["subdir"])
    os.makedirs(out_dir, exist_ok=True)
    label = exp["label"]
    summary_csv = os.path.join(out_dir, f"summary_per_L_{label}.csv")
    raw_npz = os.path.join(out_dir, f"raw_vectors_{label}.npz")

    # Clear stale artifacts so the rerun starts clean.
    _delete_if_exists(
        summary_csv,
        raw_npz,
        os.path.join(out_dir, f"summary_per_L_live_{label}.csv"),
        os.path.join(out_dir, f"per_replicate_{label}.csv"),
        os.path.join(out_dir, f"fit_obs_vs_pred_{label}.csv"),
        os.path.join(out_dir, f"fit_params_{label}.json"),
        os.path.join(out_dir, f"cbf_obs_pred_vs_L_{label}.pdf"),
    )

    sampler_mode = os.getenv("SAMPLER_MODE", "fixed_embedding")
    safe_print(f"\n[rerun:{entry['key']}] full re-run "
               f"(L_list={entry['L_to_run']}, AT={exp['at_us']}us, "
               f"sweep={exp['sweep']})")
    started = time.time()
    started_iso = utc_now_iso()

    from rq234_revised import (
        run_experiment_and_fit, run_and_fit_chain_strength_sweep_flat,
    )
    try:
        if exp["sweep"]:
            run_and_fit_chain_strength_sweep_flat(
                L_list=entry["L_to_run"], cs_values=cs_grid,
                n_reps=n_reps, num_reads=num_reads,
                anneal_time_us=exp["at_us"],
                out_root=out_dir,
                sampler_mode=sampler_mode,
                show_progress=False, checkpoint_every=1,
                make_plots=True, verbose=True,
                label=label, exact_out_dir=True,
            )
        else:
            run_experiment_and_fit(
                L_list=entry["L_to_run"],
                n_reps=n_reps, num_reads=num_reads,
                chain_strength_mode=float(exp["cs"]),
                anneal_time_us=exp["at_us"],
                out_dir=out_dir,
                sampler_mode=sampler_mode,
                show_progress=False, checkpoint_every=1,
                make_plots=True, verbose=True,
                label=label, exact_out_dir=True,
            )
        status = "PASS"
        reason = None
    except Exception as e:
        traceback.print_exc()
        safe_print(f"[rerun:{entry['key']}] FAILED: {safe_str(e)}")
        status, reason = "FAIL", safe_str(e)

    failed_L = _check_failed_L_in_csv(summary_csv, entry["L_to_run"])
    qpu_sec = _qpu_access_sec_from_csv(summary_csv, num_reads)
    return dict(
        status=("PASS" if status == "PASS" and not failed_L else
                ("PASS" if status == "PASS" else "FAIL")),
        reason=reason,
        wall_sec=time.time() - started,
        qpu_access_sec=qpu_sec,
        failed_L=failed_L,
        started=started_iso,
        finished=utc_now_iso(),
    )


def partial_rerun_experiment(entry, run_dir, n_reps, num_reads, cs_grid):
    """Re-run only the failed L values, then merge into the existing
    summary CSV and raw NPZ."""
    exp = entry["exp"]
    out_dir = os.path.join(run_dir, exp["subdir"])
    label = exp["label"]
    L_subset = list(map(int, entry["L_to_run"]))

    summary_csv = os.path.join(out_dir, f"summary_per_L_{label}.csv")
    raw_npz = os.path.join(out_dir, f"raw_vectors_{label}.npz")

    backup_csv = summary_csv + ".pre_rerun"
    backup_npz = raw_npz + ".pre_rerun"
    _backup(summary_csv, backup_csv)
    _backup(raw_npz, backup_npz)

    # The runner will overwrite summary_csv / raw_npz with subset-only data.
    safe_print(f"\n[rerun:{entry['key']}] partial re-run for L={L_subset}; "
               f"backed up original CSV/NPZ as .pre_rerun")
    started = time.time()
    started_iso = utc_now_iso()
    sampler_mode = os.getenv("SAMPLER_MODE", "fixed_embedding")

    from rq234_revised import (
        run_experiment_and_fit, run_and_fit_chain_strength_sweep_flat,
    )
    try:
        if exp["sweep"]:
            # cs sweep doesn't usually go partial — but support it
            run_and_fit_chain_strength_sweep_flat(
                L_list=L_subset, cs_values=cs_grid,
                n_reps=n_reps, num_reads=num_reads,
                anneal_time_us=exp["at_us"],
                out_root=out_dir,
                sampler_mode=sampler_mode,
                show_progress=False, checkpoint_every=1,
                make_plots=True, verbose=True,
                label=label, exact_out_dir=True,
            )
        else:
            run_experiment_and_fit(
                L_list=L_subset,
                n_reps=n_reps, num_reads=num_reads,
                chain_strength_mode=float(exp["cs"]),
                anneal_time_us=exp["at_us"],
                out_dir=out_dir,
                sampler_mode=sampler_mode,
                show_progress=False, checkpoint_every=1,
                make_plots=True, verbose=True,
                label=label, exact_out_dir=True,
            )
        runner_status = "PASS"
        runner_reason = None
    except Exception as e:
        traceback.print_exc()
        safe_print(f"[rerun:{entry['key']}] runner FAILED: {safe_str(e)}")
        runner_status, runner_reason = "FAIL", safe_str(e)

    # Merge: take the new subset rows, then add back the rows from the
    # backup that aren't in the subset.
    if os.path.isfile(backup_csv):
        merge_summary_csv(summary_csv, backup_csv, L_subset)
    if os.path.isfile(backup_npz):
        merge_npz(raw_npz, backup_npz, L_subset)

    failed_L = _check_failed_L_in_csv(summary_csv, L_subset)
    qpu_sec = _qpu_access_sec_from_csv(summary_csv, num_reads)
    return dict(
        status=("PASS" if runner_status == "PASS" and not failed_L else
                "FAIL" if runner_status != "PASS" else "PASS"),
        reason=runner_reason,
        wall_sec=time.time() - started,
        qpu_access_sec=qpu_sec,
        failed_L=failed_L,
        started=started_iso,
        finished=utc_now_iso(),
        merged_with=backup_csv,
    )


# ----------------------------------------------------------------------------
# Merge helpers
# ----------------------------------------------------------------------------
def merge_summary_csv(current_csv, backup_csv, L_subset):
    """Replace rows of L in L_subset with the new rows in `current_csv`,
    keeping all other L rows from `backup_csv`."""
    if not os.path.isfile(current_csv) or not os.path.isfile(backup_csv):
        return
    new_df = pd.read_csv(current_csv)
    old_df = pd.read_csv(backup_csv)
    L_subset_int = set(int(L) for L in L_subset)
    if "L" in old_df.columns:
        old_keep = old_df[~old_df["L"].astype("Int64").isin(L_subset_int)].copy()
    else:
        old_keep = old_df.iloc[0:0]
    merged = pd.concat([old_keep, new_df], ignore_index=True)
    if "L" in merged.columns:
        merged = merged.sort_values("L").reset_index(drop=True)
    merged.to_csv(current_csv, index=False)
    safe_print(f"[merge] {os.path.basename(current_csv)}: "
               f"old kept={len(old_keep)} + new={len(new_df)} = {len(merged)}")


def merge_npz(current_npz, backup_npz, L_subset):
    """Combine current NPZ (subset-only data) with backup NPZ (full prior
    data) into one. New keys win on conflict."""
    if not os.path.isfile(current_npz) or not os.path.isfile(backup_npz):
        return
    L_subset_int = set(int(L) for L in L_subset)
    merged = {}
    # Start from backup, drop keys belonging to any L in the subset.
    with np.load(backup_npz, allow_pickle=False) as old:
        for k in old.files:
            if any(_key_belongs_to_L(k, L) for L in L_subset_int):
                continue
            merged[k] = old[k]
    # Then overlay new keys (only subset L values).
    with np.load(current_npz, allow_pickle=False) as new:
        for k in new.files:
            merged[k] = new[k]
    np.savez(current_npz, **merged)
    safe_print(f"[merge] {os.path.basename(current_npz)}: "
               f"merged keys={len(merged)}")


# ----------------------------------------------------------------------------
# Failure detection from final CSV
# ----------------------------------------------------------------------------
def _check_failed_L_in_csv(csv_path, L_attempted):
    """Return list of L values in L_attempted whose mean_cbf_obs is NaN
    (or whose row is missing) in the produced CSV."""
    if not os.path.isfile(csv_path):
        return [int(L) for L in L_attempted]
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return [int(L) for L in L_attempted]
    if "L" not in df.columns or "mean_cbf_obs" not in df.columns:
        return [int(L) for L in L_attempted]
    bad = []
    for L in L_attempted:
        sub = df[df["L"].astype("Int64") == int(L)]
        if sub.empty or sub["mean_cbf_obs"].isna().all():
            bad.append(int(L))
    return bad


def _qpu_access_sec_from_csv(csv_path, num_reads):
    if not os.path.isfile(csv_path):
        return float("nan")
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return float("nan")
    col = "qpu_access_time_us_mean"
    if col not in df.columns or "n_total" not in df.columns:
        return float("nan")
    rep_est = df["n_total"].fillna(0).astype(float) / max(num_reads, 1)
    total_us = (df[col].fillna(0).astype(float) * rep_est).sum()
    return float(total_us / 1e6)


# ----------------------------------------------------------------------------
# Update phase1_5_summary.json in place
# ----------------------------------------------------------------------------
def update_phase1_5_summary(run_dir, plan, results):
    sj_path = os.path.join(run_dir, "logs", "phase1_5_summary.json")
    with open(sj_path) as f:
        sj = json.load(f)
    sj.setdefault("rerun_history", []).append(dict(
        timestamp=utc_now_iso(),
        rerun_keys=[entry["key"] for entry in plan
                    if entry["action"] != "skip"],
    ))
    exp_dict = sj.setdefault("experiments", {})
    for entry in plan:
        if entry["action"] == "skip":
            continue
        key = entry["key"]
        new = results.get(key)
        if not new:
            continue
        prev = exp_dict.get(key, {}) or {}
        prev_history = prev.get("history", []) + [dict(
            attempted_at=prev.get("started"),
            attempted_status=prev.get("status"),
            failed_L=prev.get("failed_L", []),
            wall_sec=prev.get("wall_sec"),
            qpu_access_sec=prev.get("qpu_access_sec"),
            reason=prev.get("reason"),
        )]
        merged = dict(prev)
        merged.update(new)
        merged["history"] = prev_history
        merged["key"] = key
        exp_dict[key] = merged
    with open(sj_path, "w") as f:
        json.dump(sj, f, indent=2, default=str)
    return sj_path


# ----------------------------------------------------------------------------
# Selftest
# ----------------------------------------------------------------------------
def _selftest_retry():
    """Verify sample_with_retry retries on rate-limit messages and propagates
    other exceptions immediately."""
    from qa_utils import sample_with_retry

    state = {"count": 0}

    class FakeRateLimit(Exception):
        pass

    def flaky(*args, **kwargs):
        state["count"] += 1
        if state["count"] < 3:
            raise FakeRateLimit("HTTP 429 too many requests; retry later")
        return "OK"

    out = sample_with_retry(flaky, max_retries=5, base_wait=0.0)
    assert out == "OK"
    assert state["count"] == 3, f"expected 3 attempts, got {state['count']}"

    # Non-rate-limit error must propagate without retry
    state2 = {"count": 0}

    def fatal(*args, **kwargs):
        state2["count"] += 1
        raise ValueError("syntax error in input")

    try:
        sample_with_retry(fatal, max_retries=5, base_wait=0.0)
    except ValueError:
        pass
    assert state2["count"] == 1, "non-rate-limit error should not retry"
    safe_print("[selftest 1] retry/backoff: PASS")


def _selftest_empty_rows_guard():
    """Verify run_experiment_and_fit produces an all-NaN summary when every
    measure_L_oneQUBO call fails (no 'L' KeyError crash)."""
    import tempfile
    import rq234_revised as mod

    original = mod.measure_L_oneQUBO

    def always_fail(L, **kwargs):
        raise RuntimeError("simulated rate-limit error from selftest")

    mod.measure_L_oneQUBO = always_fail
    try:
        with tempfile.TemporaryDirectory() as td:
            mod.run_experiment_and_fit(
                L_list=[5, 10, 15],
                n_reps=1, num_reads=10,
                chain_strength_mode=1.0,
                anneal_time_us=20.0,
                out_dir=td,
                sampler_mode="fixed_embedding",
                show_progress=False, checkpoint_every=1,
                make_plots=False, verbose=False,
                label="selftest", exact_out_dir=True,
            )
            csv_path = os.path.join(td, "summary_per_L_selftest.csv")
            assert os.path.isfile(csv_path), \
                "summary CSV missing after all-fail run"
            df = pd.read_csv(csv_path)
            assert "L" in df.columns, "summary CSV missing 'L' column"
            assert len(df) == 3, f"expected 3 rows, got {len(df)}"
            assert df["mean_cbf_obs"].isna().all(), \
                "expected all-NaN mean_cbf_obs in fall-through path"
    finally:
        mod.measure_L_oneQUBO = original
    safe_print("[selftest 2] empty-rows DataFrame guard: PASS")


def _selftest_idempotency_rejects_all_nan():
    """Verify is_idempotent_skip returns False on an all-NaN CSV."""
    import tempfile
    from run_all_experiments import is_idempotent_skip

    with tempfile.TemporaryDirectory() as td:
        df = pd.DataFrame({
            "L": [5, 10, 15],
            "mean_cbf_obs": [np.nan, np.nan, np.nan],
            "sampler_mode": ["fixed_embedding"] * 3,
        })
        df.to_csv(os.path.join(td, "summary_per_L_test.csv"), index=False)
        ok = is_idempotent_skip(td, "test", n_expected=3)
        assert ok is False, "all-NaN CSV must NOT be considered complete"

        # Sanity: a non-NaN row makes it complete.
        df2 = df.copy()
        df2.loc[0, "mean_cbf_obs"] = 0.1
        df2.to_csv(os.path.join(td, "summary_per_L_test2.csv"), index=False)
        ok2 = is_idempotent_skip(td, "test2", n_expected=3)
        assert ok2 is True, "non-NaN row should be considered complete"
    safe_print("[selftest 3] idempotency rejects all-NaN CSV: PASS")


def run_selftest():
    safe_print("\n=== rerun_failed_experiments selftest ===")
    try:
        _selftest_retry()
        _selftest_empty_rows_guard()
        _selftest_idempotency_rejects_all_nan()
    except AssertionError as e:
        safe_print(f"\n[selftest] FAIL: {e}")
        return 1
    except Exception as e:
        traceback.print_exc()
        safe_print(f"\n[selftest] ERROR: {e.__class__.__name__}: {safe_str(e)}")
        return 1
    safe_print("\n[selftest] PASS")
    return 0


# ----------------------------------------------------------------------------
# Dry-run printer
# ----------------------------------------------------------------------------
def print_plan(plan, L_list, n_reps, num_reads, cs_grid):
    safe_print("\n=== Rerun plan ===")
    for entry in plan:
        exp = entry["exp"]
        if entry["action"] == "skip":
            safe_print(f"  {entry['key']:5s} {exp['rq']:3s} AT={exp['at_us']}us "
                       f"-> SKIP (prior_status={entry['prior_status']}, "
                       f"failed_L=[])")
        else:
            kind = "sweep" if exp["sweep"] else f"cs={exp['cs']}"
            safe_print(f"  {entry['key']:5s} {exp['rq']:3s} AT={exp['at_us']}us "
                       f"{kind:>10s} -> {entry['action']:14s} "
                       f"L_to_run={entry['L_to_run']}")
    est = estimate_qpu_seconds(plan, n_reps, num_reads, len(cs_grid))
    safe_print(f"\nEstimated QPU access time (rough, AT-scaled): {est:.0f}s "
               f"(~{est/60:.1f} min)")


# ----------------------------------------------------------------------------
# CLI / main
# ----------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="Re-run Phase 1.5 experiments that failed previously.")
    p.add_argument("--run-dir", required=False, default=None,
                   help="Existing out/<ts>/ directory to update.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--n-reps", type=int, default=10)
    p.add_argument("--num-reads", type=int, default=2000)
    p.add_argument("--cs-min", type=float, default=0.1)
    p.add_argument("--cs-max", type=float, default=2.5)
    p.add_argument("--cs-step", type=float, default=0.1)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.selftest:
        return run_selftest()

    if not args.run_dir:
        safe_print("ERROR: --run-dir is required (or pass --selftest).")
        return 1
    if not os.path.isdir(args.run_dir):
        safe_print(f"ERROR: run-dir {args.run_dir} does not exist.")
        return 1

    plan, sj, prior_L_list = build_plan(args.run_dir)
    L_list, _Lmax, src = build_L_list(5, 5, dry_run=args.dry_run,
                                      fallback_max=max(prior_L_list)
                                      if prior_L_list else 100)
    cs_grid = list(np.round(
        np.arange(args.cs_min, args.cs_max + args.cs_step / 2, args.cs_step), 2))

    if args.dry_run:
        print_plan(plan, L_list, args.n_reps, args.num_reads, cs_grid)
        safe_print("\n--dry-run: no QPU calls made.")
        return 0

    print_plan(plan, L_list, args.n_reps, args.num_reads, cs_grid)

    cooldown_sec = float(os.getenv("INTER_EXPERIMENT_COOLDOWN_SEC", "60"))
    results = {}
    rerunable = [e for e in plan if e["action"] != "skip"]
    for i, entry in enumerate(rerunable):
        if i > 0 and cooldown_sec > 0:
            safe_print(f"\n[cooldown] sleeping {cooldown_sec:.0f}s between "
                       "experiments to avoid rate limit")
            time.sleep(cooldown_sec)
        if entry["action"] == "full_rerun":
            results[entry["key"]] = full_rerun_experiment(
                entry, args.run_dir, args.n_reps, args.num_reads, cs_grid)
        elif entry["action"] == "partial_rerun":
            results[entry["key"]] = partial_rerun_experiment(
                entry, args.run_dir, args.n_reps, args.num_reads, cs_grid)

    sj_path = update_phase1_5_summary(args.run_dir, plan, results)
    safe_print(f"\n[updated] {sj_path}")

    any_fail = any(r.get("status") == "FAIL" for r in results.values())
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
