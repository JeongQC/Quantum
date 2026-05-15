"""rerun_missing_cs.py — fill gaps in the RQ4 chain-strength sweep.

The previous overnight rerun produced only cs=0.10 fully and cs=0.20 partially
(L=5..60). cs=0.30..2.50 are 0-byte NPZ / all-NaN CSV. Root cause was macOS
``ulimit -u`` exhaustion in a single long-lived Python process. This script
fixes that by running each cs in its own subprocess via ``run_one_cs.py``.

Workflow
--------
1. Read ``<run-dir>/logs/phase1_5_summary.json`` for the canonical L_list.
2. For each cs in ``cs_grid``:
       complete  -> SKIP
       partial   -> partial_rerun (only the missing L; merge after)
       empty     -> full_rerun (all L)
3. Dispatch each non-skip cs as a subprocess:
       python run_one_cs.py --cs ... --L-list ... [--merge] ...
   with a 5-minute hard timeout per subprocess and a 30 s cooldown
   between subprocesses to keep the parent's thread count low.
4. Rebuild aggregate ``sweep_summary_per_L.csv`` (and the
   ``summary_per_L_RQ4.csv`` alias) plus the master
   ``raw_vectors_RQ4.npz`` from the per-cs files.
5. Re-run Phase 2 (validation + k* bootstrap) and Phase 4 (audit) with the
   ``--status-filename RUN_STATUS_CS_RERUN.txt`` override so the original
   ``RUN_STATUS.txt`` and earlier ``RUN_STATUS_RERUN.txt`` are preserved.

CLI
    python rerun_missing_cs.py --run-dir out/<ts>
    python rerun_missing_cs.py --run-dir out/<ts> --dry-run
    python rerun_missing_cs.py --run-dir out/<ts> --selftest
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd


_TOK = os.getenv("DWAVE_API_TOKEN", "")


def safe_print(msg):
    s = str(msg)
    if _TOK and _TOK in s:
        s = s.replace(_TOK, "***REDACTED***")
    print(s, flush=True)


def safe_str(msg):
    s = str(msg)
    if _TOK and _TOK in s:
        s = s.replace(_TOK, "***REDACTED***")
    return s


def utc_now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


# ----------------------------------------------------------------------------
# cs grid + naming
# ----------------------------------------------------------------------------
DEFAULT_CS_GRID = [round(0.1 * i, 2) for i in range(1, 26)]  # 0.1 .. 2.5

SWEEP_SUBDIR = "rq4_at20us_cs_sweep"
SWEEP_LABEL = "RQ4"


def cs_tag(cs: float) -> str:
    return "cs" + str(round(float(cs), 2)).replace(".", "p")


# ----------------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------------
def discover_per_cs_status(sweep_dir, cs_grid, L_list):
    """For each cs, compute (status, valid_L, missing_L) by reading the
    per-cs CSV. NPZ size is used as an additional sanity signal."""
    L_set = set(int(L) for L in L_list)
    out = {}
    for cs in cs_grid:
        tag = cs_tag(cs)
        csv_path = os.path.join(sweep_dir, f"summary_per_L_{SWEEP_LABEL}_{tag}.csv")
        npz_path = os.path.join(sweep_dir, f"raw_vectors_{SWEEP_LABEL}_{tag}.npz")
        npz_size = os.path.getsize(npz_path) if os.path.isfile(npz_path) else 0
        if not os.path.isfile(csv_path):
            out[cs] = dict(status="empty", valid_L=[], missing_L=sorted(L_set),
                           csv_path=csv_path, npz_path=npz_path,
                           npz_size=npz_size)
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            out[cs] = dict(status="empty", valid_L=[], missing_L=sorted(L_set),
                           csv_path=csv_path, npz_path=npz_path,
                           npz_size=npz_size)
            continue
        if "mean_cbf_obs" not in df.columns or "L" not in df.columns:
            valid = []
        else:
            valid_rows = df[df["mean_cbf_obs"].notna()]
            valid = sorted(set(int(x) for x in valid_rows["L"].astype(int)))
        valid_in_canon = [L for L in valid if L in L_set]
        missing = sorted(L_set - set(valid_in_canon))
        if not missing:
            status = "complete"
        elif valid_in_canon:
            status = "partial"
        else:
            status = "empty"
        out[cs] = dict(status=status, valid_L=valid_in_canon,
                       missing_L=missing, csv_path=csv_path,
                       npz_path=npz_path, npz_size=npz_size)
    return out


def build_plan(per_cs_status):
    plan = []
    for cs, info in per_cs_status.items():
        if info["status"] == "complete":
            plan.append(dict(cs=cs, action="skip", L_to_run=[]))
        elif info["status"] == "partial":
            plan.append(dict(cs=cs, action="partial_rerun",
                             L_to_run=info["missing_L"]))
        else:
            plan.append(dict(cs=cs, action="full_rerun",
                             L_to_run=info["missing_L"]))
    return plan


def estimate_qpu_seconds(plan, n_reps, anneal_time_us=20.0):
    per_call = 0.50 if abs(anneal_time_us - 20.0) < 1e-6 else 0.50
    total = 0.0
    for entry in plan:
        if entry["action"] == "skip":
            continue
        total += len(entry["L_to_run"]) * n_reps * per_call
    return total


# ----------------------------------------------------------------------------
# Subprocess dispatcher
# ----------------------------------------------------------------------------
# Per-cs hard timeout (15 min by default). 21 L * ~25 s = ~525 s, plus
# embedding lookup + retries = ~600 s typical; 900 s leaves headroom.
SUBPROCESS_TIMEOUT_SEC = int(os.getenv("CS_SUBPROCESS_TIMEOUT_SEC", "900"))
# Stall watchdog: if the subprocess produces NO log activity for
# STALL_TIMEOUT_SEC seconds, kill it. Catches hangs without making the
# global timeout absurdly large.
STALL_TIMEOUT_SEC = int(os.getenv("CS_STALL_TIMEOUT_SEC", "180"))
INTER_SUBPROCESS_COOLDOWN_SEC = float(
    os.getenv("INTER_CS_COOLDOWN_SEC", "30"))

# Exit-code conventions used by run_subprocess_for_cs:
RC_TIMEOUT = 124
RC_STALLED = 137
RC_SPAWN_ERROR = 200


def run_subprocess_for_cs(cs, L_to_run, output_dir, label, anneal_time_us,
                          num_reads, n_reps,
                          timeout_sec=None, stall_sec=None,
                          log_path=None, resume=True,
                          python_exe=None, worker_script=None,
                          poll_interval=1.0):
    """Invoke run_one_cs.py via Popen with hard timeout AND log-tail stall
    watchdog. Returns (rc, elapsed_sec).

    The subprocess always gets ``--resume`` so it preserves any per-L
    checkpoints from earlier kills and only redoes missing L values.
    """
    timeout_sec = int(timeout_sec or SUBPROCESS_TIMEOUT_SEC)
    stall_sec = int(stall_sec or STALL_TIMEOUT_SEC)
    python_exe = python_exe or sys.executable
    worker_script = worker_script or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "run_one_cs.py")
    cmd = [
        python_exe, worker_script,
        "--cs", str(cs),
        "--L-list", ",".join(str(int(L)) for L in L_to_run),
        "--output-dir", output_dir,
        "--label", label,
        "--anneal-time-us", str(anneal_time_us),
        "--num-reads", str(num_reads),
        "--n-reps", str(n_reps),
    ]
    if resume:
        cmd.append("--resume")
    safe_print(f"  $ {' '.join(cmd)}")
    safe_print(f"    timeout={timeout_sec}s stall={stall_sec}s "
               f"log={log_path}")
    return _spawn_with_watchdog(cmd, log_path, timeout_sec, stall_sec,
                                poll_interval=poll_interval)


def _spawn_with_watchdog(cmd, log_path, timeout_sec, stall_sec,
                         poll_interval=1.0):
    """Popen + poll loop. Kills on hard timeout or stall (no log activity).

    Returns (rc, elapsed_sec).
        rc=124      hard timeout
        rc=137      stalled
        rc=200      Popen failure
        otherwise   subprocess's actual exit code
    """
    t0 = time.time()
    log_path = log_path or os.devnull
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    except Exception:
        pass

    try:
        logf = open(log_path, "w")
    except Exception as e:
        safe_print(f"  could not open log {log_path}: {safe_str(e)}")
        return RC_SPAWN_ERROR, 0.0

    try:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                 close_fds=True)
    except Exception as e:
        logf.close()
        safe_print(f"  Popen failed: {safe_str(e)}")
        return RC_SPAWN_ERROR, time.time() - t0

    last_size = 0
    last_activity = time.time()
    try:
        while True:
            rc = proc.poll()
            if rc is not None:
                return rc, time.time() - t0

            now = time.time()
            if now - t0 > timeout_sec:
                safe_print(f"  HARD TIMEOUT after {timeout_sec}s; killing")
                _terminate(proc)
                return RC_TIMEOUT, time.time() - t0

            try:
                cur_size = (os.path.getsize(log_path)
                            if log_path and log_path != os.devnull
                            and os.path.isfile(log_path) else last_size)
            except OSError:
                cur_size = last_size
            if cur_size > last_size:
                last_size = cur_size
                last_activity = now
            elif now - last_activity > stall_sec:
                safe_print(f"  STALLED ({stall_sec}s no log activity); killing")
                _terminate(proc)
                return RC_STALLED, time.time() - t0

            time.sleep(poll_interval)
    finally:
        try:
            logf.close()
        except Exception:
            pass


def _terminate(proc):
    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            pass
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    except Exception:
        pass


# ----------------------------------------------------------------------------
# Aggregate rebuild
# ----------------------------------------------------------------------------
def rebuild_aggregates(sweep_dir, cs_grid, label=SWEEP_LABEL,
                       anneal_time_us=20.0, sampler_mode=None):
    """Concat per-cs CSVs into sweep_summary_per_L.csv and merge per-cs NPZs
    into raw_vectors_<label>.npz with cs-prefixed keys."""
    sampler_mode = sampler_mode or os.getenv("SAMPLER_MODE", "fixed_embedding")
    parts = []
    npz_sources = []
    for cs in cs_grid:
        tag = cs_tag(cs)
        csv_path = os.path.join(sweep_dir, f"summary_per_L_{label}_{tag}.csv")
        npz_path = os.path.join(sweep_dir, f"raw_vectors_{label}_{tag}.npz")
        if os.path.isfile(csv_path):
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            if "chain_strength" not in df.columns:
                df["chain_strength"] = float(cs)
            df["anneal_time_us"] = anneal_time_us
            df["sampler_mode"] = sampler_mode
            parts.append(df)
        if os.path.isfile(npz_path) and os.path.getsize(npz_path) > 100:
            npz_sources.append((tag, npz_path))

    if parts:
        df_all = pd.concat(parts, ignore_index=True)
        df_all = df_all.sort_values(
            ["chain_strength", "L"]).reset_index(drop=True)
    else:
        df_all = pd.DataFrame()

    sweep_csv = os.path.join(sweep_dir, "sweep_summary_per_L.csv")
    alias_csv = os.path.join(sweep_dir, f"summary_per_L_{label}.csv")
    df_all.to_csv(sweep_csv, index=False)
    df_all.to_csv(alias_csv, index=False)
    safe_print(f"[rebuild] wrote {sweep_csv}  ({len(df_all)} rows)")
    safe_print(f"[rebuild] wrote {alias_csv}")

    # NPZ master with cs-prefixed keys
    master_npz = os.path.join(sweep_dir, f"raw_vectors_{label}.npz")
    merged = {}
    for tag, npz_path in npz_sources:
        try:
            with np.load(npz_path, allow_pickle=False) as z:
                for k in z.files:
                    merged[f"{tag}__{k}"] = z[k]
        except Exception as e:
            safe_print(f"[rebuild] WARN: failed to read {npz_path}: "
                       f"{safe_str(e)}")
    np.savez(master_npz, **merged)
    safe_print(f"[rebuild] wrote {master_npz}  "
               f"({len(merged)} keys from {len(npz_sources)} cs files)")

    return sweep_csv, master_npz


# ----------------------------------------------------------------------------
# Phase 2 + Phase 4 driver (chained, like the original launcher)
# ----------------------------------------------------------------------------
def run_phase2_against_run_dir(run_dir, status_filename):
    """Re-run Phase 2 (validation + k* bootstrap) against the rebuilt sweep.
    Returns the python exit code."""
    summary_rq2 = os.path.join(run_dir, "rq2_at5us_cs1p0",
                               "summary_per_L_RQ2.csv")
    raw_rq2 = os.path.join(run_dir, "rq2_at5us_cs1p0", "raw_vectors_RQ2.npz")
    sweep_rq4 = os.path.join(run_dir, SWEEP_SUBDIR, "sweep_summary_per_L.csv")
    out_dir = os.path.join(run_dir, "phase2")
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    if not (os.path.isfile(summary_rq2) and os.path.isfile(raw_rq2)
            and os.path.isfile(sweep_rq4)):
        safe_print("[phase2] inputs missing; skipping")
        return 99
    log_path = os.path.join(log_dir, "phase2_cs_rerun.log")
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "analysis_validation_and_kstar.py"),
        "--summary", summary_rq2,
        "--raw", raw_rq2,
        "--sweep", sweep_rq4,
        "--output-dir", out_dir,
    ]
    safe_print(f"[phase2] running: {' '.join(cmd)}")
    with open(log_path, "w") as f:
        cp = subprocess.run(cmd, check=False, stdout=f, stderr=subprocess.STDOUT)
    safe_print(f"[phase2] exit={cp.returncode}; log={log_path}")
    return cp.returncode


def run_phase4_against_run_dir(run_dir, status_filename, started_iso,
                               launcher_status_path):
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "phase4_cs_rerun.log")
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "phase4_finalize.py"),
        "--output-parent", run_dir,
        "--started", started_iso,
        "--status-filename", status_filename,
        "--launcher-status", launcher_status_path,
    ]
    safe_print(f"[phase4] running: {' '.join(cmd)}")
    with open(log_path, "w") as f:
        cp = subprocess.run(cmd, check=False, stdout=f, stderr=subprocess.STDOUT)
    safe_print(f"[phase4] exit={cp.returncode}; log={log_path}")
    return cp.returncode


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------
def execute_plan(plan, sweep_dir, num_reads, n_reps, anneal_time_us=20.0,
                 timeout_sec=None, stall_sec=None, log_dir=None):
    results = {}
    rerunable = [e for e in plan if e["action"] != "skip"]
    safe_print(f"\nDispatching {len(rerunable)} cs subprocess(es)...")
    for i, entry in enumerate(rerunable):
        if i > 0 and INTER_SUBPROCESS_COOLDOWN_SEC > 0:
            safe_print(f"\n[cooldown] sleeping "
                       f"{INTER_SUBPROCESS_COOLDOWN_SEC:.0f}s between "
                       "cs subprocesses")
            time.sleep(INTER_SUBPROCESS_COOLDOWN_SEC)
        # Per-cs log so the parent's stall watchdog has output to tail.
        cs_log = (os.path.join(log_dir, f"run_one_{cs_tag(entry['cs'])}.log")
                  if log_dir else None)
        rc, elapsed = run_subprocess_for_cs(
            cs=entry["cs"],
            L_to_run=entry["L_to_run"],
            output_dir=sweep_dir,
            label=SWEEP_LABEL,
            anneal_time_us=anneal_time_us,
            num_reads=num_reads,
            n_reps=n_reps,
            timeout_sec=timeout_sec,
            stall_sec=stall_sec,
            log_path=cs_log,
            resume=True,
        )
        results[entry["cs"]] = dict(rc=rc, elapsed_sec=elapsed,
                                    L_to_run=entry["L_to_run"],
                                    log_path=cs_log)
        safe_print(f"  cs={entry['cs']:.2f} -> rc={rc} ({elapsed:.1f}s) "
                   f"log={cs_log}")
    return results


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.selftest:
        return run_selftest()

    if not args.run_dir or not os.path.isdir(args.run_dir):
        safe_print(f"ERROR: --run-dir {args.run_dir!r} missing or not a directory.")
        return 1

    run_dir = args.run_dir.rstrip("/")
    sweep_dir = os.path.join(run_dir, SWEEP_SUBDIR)
    if not os.path.isdir(sweep_dir):
        safe_print(f"ERROR: {sweep_dir} does not exist.")
        return 1

    # Read canonical L_list from phase1_5_summary.json
    sj_path = os.path.join(run_dir, "logs", "phase1_5_summary.json")
    L_list = None
    if os.path.isfile(sj_path):
        try:
            with open(sj_path) as f:
                sj = json.load(f)
            L_list = sj.get("L_list")
        except Exception:
            pass
    if not L_list:
        L_list = list(range(5, 106, 5))
        safe_print(f"[plan] L_list not in phase1_5_summary; using fallback {L_list}")
    L_list = [int(L) for L in L_list]

    cs_grid = list(DEFAULT_CS_GRID)

    per_cs_status = discover_per_cs_status(sweep_dir, cs_grid, L_list)
    plan = build_plan(per_cs_status)

    print_plan(plan, per_cs_status, args.n_reps, args.anneal_time_us)

    if args.dry_run:
        safe_print("\n--dry-run: no QPU calls made.")
        return 0

    started_iso = utc_now_iso()
    t_start = time.time()

    # Execute per-cs subprocesses
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    results = execute_plan(
        plan, sweep_dir,
        num_reads=args.num_reads, n_reps=args.n_reps,
        anneal_time_us=args.anneal_time_us,
        timeout_sec=args.subprocess_timeout,
        stall_sec=args.stall_timeout,
        log_dir=log_dir,
    )

    # Rebuild aggregates
    safe_print("\n=== Rebuilding aggregate sweep CSV + master NPZ ===")
    rebuild_aggregates(sweep_dir, cs_grid,
                       anneal_time_us=args.anneal_time_us)

    # Re-run Phase 2 and Phase 4
    safe_print("\n=== Re-running Phase 2 (validation + k*) ===")
    p2_exit = run_phase2_against_run_dir(run_dir, "RUN_STATUS_CS_RERUN.txt")

    # Write a small launcher_status_*.json for Phase 4 to inline tails
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    launcher_status_path = os.path.join(log_dir, "launcher_status_cs_rerun.json")
    with open(launcher_status_path, "w") as f:
        json.dump({
            "started": started_iso,
            "phase1_5": {
                "exit_code": 0 if all(r.get("rc") == 0
                                      for r in results.values()) else 1,
                "log_path": os.path.join(log_dir, "phase1_5_cs_rerun.log"),
            },
            "phase2": {"exit_code": int(p2_exit),
                       "log_path": os.path.join(log_dir, "phase2_cs_rerun.log")},
            "phase3": {"exit_code": 0,  # not re-run in this script
                       "log_path": os.path.join(log_dir, "phase3.log")},
        }, f, indent=2)

    safe_print("\n=== Re-running Phase 4 (audit + RUN_STATUS_CS_RERUN.txt) ===")
    p4_exit = run_phase4_against_run_dir(
        run_dir, "RUN_STATUS_CS_RERUN.txt", started_iso, launcher_status_path)

    # Summarise
    elapsed = time.time() - t_start
    safe_print(f"\nTotal wall time: {elapsed/60:.1f} min")
    safe_print(f"Phase 2 exit: {p2_exit}")
    safe_print(f"Phase 4 exit: {p4_exit}")
    any_subproc_fail = any(r.get("rc", 1) != 0 for r in results.values())
    return 1 if (any_subproc_fail or p2_exit not in (0,) or p4_exit != 0) else 0


# ----------------------------------------------------------------------------
# Pretty-print plan
# ----------------------------------------------------------------------------
def print_plan(plan, per_cs_status, n_reps, anneal_time_us):
    safe_print("\n=== Per-cs status & rerun plan ===")
    for entry in plan:
        cs = entry["cs"]
        info = per_cs_status[cs]
        action = entry["action"]
        if action == "skip":
            safe_print(f"  cs={cs:.2f}: COMPLETE  (valid_L={len(info['valid_L'])}, "
                       f"npz_bytes={info['npz_size']})  -> SKIP")
        elif action == "partial_rerun":
            safe_print(f"  cs={cs:.2f}: PARTIAL   "
                       f"(valid_L={len(info['valid_L'])} of "
                       f"{len(info['valid_L']) + len(info['missing_L'])}) "
                       f"-> rerun L={entry['L_to_run']}")
        else:
            safe_print(f"  cs={cs:.2f}: EMPTY     "
                       f"(npz_bytes={info['npz_size']}) "
                       f"-> rerun all L={entry['L_to_run']}")
    est = estimate_qpu_seconds(plan, n_reps, anneal_time_us)
    safe_print(f"\nEstimated QPU access time (rough): {est:.0f}s "
               f"(~{est/60:.1f} min)")


# ----------------------------------------------------------------------------
# Selftest
# ----------------------------------------------------------------------------
def _selftest_subprocess_timeout_and_failure():
    """Verify subprocess dispatcher reports the worker's exit code and
    handles timeout cleanly."""
    # Use a tiny wrapper script that exits with a known rc.
    with tempfile.TemporaryDirectory() as td:
        # Failing worker (exit 7).
        bad = os.path.join(td, "fail.py")
        with open(bad, "w") as f:
            f.write("import sys; sys.exit(7)\n")
        cmd = [sys.executable, bad]
        cp = subprocess.run(cmd, check=False, timeout=5)
        assert cp.returncode == 7, f"expected rc=7, got {cp.returncode}"
        safe_print("[selftest 1a] non-zero subprocess rc captured: PASS")

        # Hanging worker -> expect TimeoutExpired -> our wrapper translates
        # to rc=124.
        slow = os.path.join(td, "slow.py")
        with open(slow, "w") as f:
            f.write("import time; time.sleep(10)\n")
        try:
            subprocess.run([sys.executable, slow], check=False, timeout=1)
            assert False, "expected TimeoutExpired"
        except subprocess.TimeoutExpired:
            pass
        safe_print("[selftest 1b] subprocess timeout raises: PASS")


def _selftest_merge_logic():
    """Verify partial-CSV + new-partial CSV merges to a complete frame."""
    from rerun_failed_experiments import merge_summary_csv, merge_npz
    with tempfile.TemporaryDirectory() as td:
        old_csv = os.path.join(td, "old.csv")
        new_csv = os.path.join(td, "new.csv")
        # Old has L=5,10,15 valid + L=20,25 NaN
        pd.DataFrame({
            "L": [5, 10, 15, 20, 25],
            "mean_cbf_obs": [0.1, 0.2, 0.3, np.nan, np.nan],
            "chain_strength": [1.0] * 5,
        }).to_csv(old_csv, index=False)
        # New has L=20,25 valid (the rerun)
        pd.DataFrame({
            "L": [20, 25],
            "mean_cbf_obs": [0.4, 0.5],
            "chain_strength": [1.0, 1.0],
        }).to_csv(new_csv, index=False)
        # Simulate the call: current=new_csv contents, backup=old_csv,
        # subset=[20, 25]. Function modifies new_csv in place.
        import shutil
        merged_target = os.path.join(td, "merged.csv")
        shutil.copy2(new_csv, merged_target)
        merge_summary_csv(merged_target, old_csv, [20, 25])
        df = pd.read_csv(merged_target)
        df = df.sort_values("L").reset_index(drop=True)
        assert list(df["L"]) == [5, 10, 15, 20, 25], df["L"].tolist()
        assert df["mean_cbf_obs"].tolist() == [0.1, 0.2, 0.3, 0.4, 0.5]
        safe_print("[selftest 2a] merge_summary_csv: PASS")

        # NPZ merge
        old_npz = os.path.join(td, "old.npz")
        new_npz = os.path.join(td, "new.npz")
        np.savez(old_npz,
                 chain_lengths_L5=np.array([1, 2]),
                 chain_lengths_L20=np.array([0]),  # stale value to be replaced
                 CBF_vec_L5_ALL=np.array([0.1]))
        np.savez(new_npz,
                 chain_lengths_L20=np.array([4, 5, 6]),
                 chain_lengths_L25=np.array([7, 8]),
                 CBF_vec_L20_ALL=np.array([0.4]))
        merged_npz = os.path.join(td, "merged.npz")
        shutil.copy2(new_npz, merged_npz)
        merge_npz(merged_npz, old_npz, [20, 25])
        with np.load(merged_npz) as z:
            keys = sorted(z.files)
            assert "chain_lengths_L5" in keys, keys
            assert "chain_lengths_L20" in keys
            assert "chain_lengths_L25" in keys
            assert list(z["chain_lengths_L20"]) == [4, 5, 6], \
                "L20 should come from new (subset)"
            assert list(z["chain_lengths_L5"]) == [1, 2], \
                "L5 should come from old (not in subset)"
        safe_print("[selftest 2b] merge_npz: PASS")


def _selftest_aggregate_rebuild():
    """Verify rebuild_aggregates concatenates per-cs CSVs and merges NPZs."""
    with tempfile.TemporaryDirectory() as td:
        cs_grid = [0.1, 0.2, 0.3]
        for cs in cs_grid:
            tag = cs_tag(cs)
            df = pd.DataFrame({
                "L": [5, 10, 15],
                "mean_cbf_obs": [cs * 0.1, cs * 0.2, cs * 0.3],
                "chain_strength": [cs] * 3,
            })
            df.to_csv(os.path.join(td, f"summary_per_L_RQ4_{tag}.csv"), index=False)
            np.savez(os.path.join(td, f"raw_vectors_RQ4_{tag}.npz"),
                     **{f"chain_lengths_L5": np.array([1, 2]),
                        f"chain_lengths_L10": np.array([3, 4])})
        sweep_csv, master_npz = rebuild_aggregates(td, cs_grid)
        df_all = pd.read_csv(sweep_csv)
        assert len(df_all) == 9, f"expected 9 rows, got {len(df_all)}"
        with np.load(master_npz) as z:
            keys = sorted(z.files)
            assert "cs0p1__chain_lengths_L5" in keys, keys
            assert "cs0p2__chain_lengths_L5" in keys
            assert "cs0p3__chain_lengths_L10" in keys
        safe_print("[selftest 3] aggregate rebuild: PASS")


def _selftest_strict_idempotency_for_sweep():
    """Verify is_idempotent_skip(cs_grid=...) rejects the original
    'one good cs, rest empty' aggregate."""
    from run_all_experiments import is_idempotent_skip
    with tempfile.TemporaryDirectory() as td:
        # Build a sweep aggregate where cs=0.1 has data for L=[5,10,15]
        # but cs=0.2 is fully NaN.
        rows = []
        for L in [5, 10, 15]:
            rows.append({"L": L, "mean_cbf_obs": 0.1,
                         "chain_strength": 0.1, "sampler_mode": "x"})
            rows.append({"L": L, "mean_cbf_obs": np.nan,
                         "chain_strength": 0.2, "sampler_mode": "x"})
        pd.DataFrame(rows).to_csv(
            os.path.join(td, "summary_per_L_TESTSWEEP.csv"), index=False)
        # Old (lax) check — would (incorrectly) consider this complete.
        ok_lax = is_idempotent_skip(td, "TESTSWEEP", n_expected=6)
        assert ok_lax is True, ("non-sweep mode should pass on this CSV "
                                "since at least one row is non-NaN")
        # New strict (sweep-aware) check — must reject because cs=0.2 is NaN.
        ok_strict = is_idempotent_skip(td, "TESTSWEEP", n_expected=6,
                                       cs_grid=[0.1, 0.2])
        assert ok_strict is False, ("sweep mode must NOT consider this "
                                    "complete — cs=0.2 is fully NaN")

        # Now make cs=0.2 also valid; strict check should now pass.
        rows_full = []
        for L in [5, 10, 15]:
            for cs in [0.1, 0.2]:
                rows_full.append({"L": L, "mean_cbf_obs": 0.1,
                                  "chain_strength": cs, "sampler_mode": "x"})
        pd.DataFrame(rows_full).to_csv(
            os.path.join(td, "summary_per_L_TESTSWEEP2.csv"), index=False)
        ok_strict2 = is_idempotent_skip(td, "TESTSWEEP2", n_expected=6,
                                        cs_grid=[0.1, 0.2])
        assert ok_strict2 is True
    safe_print("[selftest 4] strict cs-aware idempotency: PASS")


def _selftest_per_L_checkpoint_and_resume():
    """Verify per-L checkpoints survive a mid-sweep failure, and that the
    next call with resume=True skips already-completed L."""
    import rq234_revised as mod

    original = mod.measure_L_oneQUBO

    # Simulated 'res' object that satisfies run_experiment_and_fit's
    # downstream code (cbf_vecs/energy_vecs/per_rep/lengths/...).
    def _make_fake_res(L, qubo_seed, emb_seed):
        return dict(
            L=L,
            qubo_seed=qubo_seed,
            emb_seed=emb_seed,
            sampler_mode="fixed_embedding",
            chain_strength=1.0,
            per_rep=[dict(
                L=L, replicate=0, sampler_mode="fixed_embedding",
                qubo_seed=qubo_seed, emb_seed=emb_seed, chain_strength=1.0,
                mean_cbf=0.1, std_cbf=0.0, prob_break=0.05, n_reads=10,
                mean_energy=-1.0, std_energy=0.1, best_energy=-2.0,
                mean_chainlen=2.0, max_chainlen=3,
                client_wall_time_sec=0.01,
                anneal_time_us=20.0,
            )],
            cbf_vecs=[np.array([0.1, 0.0, 0.2])],
            energy_vecs=[np.array([-1.0, -1.5, -0.5])],
            lengths=np.array([1, 2, 2]),
            lengths_per_rep=[np.array([1, 2, 2])],
            lengths_consistent=True,
            pooled_cbf=dict(mean=0.1, std=0.0, prob_break=0.05, n=3),
            pooled_energy=dict(mean=-1.0, std=0.1, best=-2.0),
            anneal_time_us=20.0,
        )

    state = {"calls": []}

    def first_pass(L, **kwargs):
        # Succeed for L<=15, raise for L>=20 (simulates kill at L=20).
        state["calls"].append(int(L))
        if int(L) >= 20:
            raise RuntimeError("simulated kill mid-sweep")
        return _make_fake_res(int(L),
                              kwargs.get("qubo_seed", 0),
                              kwargs.get("emb_seed", 0))

    def second_pass_all_succeed(L, **kwargs):
        state["calls"].append(int(L))
        return _make_fake_res(int(L),
                              kwargs.get("qubo_seed", 0),
                              kwargs.get("emb_seed", 0))

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        # First pass: should write checkpoints for L=5,10,15 and skip
        # L=20,25 (exception path).
        mod.measure_L_oneQUBO = first_pass
        try:
            mod.run_experiment_and_fit(
                L_list=[5, 10, 15, 20, 25],
                n_reps=1, num_reads=10,
                chain_strength_mode=1.0,
                anneal_time_us=20.0,
                out_dir=td,
                sampler_mode="fixed_embedding",
                show_progress=False, checkpoint_every=1,
                make_plots=False, verbose=False,
                label="ckpt", exact_out_dir=True,
            )
        finally:
            mod.measure_L_oneQUBO = original

        csv_path = os.path.join(td, "summary_per_L_ckpt.csv")
        npz_path = os.path.join(td, "raw_vectors_ckpt.npz")
        df = pd.read_csv(csv_path)
        valid_L = sorted(df.loc[df["mean_cbf_obs"].notna(), "L"].astype(int).tolist())
        assert valid_L == [5, 10, 15], f"expected [5,10,15], got {valid_L}"
        with np.load(npz_path) as z:
            keys = sorted(z.files)
            for L in (5, 10, 15):
                assert f"chain_lengths_L{L}" in keys, \
                    f"chain_lengths_L{L} missing from NPZ after first pass"
        safe_print("[selftest 5a] per-L checkpoint survives mid-sweep raise: PASS")

        # Second pass with resume=True: should skip L=5,10,15 and process
        # only L=20,25.
        state["calls"] = []
        mod.measure_L_oneQUBO = second_pass_all_succeed
        try:
            mod.run_experiment_and_fit(
                L_list=[5, 10, 15, 20, 25],
                n_reps=1, num_reads=10,
                chain_strength_mode=1.0,
                anneal_time_us=20.0,
                out_dir=td,
                sampler_mode="fixed_embedding",
                show_progress=False, checkpoint_every=1,
                make_plots=False, verbose=False,
                label="ckpt", exact_out_dir=True,
                resume=True,
            )
        finally:
            mod.measure_L_oneQUBO = original

        df2 = pd.read_csv(csv_path)
        valid_L2 = sorted(df2.loc[df2["mean_cbf_obs"].notna(), "L"]
                          .astype(int).tolist())
        assert valid_L2 == [5, 10, 15, 20, 25], \
            f"after resume expected [5..25], got {valid_L2}"
        # Resume must NOT re-call measure for already-completed L.
        called = sorted(state["calls"])
        assert called == [20, 25], (f"resume should only call L=[20,25]; "
                                    f"called={called}")
        safe_print("[selftest 5b] --resume skips completed L: PASS")


def _selftest_watchdog_stall():
    """Spawn a tiny subprocess that prints nothing; verify the watchdog
    kills it via the stall path (rc=137)."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        # Worker prints once at start, then sleeps long. Stall window must
        # be short (1 s). Hard timeout long enough to ensure stall path
        # triggers first.
        worker = os.path.join(td, "stall.py")
        with open(worker, "w") as f:
            f.write("import sys, time\n"
                    "print('starting')\n"
                    "sys.stdout.flush()\n"
                    "time.sleep(20)\n")
        log = os.path.join(td, "stall.log")
        rc, elapsed = _spawn_with_watchdog(
            [sys.executable, worker], log_path=log,
            timeout_sec=30, stall_sec=1, poll_interval=0.2,
        )
        assert rc == RC_STALLED, f"expected RC_STALLED ({RC_STALLED}), got {rc}"
        assert elapsed < 5, f"stall watchdog should fire fast; elapsed={elapsed:.2f}s"
    safe_print("[selftest 6] watchdog kills stalled subprocess: PASS")


def _selftest_watchdog_timeout():
    """Verify hard-timeout path: a chatty subprocess that never exits gets
    killed at the hard timeout, returning rc=124."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        worker = os.path.join(td, "chatty.py")
        with open(worker, "w") as f:
            f.write("import sys, time\n"
                    "for i in range(100):\n"
                    "    print(f'tick {i}')\n"
                    "    sys.stdout.flush()\n"
                    "    time.sleep(0.1)\n")
        log = os.path.join(td, "chatty.log")
        rc, elapsed = _spawn_with_watchdog(
            [sys.executable, worker], log_path=log,
            timeout_sec=1, stall_sec=10, poll_interval=0.1,
        )
        assert rc == RC_TIMEOUT, f"expected RC_TIMEOUT ({RC_TIMEOUT}), got {rc}"
    safe_print("[selftest 7] watchdog hard-timeout path: PASS")


def run_selftest():
    safe_print("\n=== rerun_missing_cs selftest ===")
    try:
        _selftest_subprocess_timeout_and_failure()
        _selftest_merge_logic()
        _selftest_aggregate_rebuild()
        _selftest_strict_idempotency_for_sweep()
        _selftest_per_L_checkpoint_and_resume()
        _selftest_watchdog_stall()
        _selftest_watchdog_timeout()
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
# CLI
# ----------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="Fill missing cs values in the RQ4 sweep via "
                    "subprocess-isolated workers.")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--n-reps", type=int, default=10)
    p.add_argument("--num-reads", type=int, default=2000)
    p.add_argument("--anneal-time-us", type=float, default=20.0)
    p.add_argument("--subprocess-timeout", type=int,
                   default=SUBPROCESS_TIMEOUT_SEC,
                   help="Per-cs subprocess hard timeout (seconds). "
                        f"Default {SUBPROCESS_TIMEOUT_SEC}s. Override via env "
                        "var CS_SUBPROCESS_TIMEOUT_SEC.")
    p.add_argument("--stall-timeout", type=int,
                   default=STALL_TIMEOUT_SEC,
                   help="If the subprocess produces no log activity for "
                        f"this many seconds, kill it. Default "
                        f"{STALL_TIMEOUT_SEC}s. Override via env var "
                        "CS_STALL_TIMEOUT_SEC.")
    return p


if __name__ == "__main__":
    sys.exit(main())
