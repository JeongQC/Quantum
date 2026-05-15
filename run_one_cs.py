"""run_one_cs.py — single chain-strength worker (subprocess-friendly).

Runs ONE cs value of the RQ4 chain-strength sweep over a specified L list
and writes ``summary_per_L_<label>_cs<X>p<Y>.csv`` and
``raw_vectors_<label>_cs<X>p<Y>.npz`` into ``--output-dir``.

This script is invoked as a subprocess by ``rerun_missing_cs.py`` so that
each cs value runs in a fresh Python process. Threads/sockets accumulated
during one cs are released when the subprocess exits, sidestepping the
macOS ``ulimit -u 8000`` that broke the previous in-process sweep.

CLI
    python run_one_cs.py \\
        --cs 1.0 \\
        --L-list 5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105 \\
        --output-dir out/<ts>/rq4_at20us_cs_sweep \\
        --label RQ4 \\
        --anneal-time-us 20 \\
        --num-reads 2000 \\
        --n-reps 10 \\
        [--merge]

Exit code: 0 on success, non-zero on failure.

``--merge``: if set, back up the existing per-cs CSV / NPZ for this cs value
to ``*.pre_rerun`` siblings, run for L-list, then merge the new rows in
(keeping the existing rows for L not in --L-list). Used when topping up a
partially-completed cs.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import traceback


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


def cs_tag(cs: float) -> str:
    return "cs" + str(round(float(cs), 2)).replace(".", "p")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="One-cs RQ4 worker.")
    p.add_argument("--cs", type=float, required=True)
    p.add_argument("--L-list", required=True,
                   help="Comma-separated L values, e.g. '5,10,15'")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--label", default="RQ4",
                   help="Sweep-level label, e.g. 'RQ4'. Per-cs label is "
                        "<label>_cs<X>p<Y>.")
    p.add_argument("--anneal-time-us", type=float, default=20.0)
    p.add_argument("--num-reads", type=int, default=2000)
    p.add_argument("--n-reps", type=int, default=10)
    p.add_argument("--merge", action="store_true",
                   help="Merge new rows with existing per-cs CSV/NPZ "
                        "(legacy; prefer --resume).")
    p.add_argument("--resume", action="store_true",
                   help="If existing per-cs CSV/NPZ already have rows for "
                        "some L with non-NaN data, skip those L. Per-L "
                        "checkpoints are written atomically after each L "
                        "so a SIGKILL preserves all completed L data.")
    p.add_argument("--qubo-seed-base", type=int, default=0)
    p.add_argument("--emb-seed-base", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    L_list = [int(x) for x in args.L_list.split(",") if x.strip()]
    if not L_list:
        safe_print("ERROR: --L-list parsed to empty.")
        return 1

    sub_label = f"{args.label}_{cs_tag(args.cs)}"
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"summary_per_L_{sub_label}.csv")
    npz_path = os.path.join(out_dir, f"raw_vectors_{sub_label}.npz")
    backup_csv = csv_path + ".pre_rerun"
    backup_npz = npz_path + ".pre_rerun"

    safe_print(f"[run_one_cs] cs={args.cs} label={sub_label} "
               f"L_list={L_list} merge={args.merge}")

    if args.merge:
        if os.path.isfile(csv_path):
            shutil.copy2(csv_path, backup_csv)
            safe_print(f"  backed up CSV -> {backup_csv}")
        if os.path.isfile(npz_path):
            shutil.copy2(npz_path, backup_npz)
            safe_print(f"  backed up NPZ -> {backup_npz}")

    sampler_mode = os.getenv("SAMPLER_MODE", "fixed_embedding")

    # Import here so --help doesn't drag in dwave-system.
    from rq234_revised import run_experiment_and_fit

    try:
        run_experiment_and_fit(
            L_list=L_list,
            n_reps=args.n_reps,
            num_reads=args.num_reads,
            qubo_seed_base=args.qubo_seed_base,
            emb_seed_base=args.emb_seed_base,
            chain_strength_mode=float(args.cs),
            anneal_time_us=args.anneal_time_us,
            anneal_schedule=None,
            out_dir=out_dir,
            sampler_mode=sampler_mode,
            show_progress=False,
            checkpoint_every=1,
            make_plots=False,
            verbose=True,
            label=sub_label,
            exact_out_dir=True,
            resume=args.resume,
        )
    except Exception as e:
        safe_print(f"[run_one_cs] runner raised: {safe_str(e)}")
        traceback.print_exc()
        return 2

    # Merge with backup (if any) so partial reruns combine cleanly.
    if args.merge and os.path.isfile(backup_csv):
        try:
            from rerun_failed_experiments import merge_summary_csv, merge_npz
            merge_summary_csv(csv_path, backup_csv, L_list)
            if os.path.isfile(backup_npz):
                merge_npz(npz_path, backup_npz, L_list)
        except Exception as e:
            safe_print(f"[run_one_cs] WARN: merge failed: {safe_str(e)}; "
                       "leaving new files intact and .pre_rerun siblings")
            # Don't fail the worker — caller can inspect.

    # Sanity: at least one row of the produced CSV should have non-NaN
    # mean_cbf_obs. If not, exit non-zero so the caller knows.
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        if "mean_cbf_obs" not in df.columns or df["mean_cbf_obs"].isna().all():
            safe_print(f"[run_one_cs] FAIL: every L is NaN in {csv_path}")
            return 3
    except Exception as e:
        safe_print(f"[run_one_cs] FAIL: cannot read {csv_path}: {safe_str(e)}")
        return 4

    safe_print(f"[run_one_cs] cs={args.cs} done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
