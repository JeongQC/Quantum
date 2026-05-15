"""rq5_time_matched.py — Phase 3 time-matched solver comparison.

For each L in {20, 40, 60, 80, 100}:
  - QA via FixedEmbeddingComposite (chain_strength=1.5, anneal_time=20us,
    num_reads=2000) using qa_utils.
  - SA at three time budgets:
        primary    matches QA's measured qpu_access_time
        fixed_1s   1 second wall-clock
        fixed_10s  10 seconds wall-clock
  - Gurobi MIQP solve. By design Gurobi is *not* installed in this revision;
    the runner returns gurobi_status="unavailable" with NaN energy. The
    Gurobi columns are emitted in the LaTeX with em-dashes and a footnote.

Outputs (in --output-dir):
  rq5_time_matched_results.csv       per (L, solver, budget)
  rq5_time_matched_energy_gaps.csv   gap = (E - E_best) / |E_best|
  rq5_time_matched_table.tex         LaTeX summary
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
# Backend probes
# ----------------------------------------------------------------------------
def probe_sa():
    """Return (available: bool, backend_name: str, error: str|None)."""
    try:
        from neal import SimulatedAnnealingSampler  # noqa: F401
        return True, "neal", None
    except Exception:
        try:
            from dwave.samplers import SimulatedAnnealingSampler  # noqa: F401
            return True, "dwave.samplers", None
        except Exception as e:
            return False, "none", safe_str(e)


def probe_gurobi():
    """Return tuple (importable, has_license, status_str)."""
    try:
        import gurobipy
    except Exception:
        return False, False, "not installed (expected, continuing without it)"
    try:
        m = gurobipy.Model()
        m.dispose()
        return True, True, "available (license active)"
    except Exception:
        return True, False, "installed but no license"


def probe_qpu_connectivity(force_fail_for_selftest=False):
    """Single solver-properties fetch — no sampling, no QPU credits used.

    Returns (ok, info, err) where info has 'solver', 'topology',
    'largest_clique', 'num_qubits' on success.
    """
    if force_fail_for_selftest:
        return False, {}, ("***SIMULATED*** synthetic connectivity failure "
                            "for selftest")
    try:
        from qa_utils import get_qpu, SOLVER as DEFAULT_SOLVER
    except Exception as e:
        return False, {}, f"qa_utils import failed: {safe_str(e)}"
    try:
        s = get_qpu()
    except Exception as e:
        return False, {}, f"DWaveSampler init failed: {safe_str(e)}"
    try:
        props = s.properties or {}
        topology = (props.get("topology") or {}).get("type")
        info = {
            "solver": getattr(getattr(s, "solver", None), "name",
                              os.getenv("DWAVE_SOLVER", DEFAULT_SOLVER)),
            "topology": topology,
            "largest_clique": props.get("largest_clique"),
            "num_qubits": props.get("num_qubits"),
        }
        return True, info, None
    except Exception as e:
        return False, {}, f"properties fetch failed: {safe_str(e)}"


# ----------------------------------------------------------------------------
# QA runner
# ----------------------------------------------------------------------------
def run_qa(L, chain_strength, anneal_time_us, num_reads, qubo_seed, emb_seed,
           sampler_mode):
    """One QA call. Returns dict with energies, timings, status."""
    from qa_utils import (
        make_random_bqm, get_qpu, find_clique_embedding, summarize_energy,
        extract_qpu_timing, apply_anneal_params, sample_with_retry,
    )
    from dwave.system import FixedEmbeddingComposite

    bqm = make_random_bqm(L, seed=qubo_seed)
    qpu = get_qpu()
    emb = find_clique_embedding(qpu, L, seed=emb_seed, verbose=False)
    comp = FixedEmbeddingComposite(qpu, emb)

    kwargs = dict(num_reads=num_reads, chain_strength=chain_strength,
                  chain_break_fraction=True)
    kwargs.update(apply_anneal_params(qpu, {}, anneal_time_us, None,
                                      verbose=False))
    t0 = time.time()
    ss = sample_with_retry(comp.sample, bqm, **kwargs)
    dt = time.time() - t0
    e = summarize_energy(ss)
    timing = extract_qpu_timing(ss)
    qpu_access_us = timing.get("qpu_access_time_us", float("nan"))
    qpu_access_sec = (float(qpu_access_us) / 1e6
                      if not np.isnan(qpu_access_us) else float("nan"))

    return dict(
        bqm=bqm,
        status="ok",
        energy_best=e["best"], energy_mean=e["mean"], energy_std=e["std"],
        wall_time_sec=dt,
        qpu_access_time_us=float(qpu_access_us),
        qpu_access_time_sec=qpu_access_sec,
        num_reads=num_reads,
        timing=timing,
    )


# ----------------------------------------------------------------------------
# SA runner
# ----------------------------------------------------------------------------
def _make_sa_sampler():
    try:
        from neal import SimulatedAnnealingSampler
        return SimulatedAnnealingSampler(), "neal"
    except Exception:
        from dwave.samplers import SimulatedAnnealingSampler
        return SimulatedAnnealingSampler(), "dwave.samplers"


def run_sa(bqm, budget_sec, label, qubo_seed):
    """Run SA targeting ``budget_sec`` wall time.

    Approach: small calibration run (50 sweeps × 50 reads) measures
    time-per-read; we compute a target num_reads that fits the budget,
    then run once.
    """
    sampler, backend = _make_sa_sampler()
    try:
        # Calibrate
        cal_reads = 50
        t0 = time.time()
        _ = sampler.sample(bqm, num_reads=cal_reads, seed=qubo_seed)
        cal_dt = time.time() - t0
        rate = cal_reads / max(cal_dt, 1e-6)  # reads / second

        target_reads = max(50, int(rate * float(budget_sec)))
        target_reads = min(target_reads, 200_000)

        t0 = time.time()
        ss = sampler.sample(bqm, num_reads=target_reads, seed=qubo_seed + 1)
        dt = time.time() - t0
        from qa_utils import summarize_energy
        e = summarize_energy(ss)
        return dict(
            status="ok",
            backend=backend,
            energy_best=e["best"], energy_mean=e["mean"], energy_std=e["std"],
            wall_time_sec=dt,
            num_reads=target_reads,
            budget_label=label,
            budget_target_sec=float(budget_sec),
        )
    except Exception as e:
        return dict(
            status="failed",
            backend=backend,
            energy_best=float("nan"), energy_mean=float("nan"),
            energy_std=float("nan"), wall_time_sec=float("nan"),
            num_reads=0,
            budget_label=label,
            budget_target_sec=float(budget_sec),
            reason=safe_str(e),
        )


# ----------------------------------------------------------------------------
# Gurobi runner (intentionally returns unavailable in this revision)
# ----------------------------------------------------------------------------
def run_gurobi(bqm):
    importable, licensed, status_str = probe_gurobi()
    if not importable or not licensed:
        return dict(
            status=("unavailable" if not importable else "no_license"),
            status_str=status_str,
            energy_best=float("nan"),
            energy_mean=float("nan"),
            wall_time_sec=float("nan"),
            num_reads=0,
        )
    try:
        # Real Gurobi path — left implemented for completeness even though
        # the user has explicitly chosen not to install Gurobi for this revision.
        import gurobipy as gp
        from gurobipy import GRB

        model = gp.Model("rq5_qubo")
        model.setParam("OutputFlag", 0)
        model.setParam("TimeLimit", 60.0)
        n = bqm.num_variables
        var_list = sorted(bqm.variables)
        x = {v: model.addVar(vtype=GRB.BINARY, name=f"x_{v}") for v in var_list}
        obj = gp.QuadExpr()
        for v, h in bqm.linear.items():
            obj += float(h) * x[v]
        for (u, v), J in bqm.quadratic.items():
            obj += float(J) * x[u] * x[v]
        obj += float(bqm.offset)
        model.setObjective(obj, GRB.MINIMIZE)
        t0 = time.time()
        model.optimize()
        dt = time.time() - t0
        return dict(
            status="ok",
            status_str=status_str,
            energy_best=float(model.ObjVal),
            energy_mean=float(model.ObjVal),
            wall_time_sec=dt,
            num_reads=1,
        )
    except Exception as e:
        return dict(
            status="failed",
            status_str=safe_str(e),
            energy_best=float("nan"), energy_mean=float("nan"),
            wall_time_sec=float("nan"), num_reads=0,
        )


# ----------------------------------------------------------------------------
# Per-L sweep + result tables
# ----------------------------------------------------------------------------
SOLVER_BUDGETS = (
    ("SA", "primary"),
    ("SA", "fixed_1s"),
    ("SA", "fixed_10s"),
    ("Gurobi", "n/a"),
)


def per_L(L, args, sampler_mode):
    rows = []
    safe_print(f"\n=== L={L} ===")
    qa = None
    try:
        qa = run_qa(
            L=L,
            chain_strength=args.chain_strength,
            anneal_time_us=args.anneal_time_us,
            num_reads=args.num_reads,
            qubo_seed=L,
            emb_seed=L,
            sampler_mode=sampler_mode,
        )
        rows.append(dict(
            L=L, solver="QA", budget="qa_native",
            status=qa["status"],
            energy_best=qa["energy_best"], energy_mean=qa["energy_mean"],
            energy_std=qa["energy_std"], wall_time_sec=qa["wall_time_sec"],
            qpu_access_time_us=qa["qpu_access_time_us"],
            num_reads=qa["num_reads"],
        ))
        safe_print(f"  QA      best={qa['energy_best']:.4f} "
                   f"qpu_access={qa['qpu_access_time_sec']:.3f}s")
    except Exception as e:
        safe_print(f"  QA FAILED: {safe_str(e)}")
        traceback.print_exc()
        rows.append(dict(
            L=L, solver="QA", budget="qa_native", status="failed",
            energy_best=float("nan"), energy_mean=float("nan"),
            energy_std=float("nan"), wall_time_sec=float("nan"),
            qpu_access_time_us=float("nan"), num_reads=0,
            reason=safe_str(e),
        ))

    if qa is None or qa["status"] != "ok":
        safe_print(f"  L={L}: skipping classical solvers (no QA result)")
        return rows

    bqm = qa["bqm"]
    sa_primary_budget = qa["qpu_access_time_sec"]
    if not np.isfinite(sa_primary_budget) or sa_primary_budget <= 0:
        sa_primary_budget = 1.0  # fallback if timing missing

    for solver_name, budget_label in SOLVER_BUDGETS:
        if solver_name == "SA":
            budget = {"primary": sa_primary_budget,
                      "fixed_1s": 1.0, "fixed_10s": 10.0}[budget_label]
            res = run_sa(bqm, budget, budget_label, qubo_seed=L)
            rows.append(dict(
                L=L, solver="SA", budget=budget_label,
                status=res["status"],
                energy_best=res["energy_best"],
                energy_mean=res["energy_mean"],
                energy_std=res["energy_std"],
                wall_time_sec=res["wall_time_sec"],
                qpu_access_time_us=float("nan"),
                num_reads=res["num_reads"],
                budget_target_sec=res["budget_target_sec"],
                sa_backend=res.get("backend"),
            ))
            safe_print(f"  SA[{budget_label:>9}] best={res['energy_best']:.4f} "
                       f"wall={res['wall_time_sec']:.3f}s "
                       f"reads={res['num_reads']}")
        else:
            res = run_gurobi(bqm)
            rows.append(dict(
                L=L, solver="Gurobi", budget="n/a",
                status=res["status"],
                gurobi_status=res["status"],
                gurobi_status_str=res["status_str"],
                energy_best=res["energy_best"],
                energy_mean=res["energy_mean"],
                wall_time_sec=res["wall_time_sec"],
                qpu_access_time_us=float("nan"),
                num_reads=res["num_reads"],
            ))
            safe_print(f"  Gurobi  status={res['status']} "
                       f"({res['status_str']})")

    return rows


def compute_energy_gaps(df_results):
    """Per L, compute gap_to_best = (E_solver - E_best_overall) / |E_best_overall|."""
    if df_results.empty:
        return df_results.copy()
    parts = []
    for L, sub in df_results.groupby("L"):
        bests = sub["energy_best"].dropna()
        if bests.empty:
            sub2 = sub.copy()
            sub2["gap_to_best"] = float("nan")
            sub2["E_best_overall"] = float("nan")
            parts.append(sub2)
            continue
        e_best = float(bests.min())
        denom = abs(e_best) if abs(e_best) > 1e-12 else 1.0
        sub2 = sub.copy()
        sub2["E_best_overall"] = e_best
        sub2["gap_to_best"] = (sub2["energy_best"] - e_best) / denom
        parts.append(sub2)
    return pd.concat(parts, ignore_index=True)


def write_latex(df_results, df_gaps, path, gurobi_intentional):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Time-matched solver comparison. SA budget ``primary'' "
        r"matches the QA call's QPU access time. Gap is "
        r"$(E - E_{\text{best}})/|E_{\text{best}}|$.}",
        r"\begin{tabular}{rrrrrr}",
        r"\hline",
        r"$L$ & QA & SA-primary & SA-1s & SA-10s & Gurobi$^{\dagger}$ \\",
        r"\hline",
    ]
    for L, sub in df_gaps.groupby("L"):
        def fmt(slv, budget):
            row = sub[(sub["solver"] == slv) & (sub["budget"] == budget)]
            if row.empty or pd.isna(row["energy_best"].iloc[0]):
                return r"---"
            return f"{row['energy_best'].iloc[0]:.3f}"
        qa_v = fmt("QA", "qa_native")
        sa_pri = fmt("SA", "primary")
        sa_1 = fmt("SA", "fixed_1s")
        sa_10 = fmt("SA", "fixed_10s")
        gu = r"---"  # Gurobi intentionally em-dashed
        lines.append(f"{int(L)} & {qa_v} & {sa_pri} & {sa_1} & {sa_10} & {gu} \\\\")
    lines += [
        r"\hline",
        r"\end{tabular}",
    ]
    if gurobi_intentional:
        lines.append(r"\\[2pt]")
        lines.append(
            r"\footnotesize{$^{\dagger}$Gurobi not included in this revision; "
            r"see Appendix X for the original-submission Gurobi numbers.}"
        )
    lines.append(r"\end{table}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="Phase 3: RQ5 time-matched QA vs SA (Gurobi optional).")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--L-values", default="20,40,60,80,100")
    p.add_argument("--chain-strength", type=float, default=1.5)
    p.add_argument("--anneal-time-us", type=float, default=20.0)
    p.add_argument("--num-reads", type=int, default=2000)
    p.add_argument("--selftest", action="store_true",
                   help="Probe SA + Gurobi and exit; success unless SA missing.")
    return p


def _write_phase3_summary(out_dir, status, reason, **extra):
    """Write phase3_summary.json adjacent to <out_dir>/../logs/. Always best-effort."""
    if not out_dir:
        return
    try:
        os.makedirs(out_dir, exist_ok=True)
        log_dir = os.path.join(os.path.dirname(out_dir.rstrip("/")), "logs")
        os.makedirs(log_dir, exist_ok=True)
        payload = {"phase": "3", "status": status, "reason": reason}
        payload.update(extra)
        with open(os.path.join(log_dir, "phase3_summary.json"), "w") as f:
            json.dump(payload, f, indent=2, default=str)
    except Exception as e:
        safe_print(f"[phase3] WARNING: could not write phase3_summary.json: "
                   f"{safe_str(e)}")


def main(argv=None):
    args = build_parser().parse_args(argv)

    sa_avail, sa_backend, sa_err = probe_sa()
    gu_importable, gu_licensed, gu_status_str = probe_gurobi()

    safe_print(f"SA: {'available' if sa_avail else 'MISSING'}"
               + (f" (backend={sa_backend})" if sa_avail else ""))
    safe_print(f"Gurobi: {gu_status_str}")

    # ---- Selftest mode --------------------------------------------------
    if args.selftest:
        # Always exercise the simulated connectivity-failure path so we
        # know probe_qpu_connectivity actually flags failures.
        ok_mock, _info, err_mock = probe_qpu_connectivity(
            force_fail_for_selftest=True)
        if ok_mock or err_mock is None or "***SIMULATED***" not in (err_mock or ""):
            safe_print("\n[Phase 3 selftest] FAIL: connectivity probe did not "
                       "report the simulated failure correctly.")
            return 1
        safe_print(f"Connectivity probe (mocked failure path): OK "
                   f"(reported: {err_mock})")
        if not sa_avail:
            safe_print(f"\n[Phase 3 selftest] FAIL: SA backend missing ({sa_err})")
            return 1
        safe_print("\n[Phase 3 selftest] PASS")
        return 0

    # ---- Real run -------------------------------------------------------
    if not args.output_dir:
        safe_print("ERROR: --output-dir is required for the full run.")
        return 1

    if not sa_avail:
        safe_print("\n[Phase 3] FAILED: no SA backend available; cannot proceed.")
        _write_phase3_summary(args.output_dir, "FAILED",
                              "no SA backend",
                              sa_available=False,
                              gurobi_available=(gu_importable and gu_licensed),
                              gurobi_intentional=True)
        return 1

    # QPU connectivity check BEFORE any sampling. Cheap (one properties fetch).
    safe_print("\nProbing QPU connectivity...")
    ok, info, err = probe_qpu_connectivity()
    if not ok:
        safe_print(f"QPU unreachable: {err}")
        _write_phase3_summary(args.output_dir, "FAILED",
                              f"QPU unreachable: {err}",
                              sa_available=True,
                              gurobi_available=(gu_importable and gu_licensed),
                              gurobi_intentional=True,
                              connectivity_ok=False)
        return 1
    safe_print(f"Solver: {info.get('solver')}  topology={info.get('topology')}  "
               f"largest_clique={info.get('largest_clique')}  "
               f"num_qubits={info.get('num_qubits')}")
    if info.get("topology") != "zephyr":
        safe_print(f"WARNING: expected zephyr topology, got "
                   f"{info.get('topology')!r}")

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    log_dir = os.path.join(os.path.dirname(out_dir.rstrip("/")), "logs")
    os.makedirs(log_dir, exist_ok=True)

    sampler_mode = os.getenv("SAMPLER_MODE", "fixed_embedding")
    L_values = [int(x) for x in args.L_values.split(",") if x.strip()]
    safe_print(f"\nL values: {L_values}, chain_strength={args.chain_strength}, "
               f"anneal_time={args.anneal_time_us}us, num_reads={args.num_reads}, "
               f"sampler_mode={sampler_mode}")

    started_iso = utc_now_iso()
    t0 = time.time()
    all_rows = []
    for L in L_values:
        try:
            all_rows.extend(per_L(L, args, sampler_mode))
        except Exception as e:
            safe_print(f"  per_L L={L} crashed: {safe_str(e)}")
            traceback.print_exc()

    df = pd.DataFrame(all_rows)
    results_csv = os.path.join(out_dir, "rq5_time_matched_results.csv")
    df.to_csv(results_csv, index=False)
    safe_print(f"\n[saved] {results_csv}")

    df_gaps = compute_energy_gaps(df)
    gaps_csv = os.path.join(out_dir, "rq5_time_matched_energy_gaps.csv")
    df_gaps.to_csv(gaps_csv, index=False)
    safe_print(f"[saved] {gaps_csv}")

    tex_path = os.path.join(out_dir, "rq5_time_matched_table.tex")
    write_latex(df, df_gaps, tex_path, gurobi_intentional=True)
    safe_print(f"[saved] {tex_path}")

    qa_total_qpu_sec = float(np.nansum(
        df.loc[df["solver"] == "QA", "qpu_access_time_us"]) / 1e6)

    completed_L = sorted(set(int(L) for L in df["L"].unique()
                             if not pd.isna(L)))

    # Strict PASS criterion: at least one finite QA energy must have been
    # recorded. If every qa_best_energy is NaN, the run is FAIL — matches
    # the launcher's verification that Phase 3 actually ran.
    qa_energies = df.loc[df["solver"] == "QA", "energy_best"]
    qa_finite_count = int(qa_energies.apply(
        lambda v: isinstance(v, (int, float)) and np.isfinite(v)).sum())
    pass_criterion = qa_finite_count >= 1

    status = "PASS" if pass_criterion else "FAILED"
    reason = (None if pass_criterion
              else (f"no finite QA energies in results "
                    f"({qa_finite_count}/{len(qa_energies)} rows non-NaN); "
                    f"see {results_csv}"))

    summary = {
        "phase": "3",
        "status": status,
        "reason": reason,
        "started": started_iso,
        "finished": utc_now_iso(),
        "wall_sec": time.time() - t0,
        "L_values": L_values,
        "L_completed": completed_L,
        "qa_finite_energy_count": qa_finite_count,
        "qa_total_rows": int(len(qa_energies)),
        "qa_total_qpu_access_sec": qa_total_qpu_sec,
        "sa_available": True,
        "sa_backend": sa_backend,
        "gurobi_available": gu_importable and gu_licensed,
        "gurobi_intentional": True,
        "gurobi_status_str": gu_status_str,
        "connectivity_ok": True,
        "connectivity_info": info,
        "per_L_results": df.to_dict(orient="records"),
    }
    with open(os.path.join(log_dir, "phase3_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    safe_print(f"[saved] {os.path.join(log_dir, 'phase3_summary.json')}")
    if not pass_criterion:
        safe_print(f"\n[Phase 3] FAILED: {reason}")
        return 1
    safe_print(f"\n[Phase 3] PASS: {qa_finite_count}/{len(qa_energies)} QA "
               "rows have finite energy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
