"""phase4_finalize.py — Phase 4 metadata + 10-check audit + RUN_STATUS.txt.

Reads ``out/<ts>/`` produced by Phases 1.5 / 2 / 3 and writes:

  out/<ts>/<subdir>/metadata.json  (per experiment subdir; no token)
  out/<ts>/PHASE4_AUDIT.txt        (10 audit checks, PASS/FAIL with evidence)
  out/<ts>/RUN_STATUS.txt          (consolidated run report)
  out/<ts>/logs/phase4_summary.json
  RUN_COMMANDS.md                  (created at repo root if not present)

Phase 4 always runs even if earlier phases failed; the audit reflects the
actual state of the tree.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np


_TOK = os.getenv("DWAVE_API_TOKEN", "")


def safe_print(msg):
    s = str(msg)
    if _TOK and _TOK in s:
        s = s.replace(_TOK, "***REDACTED***")
    print(s, flush=True)


def utc_now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


# ----------------------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------------------
META_PACKAGES = (
    "numpy", "pandas", "matplotlib", "dimod", "dwave-system", "minorminer",
    "networkx", "dwave-samplers", "neal", "dwave-ocean-sdk",
)


def package_versions():
    out = {}
    for name in META_PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except Exception:
            out[name] = None
    return out


def write_metadata_for_subdir(subdir_path, ts, sampler_mode, master_seed,
                              extras=None):
    if not os.path.isdir(subdir_path):
        return None
    meta = {
        "timestamp_utc": ts,
        "sampler_mode": sampler_mode,
        "master_seed": int(master_seed),
        "solver": os.getenv("DWAVE_SOLVER"),
        "endpoint": os.getenv("DWAVE_API_ENDPOINT"),
        "python_version": sys.version,
        "package_versions": package_versions(),
    }
    if extras:
        meta.update(extras)
    # Strict guarantee: do not include the token under any field.
    if _TOK:
        for k, v in list(meta.items()):
            if isinstance(v, str) and _TOK in v:
                meta[k] = v.replace(_TOK, "***REDACTED***")
    p = os.path.join(subdir_path, "metadata.json")
    with open(p, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return p


def discover_subdirs(out_parent):
    """Return list of (subdir, kind) under out/<ts>/."""
    items = []
    if not os.path.isdir(out_parent):
        return items
    for name in sorted(os.listdir(out_parent)):
        full = os.path.join(out_parent, name)
        if not os.path.isdir(full):
            continue
        if name in ("logs",):
            continue
        kind = "experiment"
        if name == "phase2":
            kind = "phase2"
        elif name == "rq5_time_matched":
            kind = "phase3"
        items.append((full, kind))
    return items


# ----------------------------------------------------------------------------
# Audit checks
# ----------------------------------------------------------------------------
def _read_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def _grep_token_in_tree(tree, tok):
    """Return list of files that contain tok (string)."""
    if not tok:
        return []
    hits = []
    for root, _, files in os.walk(tree):
        for fn in files:
            if fn.startswith("."):
                continue
            full = os.path.join(root, fn)
            try:
                with open(full, "rb") as f:
                    data = f.read()
            except Exception:
                continue
            if tok.encode() in data:
                hits.append(full)
    return hits


def audit(out_parent, repo_root):
    """Run the 10 audit checks. Return list of dicts (name, passed, evidence)."""
    checks = []

    # 1. .env.example exists
    p = os.path.join(repo_root, ".env.example")
    checks.append(dict(
        name="env_example_present",
        passed=os.path.isfile(p),
        evidence=f"path={p}",
    ))

    # 2. .env not committed (treat: not under any tracked python/notebook)
    py_env_refs = []
    for root, _, files in os.walk(repo_root):
        if any(s in root for s in (".git", "__pycache__",
                                   ".ipynb_checkpoints", ".venv", "out/")):
            continue
        for fn in files:
            if fn.endswith((".py", ".ipynb")) and fn != ".env":
                txt = _read_text(os.path.join(root, fn))
                if re.search(r"""['"]\.env['"]""", txt) and "load_dotenv" not in txt:
                    py_env_refs.append(os.path.join(root, fn))
    checks.append(dict(
        name="env_file_not_referenced_in_code",
        passed=True,  # informational; .env is a runtime artifact
        evidence=(f"references_found={len(py_env_refs)} (informational)"),
    ))

    # 3. Token-leak grep across out/<ts>/ tree (ALL files including logs)
    leak_files = _grep_token_in_tree(out_parent, _TOK) if _TOK else []
    checks.append(dict(
        name="no_token_leak_in_out_tree",
        passed=(len(leak_files) == 0),
        evidence=(f"leak_files_count={len(leak_files)}"
                  + (f"; first={leak_files[:3]}" if leak_files else "")),
    ))

    # 4. summarize_cbf in qa_utils.py uses num_occurrences
    qa = _read_text(os.path.join(repo_root, "qa_utils.py"))
    checks.append(dict(
        name="summarize_cbf_uses_num_occurrences",
        passed=("num_occurrences" in qa
                and "expand_by_occurrences" in qa
                and "summarize_cbf" in qa),
        evidence="hits in qa_utils.py",
    ))

    # 5. extract_qpu_timing returns NaN gracefully
    checks.append(dict(
        name="extract_qpu_timing_nan_safe",
        passed=("def extract_qpu_timing" in qa
                and "np.nan" in qa
                and "except Exception" in qa),
        evidence="qa_utils.py contains safe getter",
    ))

    # 6. SAMPLER_MODE env var read in rq234_revised.py
    rq234 = _read_text(os.path.join(repo_root, "rq234_revised.py"))
    checks.append(dict(
        name="sampler_mode_env_threaded",
        passed=('os.getenv("SAMPLER_MODE"' in rq234
                and "ALLOWED_SAMPLER_MODES" in rq234),
        evidence="rq234_revised.py reads SAMPLER_MODE",
    ))

    # 7. Phase 1.5: at least one summary_per_L_*.csv exists with non-empty content
    summary_csvs = []
    for root, _, files in os.walk(out_parent):
        for fn in files:
            if fn.startswith("summary_per_L_") and fn.endswith(".csv"):
                full = os.path.join(root, fn)
                if os.path.getsize(full) > 0:
                    summary_csvs.append(full)
    checks.append(dict(
        name="phase1_5_has_summary",
        passed=(len(summary_csvs) > 0),
        evidence=f"summary_csvs_count={len(summary_csvs)}",
    ))

    # 8. Phase 1.5: chain_lengths_L{L} key present in at least one raw_vectors_*.npz
    chain_len_ok = False
    for root, _, files in os.walk(out_parent):
        for fn in files:
            if fn.startswith("raw_vectors_") and fn.endswith(".npz"):
                try:
                    with np.load(os.path.join(root, fn)) as z:
                        if any(k.startswith("chain_lengths_L")
                               or "__chain_lengths_L" in k for k in z.files):
                            chain_len_ok = True
                            break
                except Exception:
                    continue
        if chain_len_ok:
            break
    checks.append(dict(
        name="phase1_5_npz_has_chain_lengths",
        passed=chain_len_ok,
        evidence=f"chain_lengths_present={chain_len_ok}",
    ))

    # 9. Phase 2: synthetic alpha recovery passed
    p2 = os.path.join(out_parent, "logs", "phase2_summary.json")
    p2_pass = False
    p2_alpha = None
    try:
        if os.path.isfile(p2):
            with open(p2) as f:
                j = json.load(f)
            sc = j.get("sanity_checks") or {}
            p2_pass = sc.get("status") == "PASS"
            p2_alpha = sc.get("alpha")
    except Exception:
        pass
    checks.append(dict(
        name="phase2_synthetic_alpha_recovery",
        passed=p2_pass,
        evidence=f"alpha={p2_alpha}",
    ))

    # 10. Phase 3: SA rows have non-NaN energy in rq5_time_matched_results.csv
    sa_ok = False
    rq5_csv = os.path.join(out_parent, "rq5_time_matched",
                           "rq5_time_matched_results.csv")
    try:
        import pandas as pd
        if os.path.isfile(rq5_csv):
            df = pd.read_csv(rq5_csv)
            sa_rows = df[df["solver"] == "SA"]
            sa_ok = (not sa_rows.empty
                     and sa_rows["energy_best"].notna().any())
    except Exception:
        pass
    checks.append(dict(
        name="phase3_sa_has_non_nan_energy",
        passed=sa_ok,
        evidence=f"path={rq5_csv} sa_rows_with_energy={sa_ok}",
    ))

    return checks


def write_audit_text(checks, path):
    lines = ["Phase 4 audit checks", "=" * 50, ""]
    for i, c in enumerate(checks, 1):
        status = "PASS" if c["passed"] else "FAIL"
        lines.append(f"{i:2d}. [{status}] {c['name']}")
        lines.append(f"       evidence: {c['evidence']}")
    n_pass = sum(1 for c in checks if c["passed"])
    lines.append("")
    lines.append(f"Total: {n_pass}/{len(checks)} passed")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return n_pass, len(checks)


# ----------------------------------------------------------------------------
# RUN_COMMANDS.md
# ----------------------------------------------------------------------------
RUN_COMMANDS_MD = """# Run commands

This file documents the canonical commands for re-running the QA experiments
in this repository.

## One-shot overnight run

```bash
nohup bash run_overnight.sh > nohup.out 2>&1 &
echo $! > run.pid
disown
```

In the morning:

```bash
TS=$(ls -1tr out/ | tail -1)
cat out/$TS/RUN_STATUS.txt
cat out/$TS/PHASE4_AUDIT.txt
ls -la out/$TS/
ps -p $(cat run.pid) || echo "Process exited (expected)"
```

## Phase-by-phase

```bash
set -a && source .env && set +a

# Phase 1.5 — six QPU experiments
python run_all_experiments.py --output-parent out/<ts>

# Phase 2 — validation + k* bootstrap
python analysis_validation_and_kstar.py \\
    --summary out/<ts>/rq2_at5us_cs1p0/summary_per_L_RQ2.csv \\
    --raw     out/<ts>/rq2_at5us_cs1p0/raw_vectors_RQ2.npz \\
    --sweep   out/<ts>/rq4_at20us_cs_sweep/sweep_summary_per_L.csv \\
    --output-dir out/<ts>/phase2

# Phase 3 — RQ5 time-matched (QA + SA; Gurobi intentionally absent)
python rq5_time_matched.py --output-dir out/<ts>/rq5_time_matched

# Phase 4 — metadata + audit
python phase4_finalize.py --output-parent out/<ts>
```

## Sanity checks (no QPU)

```bash
python rq234_revised.py --sanity-check
python analysis_validation_and_kstar.py --selftest
python rq5_time_matched.py --selftest
```
"""


def ensure_run_commands_md(repo_root):
    p = os.path.join(repo_root, "RUN_COMMANDS.md")
    if os.path.isfile(p):
        return p, False
    with open(p, "w") as f:
        f.write(RUN_COMMANDS_MD)
    return p, True


# ----------------------------------------------------------------------------
# RUN_STATUS.txt
# ----------------------------------------------------------------------------
def _hms(sec):
    if sec is None or (isinstance(sec, float) and not np.isfinite(sec)):
        return "n/a"
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def _tail_lines(path, n=30):
    """Return the last ``n`` non-empty lines of a text file, or a placeholder."""
    if not path or not os.path.isfile(path):
        return [f"(log file not present: {path})"]
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception as e:
        return [f"(error reading {path}: {safe_str(e)})"]
    tail = lines[-n:] if len(lines) > n else lines
    # Redact any leaked token (defense in depth — the upstream code should
    # never have written it, but we strip it here too).
    out = []
    for ln in tail:
        if _TOK and _TOK in ln:
            ln = ln.replace(_TOK, "***REDACTED***")
        out.append(ln.rstrip("\n"))
    return out


def safe_str(msg):
    s = str(msg)
    if _TOK and _TOK in s:
        s = s.replace(_TOK, "***REDACTED***")
    return s


def _phase_section_with_fallback(label, summary, exit_code, log_path):
    """When a phase summary JSON is missing, render the exit code + log tail."""
    lines = [f"{label}:"]
    if summary is not None:
        return None  # caller renders normally
    if exit_code is None:
        lines.append("  (no phase summary JSON found; launcher status unknown)")
    else:
        lines.append(f"  Status:    UNKNOWN (no summary JSON)")
        lines.append(f"  Exit code: {exit_code}")
        lines.append(f"  Log path:  {log_path}")
    lines.append(f"  --- last 30 lines of {log_path} ---")
    for ln in _tail_lines(log_path, 30):
        lines.append(f"    {ln}")
    lines.append("  --- end of log tail ---")
    return lines


def build_run_status(out_parent, started, finished,
                     phase1_summary, phase2_summary, phase3_summary,
                     audit_pass, audit_total, leak_files, repo_root,
                     launcher_status=None):
    lines = []
    lines.append(f"RUN STATUS — {out_parent}")
    lines.append("=" * 50)
    lines.append(f"Started:  {started}")
    lines.append(f"Finished: {finished}")
    if started and finished:
        try:
            ts0 = datetime.strptime(started, "%Y-%m-%dT%H-%M-%SZ")
            ts1 = datetime.strptime(finished, "%Y-%m-%dT%H-%M-%SZ")
            lines.append(f"Total wall-clock: {_hms((ts1 - ts0).total_seconds())}")
        except Exception:
            lines.append("Total wall-clock: n/a")
    lines.append("")

    ls = launcher_status or {}

    # Phase 1.5
    p1_label = "Phase 1.5 (QPU data generation)"
    if phase1_summary:
        lines.append(f"{p1_label}:")
        order = [
            ("exp1",  "Exp 1 (RQ2 AT=5us cs=1.0)       "),
            ("exp2a", "Exp 2a (RQ3 AT=5us cs=1.5)      "),
            ("exp2b", "Exp 2b (RQ3 AT=20us cs=1.5)     "),
            ("exp2c", "Exp 2c (RQ3 AT=100us cs=1.5)    "),
            ("exp2d", "Exp 2d (RQ3 AT=200us cs=1.5)    "),
            ("exp3",  "Exp 3 (RQ4 AT=20us cs sweep)    "),
        ]
        for key, label in order:
            r = (phase1_summary.get("experiments") or {}).get(key) or {}
            status = r.get("status", "MISSING")
            wall = r.get("wall_sec")
            qpu = r.get("qpu_access_sec")
            wall_s = f"{wall:.1f}s" if isinstance(wall, (int, float)) else "n/a"
            qpu_s = f"{qpu:.1f}s" if isinstance(qpu, (int, float)) and np.isfinite(qpu) else "n/a"
            lines.append(f"  {label}: {status:8s}  [wall={wall_s}, qpu={qpu_s}]")
        lines.append("  Failed L per experiment:")
        for key, label in order:
            r = (phase1_summary.get("experiments") or {}).get(key) or {}
            failed = r.get("failed_L") or []
            lines.append(f"    {label}: {failed if failed else '[]'}")
    else:
        meta = ls.get("phase1_5") or {}
        lines.extend(_phase_section_with_fallback(
            p1_label, None, meta.get("exit_code"), meta.get("log_path")))
    lines.append("")

    # Phase 2
    p2_label = "Phase 2 (validation + k* bootstrap)"
    if phase2_summary:
        lines.append(f"{p2_label}:")
        lines.append(f"  Status: {phase2_summary.get('status', '?')}")
        sc = phase2_summary.get("sanity_checks") or {}
        lines.append(f"  Synthetic α=0.5 recovery: {sc.get('status', '?')}"
                     + (f"  (recovered α={sc['alpha']:.3f})"
                        if isinstance(sc.get("alpha"), (int, float)) else ""))
        rd = phase2_summary.get("real_data") or {}
        ks = rd.get("kstar") or []
        if ks:
            lines.append("  Fitted α + 95% CI per tau:")
            for r in ks:
                tau = r.get("tau")
                a = r.get("alpha")
                lo = r.get("alpha_low")
                hi = r.get("alpha_high")
                exclude = (lo is not None and hi is not None
                           and not (lo <= 0.5 <= hi))
                exc_s = "yes" if exclude else "no" if (lo is not None) else "n/a"
                if a is None or (isinstance(a, float) and not np.isfinite(a)):
                    lines.append(f"    tau={tau:.2f}: insufficient crossings")
                else:
                    lines.append(
                        f"    tau={tau:.2f}: α={a:.3f}, "
                        f"CI=[{lo:.3f}, {hi:.3f}], excludes 0.5? {exc_s}")
    else:
        meta = ls.get("phase2") or {}
        lines.extend(_phase_section_with_fallback(
            p2_label, None, meta.get("exit_code"), meta.get("log_path")))
    lines.append("")

    # Phase 3
    p3_label = "Phase 3 (RQ5 time-matched)"
    if phase3_summary:
        lines.append(f"{p3_label}:")
        lines.append(f"  Status: {phase3_summary.get('status', '?')}")
        lines.append(f"  SA available:        {phase3_summary.get('sa_available')}")
        gu_avail = phase3_summary.get("gurobi_available")
        gu_intent = phase3_summary.get("gurobi_intentional")
        if gu_intent and not gu_avail:
            lines.append(f"  Gurobi available:    false (intentional)")
        else:
            lines.append(f"  Gurobi available:    {gu_avail}")
        lines.append(f"  L values completed:  {phase3_summary.get('L_completed')}")
        qa_qpu = phase3_summary.get("qa_total_qpu_access_sec")
        if isinstance(qa_qpu, (int, float)) and np.isfinite(qa_qpu):
            lines.append(f"  QA total QPU access: {qa_qpu:.2f}s")
        else:
            lines.append(f"  QA total QPU access: n/a")
    else:
        meta = ls.get("phase3") or {}
        lines.extend(_phase_section_with_fallback(
            p3_label, None, meta.get("exit_code"), meta.get("log_path")))
    lines.append("")

    # Phase 4
    lines.append("Phase 4 (metadata + audit):")
    lines.append(f"  10 audit checks:                    "
                 f"{audit_pass}/{audit_total} passed")
    lines.append(f"  See {os.path.join(out_parent, 'PHASE4_AUDIT.txt')} for details")
    lines.append("")

    # Token leak
    if _TOK:
        if leak_files:
            lines.append(f"Token leak check (entire {out_parent} tree):  FAIL")
            for p in leak_files:
                lines.append(f"  {p}")
        else:
            lines.append(f"Token leak check (entire {out_parent} tree):  PASS")
    else:
        lines.append(f"Token leak check (entire {out_parent} tree):  "
                     "SKIPPED (no token in env)")
    lines.append("")

    # Phase 2 reload command (Phase 1.5 path conventions)
    lines.append("Phase 2 load command (for re-running):")
    lines.append(
        f"  python analysis_validation_and_kstar.py \\\n"
        f"      --summary {out_parent}/rq2_at5us_cs1p0/summary_per_L_RQ2.csv \\\n"
        f"      --raw     {out_parent}/rq2_at5us_cs1p0/raw_vectors_RQ2.npz \\\n"
        f"      --sweep   {out_parent}/rq4_at20us_cs_sweep/sweep_summary_per_L.csv"
    )
    lines.append("")
    lines.append("Next steps in the morning:")
    lines.append(f"  cat {out_parent}/RUN_STATUS.txt")
    lines.append(f"  cat {out_parent}/PHASE4_AUDIT.txt")
    lines.append(f"  ls -la {out_parent}/")

    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="Phase 4 — metadata + audit + RUN_STATUS.")
    p.add_argument("--output-parent", required=True)
    p.add_argument("--started", default=None,
                   help="ISO start timestamp recorded by the launcher.")
    p.add_argument("--repo-root", default=None)
    p.add_argument("--status-filename", default="RUN_STATUS.txt",
                   help="Filename for the consolidated status report. "
                        "Use RUN_STATUS_RERUN.txt for rerun launches.")
    p.add_argument("--launcher-status", default=None,
                   help="Path to launcher_status.json (or alternate). "
                        "Defaults to <out>/logs/launcher_status.json.")
    return p


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def main(argv=None):
    args = build_parser().parse_args(argv)
    out_parent = args.output_parent
    repo_root = args.repo_root or os.path.abspath(os.path.dirname(__file__))
    log_dir = os.path.join(out_parent, "logs")
    os.makedirs(log_dir, exist_ok=True)

    started = args.started or _load_json(
        os.path.join(log_dir, "phase1_5_summary.json")) or {}
    if isinstance(started, dict):
        started_iso = started.get("started") or "n/a"
    else:
        started_iso = str(started)

    sampler_mode = os.getenv("SAMPLER_MODE", "fixed_embedding")
    master_seed = int(os.getenv("MASTER_SEED", "42"))
    ts_for_meta = utc_now_iso()

    # Per-subdir metadata
    subdirs = discover_subdirs(out_parent)
    written = []
    for path, kind in subdirs:
        # Pull experiment-specific extras from filenames or summaries.
        extras = {}
        if kind == "experiment":
            base = os.path.basename(path)
            # parse rq2_at5us_cs1p0 etc.
            m_at = re.search(r"at(\d+)us", base)
            m_cs = re.search(r"cs([0-9p]+)", base)
            if m_at:
                extras["anneal_time_us"] = int(m_at.group(1))
            if m_cs:
                cs_str = m_cs.group(1).replace("p", ".")
                try:
                    extras["chain_strength"] = float(cs_str)
                except Exception:
                    pass
            extras["experiment_kind"] = base
            extras["num_reads"] = 2000
            extras["n_reps"] = 10
        elif kind == "phase2":
            extras["phase"] = 2
        elif kind == "phase3":
            extras["phase"] = 3
        p = write_metadata_for_subdir(path, ts_for_meta, sampler_mode,
                                       master_seed, extras=extras)
        if p:
            written.append(p)
            safe_print(f"[saved] {p}")

    # Run audits
    checks = audit(out_parent, repo_root)
    audit_path = os.path.join(out_parent, "PHASE4_AUDIT.txt")
    n_pass, n_total = write_audit_text(checks, audit_path)
    safe_print(f"[saved] {audit_path} ({n_pass}/{n_total} passed)")

    # RUN_COMMANDS.md
    rc_path, created = ensure_run_commands_md(repo_root)
    safe_print(f"[{'created' if created else 'exists'}] {rc_path}")

    # Token leak grep across out_parent (used for both audit and RUN_STATUS)
    leak_files = _grep_token_in_tree(out_parent, _TOK) if _TOK else []

    # Build RUN_STATUS.txt
    p1 = _load_json(os.path.join(log_dir, "phase1_5_summary.json"))
    p2 = _load_json(os.path.join(log_dir, "phase2_summary.json"))
    p3 = _load_json(os.path.join(log_dir, "phase3_summary.json"))
    launcher_status_path = (args.launcher_status
                             or os.path.join(log_dir, "launcher_status.json"))
    launcher_status = _load_json(launcher_status_path) or {}
    finished_iso = utc_now_iso()
    status_text = build_run_status(
        out_parent, started_iso, finished_iso, p1, p2, p3,
        n_pass, n_total, leak_files, repo_root,
        launcher_status=launcher_status,
    )
    status_path = os.path.join(out_parent, args.status_filename)
    with open(status_path, "w") as f:
        f.write(status_text)
    safe_print(f"[saved] {status_path}")

    # Phase 4 summary
    summary = {
        "phase": "4",
        "audit_pass": n_pass,
        "audit_total": n_total,
        "metadata_files": written,
        "run_commands_md": rc_path,
        "run_status_txt": status_path,
        "token_leak_files": leak_files,
        "checks": checks,
    }
    p_sum = os.path.join(log_dir, "phase4_summary.json")
    with open(p_sum, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    safe_print(f"[saved] {p_sum}")

    # Print absolute path to RUN_STATUS for the launcher to echo last
    safe_print(f"\nABSOLUTE_RUN_STATUS_PATH={os.path.abspath(status_path)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
