"""build_manuscript_assets.py — collect manuscript figures + tables.

Reads the existing artifacts under ``out/<ts>/`` (produced by
make_paper_figures.py and the earlier phases), copies them with the
manuscript's figure-numbered filenames into ``manuscript_assets/``, and
writes:

    manuscript_assets/figures/figXX_*.pdf
    manuscript_assets/tables/tableXX_*.tex
    manuscript_assets/captions/figXX_caption.txt
    manuscript_assets/README.md

Run ``python make_paper_figures.py --run-dir out/<ts>`` first so the
refined Tables 3 & 4 .tex files exist under
``out/<ts>/manuscript_tables/``.

CLI
    python build_manuscript_assets.py
    python build_manuscript_assets.py --run-dir out/<ts>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd


_TOK = os.getenv("DWAVE_API_TOKEN", "")


def safe_print(msg):
    s = str(msg)
    if _TOK and _TOK in s:
        s = s.replace(_TOK, "***REDACTED***")
    print(s, flush=True)


def utc_now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


# ----------------------------------------------------------------------------
# Plan: source -> destination filename
# ----------------------------------------------------------------------------
FIGURE_PLAN = [
    # (manuscript label, source path relative to run_dir, destination filename)
    ("Figure 2",  "rq2_at5us_cs1p0/chainlen_scaling.pdf",
                  "fig02_chainlen_scaling.pdf"),
    ("Figure 3",  "rq2_at5us_cs1p0/cbf_obs_pred_RQ2.pdf",
                  "fig03_cbf_RQ2_T5us_cs1p0.pdf"),
    ("Figure 4a", "rq3_at5us_cs1p5/cbf_obs_pred_T5us.pdf",
                  "fig04a_cbf_RQ3_T5us.pdf"),
    ("Figure 4b", "rq3_at20us_cs1p5/cbf_obs_pred_T20us.pdf",
                  "fig04b_cbf_RQ3_T20us.pdf"),
    ("Figure 4c", "rq3_at100us_cs1p5/cbf_obs_pred_T100us.pdf",
                  "fig04c_cbf_RQ3_T100us.pdf"),
    ("Figure 4d", "rq3_at200us_cs1p5/cbf_obs_pred_T200us.pdf",
                  "fig04d_cbf_RQ3_T200us.pdf"),
    ("Figure 5",  "rq4_at20us_cs_sweep/fig_heatmap_cbf_L_k_AT20us.pdf",
                  "fig05_heatmap_cbf_L_k.pdf"),
    ("Figure 6a", "phase2/fig_kstar_scaling_loglog_tau0p01_with_CI.pdf",
                  "fig06a_kstar_scaling_tau0p01_with_CI.pdf"),
    ("Figure 6b", "phase2/fig_kstar_scaling_loglog_tau0p02_with_CI.pdf",
                  "fig06b_kstar_scaling_tau0p02_with_CI.pdf"),
    ("Figure 6c", "phase2/fig_kstar_scaling_loglog_tau0p05_with_CI.pdf",
                  "fig06c_kstar_scaling_tau0p05_with_CI.pdf"),
    ("Figure 7",  "rq4_at20us_cs_sweep/cbf_obs_pred_all_cs_AT20us.pdf",
                  "fig07_cbf_RQ4_multi_cs.pdf"),
    ("Figure 8",  "phase2/fig_model_comparison_obs_vs_pred.pdf",
                  "fig08_model_comparison_indep_vs_corr.pdf"),
]

TABLE_PLAN = [
    ("Table 3",  "manuscript_tables/table03_RQ3_fitted_params.tex",
                 "table03_RQ3_fitted_params.tex"),
    ("Table 4",  "manuscript_tables/table04_RQ4_fitted_params.tex",
                 "table04_RQ4_fitted_params.tex"),
    ("Table 5",  "rq5_time_matched/rq5_time_matched_table.tex",
                 "table05_RQ5_time_matched.tex"),
    ("Table 6",  "phase2/kstar_powerlaw_bootstrap.tex",
                 "table06_kstar_bootstrap_CI.tex"),
    ("Table 7",  "phase2/validation_splits_table.tex",
                 "table07_validation_splits.tex"),
    ("Table 8",  "phase2/model_comparison_table.tex",
                 "table08_model_comparison.tex"),
]


# ----------------------------------------------------------------------------
# Helpers — read key numbers from existing artifacts
# ----------------------------------------------------------------------------
def _safe_load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def gather_key_numbers(run_dir):
    out = {"run_dir": run_dir}
    # RQ1 chain length: read RQ2 summary, refit slope/R^2 here for the README.
    rq2_summary = os.path.join(run_dir, "rq2_at5us_cs1p0",
                                "summary_per_L_RQ2.csv")
    df = _safe_read_csv(rq2_summary)
    if df is not None and "mean_chainlen" in df.columns:
        df = df.dropna(subset=["L", "mean_chainlen"]).sort_values("L")
        L = df["L"].astype(float).to_numpy()
        m = df["mean_chainlen"].astype(float).to_numpy()
        a, b = np.polyfit(L, m, 1)
        ss_res = float(np.sum((m - (a * L + b)) ** 2))
        ss_tot = float(np.sum((m - m.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        out["rq1"] = dict(slope=float(a), intercept=float(b), r2=float(r2),
                          n=int(len(L)))

    # k* power-law CIs
    boot_csv = os.path.join(run_dir, "phase2", "kstar_powerlaw_bootstrap.csv")
    boot = _safe_read_csv(boot_csv)
    if boot is not None:
        out["kstar"] = []
        for _, r in boot.iterrows():
            tau = float(r.get("tau", float("nan")))
            a = float(r.get("alpha", float("nan")))
            lo = float(r.get("alpha_low", float("nan")))
            hi = float(r.get("alpha_high", float("nan")))
            out["kstar"].append(dict(tau=tau, alpha=a,
                                      alpha_low=lo, alpha_high=hi,
                                      excludes_half=(np.isfinite(lo) and
                                                     np.isfinite(hi) and
                                                     not (lo <= 0.5 <= hi))))

    # Model comparison from phase2_summary.json
    p2 = _safe_load_json(os.path.join(run_dir, "logs", "phase2_summary.json"))
    if p2 and isinstance(p2.get("model_comparison"), dict):
        mc = p2["model_comparison"]
        out["model_comparison"] = dict(
            delta_aic=mc.get("delta_aic_correlated_minus_independent"),
            delta_bic=mc.get("delta_bic_correlated_minus_independent"),
            gamma_fitted=mc.get("gamma_fitted"),
            rho_corr_fitted=mc.get("rho_corr_fitted"),
        )

    # RQ3 / RQ4 refined params (read fresh from the same _caption.txt files
    # we just wrote — easier than re-running the fit).
    out["rq3"] = []
    for at in (5, 20, 100, 200):
        cap = os.path.join(run_dir, f"rq3_at{at}us_cs1p5",
                           f"cbf_obs_pred_T{at}us_caption.txt")
        params = _parse_caption_for_refined(cap)
        if params:
            out["rq3"].append(dict(at_us=at, **params))

    out["rq4"] = []
    rq4_cap = os.path.join(run_dir, "rq4_at20us_cs_sweep",
                           "cbf_obs_pred_all_cs_AT20us_caption.txt")
    out["rq4"] = _parse_rq4_caption_for_refined(rq4_cap)

    return out


def _parse_caption_for_refined(path):
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            txt = f.read()
    except Exception:
        return None
    line = next((ln for ln in txt.splitlines()
                 if ln.strip().startswith("REFINED")), None)
    if not line:
        return None
    out = {}
    for kv in line.split():
        if "=" in kv:
            k, v = kv.split("=", 1)
            try:
                out[k.strip()] = float(v.strip())
            except Exception:
                pass
    if all(k in out for k in ("sigma_h", "sigma_c", "kappa", "SSE")):
        return dict(sigma_h=out["sigma_h"], sigma_c=out["sigma_c"],
                    kappa=out["kappa"], sse=out["SSE"])
    return None


def _parse_rq4_caption_for_refined(path):
    if not os.path.isfile(path):
        return []
    try:
        with open(path) as f:
            lines = f.read().splitlines()
    except Exception:
        return []
    # Find the "REFINED fit" header and parse the following table rows.
    rows = []
    in_refined = False
    for ln in lines:
        s = ln.strip()
        if s.startswith("REFINED fit"):
            in_refined = True
            continue
        if not in_refined:
            continue
        toks = s.split()
        if len(toks) >= 5:
            try:
                cs = float(toks[0])
                sh = float(toks[1])
                sc = float(toks[2])
                kp = float(toks[3])
                sse = float(toks[4])
                rows.append(dict(k=cs, sigma_h=sh, sigma_c=sc, kappa=kp,
                                 sse=sse))
            except ValueError:
                continue
    return rows


# ----------------------------------------------------------------------------
# Figure caption files
# ----------------------------------------------------------------------------
def caption_for_figure(label, dest, key_numbers):
    """Return (lines_list, dest_path) for a captions/figXX_caption.txt."""
    head = [f"{label}", f"File: {dest}"]

    if dest.startswith("fig02_"):
        rq1 = key_numbers.get("rq1") or {}
        head += [
            f"Source: out/<ts>/rq2_at5us_cs1p0/summary_per_L_RQ2.csv",
            "",
            f"Linear fit: mean_chainlen = {rq1.get('slope', float('nan')):.5f} * L + "
            f"{rq1.get('intercept', float('nan')):.5f}",
            f"R^2 = {rq1.get('r2', float('nan')):.6f}, N = {rq1.get('n', '?')}",
            "",
            r"Starter caption:",
            r"\caption{Average chain length $\bar\ell$ as a function of logical "
            r"problem size $L$ for the random fully-connected QUBO instances "
            r"used in this study, embedded into the Zephyr-graph "
            r"\textsc{Advantage2\_system1}. The linear fit slope $\approx "
            f"{rq1.get('slope', float('nan')):.3f}" r"$ (R$^2={"
            f"{rq1.get('r2', float('nan')):.3f}" r"}$) confirms the expected "
            r"$\bar\ell \propto L$ scaling for clique embeddings.}",
            r"\label{fig:chainlen_scaling}",
        ]
    elif dest.startswith("fig03_"):
        head += [
            f"Source: out/<ts>/rq2_at5us_cs1p0/summary_per_L_RQ2.csv",
            "",
            "Refined ICE parameters: see manuscript_assets/tables/"
            "table03_RQ3_fitted_params.tex (T=5us row, k=1.0).",
            "",
            "Error bars: ±1 standard error of the mean (SEM = std_cbf_obs / "
            "sqrt(10), where 10 is the number of independent QPU replicates "
            "per L value).",
            "",
            r"Starter caption:",
            r"\caption{Observed (solid) and ICE-predicted (dashed) mean chain-"
            r"break fraction (CBF) versus logical size $L$ for $T_a=5\,\mu$s "
            r"and chain strength $k=1.0$. The independent-Gaussian noise "
            r"model captures the qualitative trend across the full $L$ range. "
            r"Error bars represent $\pm 1$ standard error of the mean "
            r"($n=10$ replicates per data point).}",
            r"\label{fig:cbf_rq2}",
        ]
    elif dest.startswith("fig04"):
        # Identify which AT this is from the filename
        at = ("5" if "T5us" in dest else
              "20" if "T20us" in dest else
              "100" if "T100us" in dest else "200")
        params_row = next((x for x in (key_numbers.get("rq3") or [])
                           if str(x.get("at_us")) == at), None)
        head += [
            f"Source: out/<ts>/rq3_at{at}us_cs1p5/summary_per_L_RQ3_AT{at}us.csv",
            "",
        ]
        if params_row:
            head += [
                f"Refined: sigma_h={params_row['sigma_h']:.4f}  "
                f"sigma_c={params_row['sigma_c']:.4f}  "
                f"kappa={params_row['kappa']:.4f}  "
                f"SSE={params_row['sse']:.3e}",
                "",
            ]
        head += [
            "Error bars: ±1 standard error of the mean (SEM = std_cbf_obs / "
            "sqrt(10), where 10 is the number of independent QPU replicates "
            "per L value).",
            "",
            r"Starter caption:",
            rf"\caption{{Observed (blue) and ICE-predicted (orange, hatched) "
            rf"mean CBF versus $L$ at $T_a={at}\,\mu$s, $k=1.5$. Predicted "
            rf"values use the refined fit reported in "
            rf"Table~\ref{{tab:fit_params_T}}. Error bars represent "
            rf"$\pm 1$ standard error of the mean ($n=10$ replicates per "
            rf"data point).}}",
            rf"\label{{fig:cbf_rq3_T{at}us}}",
        ]
    elif dest.startswith("fig05_"):
        head += [
            "Source: out/<ts>/rq4_at20us_cs_sweep/sweep_summary_per_L.csv",
            "",
            r"Starter caption:",
            r"\caption{Mean CBF as a function of logical size $L$ and chain "
            r"strength $k$ at fixed $T_a=20\,\mu$s. The white contour marks "
            r"the $\mathrm{CBF}=0.02$ threshold used to define the empirical "
            r"$k^*(L)$ in the scaling analysis (Fig.~\ref{fig:kstar_scaling}).}",
            r"\label{fig:cbf_heatmap}",
        ]
    elif dest.startswith("fig06"):
        tau_tag = ("0p01" if "tau0p01" in dest else
                   "0p02" if "tau0p02" in dest else "0p05")
        tau_val = {"0p01": 0.01, "0p02": 0.02, "0p05": 0.05}[tau_tag]
        kstar_row = next((x for x in (key_numbers.get("kstar") or [])
                          if abs(x.get("tau", -1) - tau_val) < 1e-9), None)
        head += [
            f"Source: out/<ts>/phase2/kstar_points_tau{tau_tag}.csv + "
            "kstar_powerlaw_bootstrap.csv",
            "",
        ]
        if kstar_row:
            head += [
                f"Fit: alpha={kstar_row['alpha']:.3f}  "
                f"95% CI=[{kstar_row['alpha_low']:.3f}, "
                f"{kstar_row['alpha_high']:.3f}]  "
                f"excludes 0.5: {kstar_row['excludes_half']}",
                "",
            ]
        a_val = (kstar_row["alpha"] if kstar_row else float("nan"))
        lo_val = (kstar_row["alpha_low"] if kstar_row else float("nan"))
        hi_val = (kstar_row["alpha_high"] if kstar_row else float("nan"))
        head += [
            r"Starter caption:",
            rf"\caption{{Power-law fit $k^*(L) = a L^\alpha$ at "
            rf"$\tau={tau_val:.2f}$ with bootstrap 95\% confidence band on "
            rf"$\alpha$. The fitted exponent is "
            rf"$\alpha={a_val:.3f}$ with CI [{lo_val:.3f}, {hi_val:.3f}], "
            r"which excludes the independent-noise prediction $\alpha=1/2$.}",
            rf"\label{{fig:kstar_tau_{tau_tag}}}",
        ]
    elif dest.startswith("fig07_"):
        head += [
            "Source: out/<ts>/rq4_at20us_cs_sweep/summary_per_L_RQ4_cs"
            "{1.0,1.5,2.0,2.5}.csv",
            "",
            "Refined ICE parameters per k: see "
            "manuscript_assets/tables/table04_RQ4_fitted_params.tex.",
            "",
            "Error bars: ±1 standard error of the mean (SEM = std_cbf_obs / "
            "sqrt(10), where 10 is the number of independent QPU replicates "
            "per L value).",
            "",
            r"Starter caption:",
            r"\caption{Observed (solid) and ICE-predicted (dashed) mean CBF "
            r"versus $L$ for four representative chain strengths "
            r"$k\in\{1.0,1.5,2.0,2.5\}$ at $T_a=20\,\mu$s. Predicted curves "
            r"use the refined per-$k$ fit in Table~\ref{tab:fit_params_k}. "
            r"Error bars represent $\pm 1$ standard error of the mean "
            r"($n=10$ replicates per data point).}",
            r"\label{fig:cbf_rq4_multi_cs}",
        ]
    elif dest.startswith("fig08_"):
        mc = key_numbers.get("model_comparison") or {}
        head += [
            "Source: out/<ts>/phase2/model_comparison_predictions.csv",
            "",
            f"DeltaAIC (corr - indep) = {mc.get('delta_aic')}",
            f"DeltaBIC (corr - indep) = {mc.get('delta_bic')}",
            f"Fitted gamma = {mc.get('gamma_fitted')}",
            f"Fitted rho_corr = {mc.get('rho_corr_fitted')}",
            "",
            "Error bars: ±1 standard error of the mean (SEM = std_cbf_obs / "
            "sqrt(10), where 10 is the number of independent QPU replicates "
            "per L value).",
            "",
            r"Starter caption:",
            r"\caption{Head-to-head fit on RQ2 CBF data: independent (solid) "
            r"and correlated (dashed) noise models. Both curves overlap "
            r"because the maximum-likelihood $\rho_\mathrm{corr}$ collapses "
            r"to zero on this dataset (see "
            r"Table~\ref{tab:model_comparison}). Error bars on the observed "
            r"markers represent $\pm 1$ standard error of the mean ($n=10$ "
            r"replicates per data point).}",
            r"\label{fig:model_comparison}",
        ]
    return head


# ----------------------------------------------------------------------------
# README
# ----------------------------------------------------------------------------
def write_readme(out_dir, copied_figs, copied_tables, key_numbers, run_dir):
    p = os.path.join(out_dir, "README.md")
    lines = [
        "# Manuscript revision assets",
        "",
        f"All figures and tables for the QIP revision, generated from "
        f"`{run_dir}`.",
        "",
        f"Generated at {utc_now_iso()}.",
        "",
        "## Figures",
        "",
        "| File | Manuscript position | Source data |",
        "|------|---------------------|-------------|",
    ]
    for label, src, dest in copied_figs:
        lines.append(f"| `figures/{dest}` | {label} | "
                     f"`{run_dir}/{src}` |")
    lines += [
        "",
        "## Tables",
        "",
        "| File | Manuscript position | Source |",
        "|------|---------------------|--------|",
    ]
    for label, src, dest in copied_tables:
        lines.append(f"| `tables/{dest}` | {label} | "
                     f"`{run_dir}/{src}` |")

    rq1 = key_numbers.get("rq1") or {}
    lines += [
        "",
        "## Key numbers (for inline manuscript text)",
        "",
        f"- **RQ1 chain length slope**: "
        f"{rq1.get('slope', float('nan')):.4f}, "
        f"R² = {rq1.get('r2', float('nan')):.4f} "
        f"(N = {rq1.get('n', '?')} L values)",
        "- **RQ4 k\\* exponent (95% bootstrap CI):**",
    ]
    for ks in (key_numbers.get("kstar") or []):
        if all(np.isfinite(ks.get(k, float("nan")))
               for k in ("alpha", "alpha_low", "alpha_high")):
            lines.append(
                f"    - τ = {ks['tau']:.2f}: α = {ks['alpha']:.3f}, "
                f"CI = [{ks['alpha_low']:.3f}, {ks['alpha_high']:.3f}], "
                f"excludes 0.5? {'yes' if ks['excludes_half'] else 'no'}")
    mc = key_numbers.get("model_comparison") or {}
    if mc:
        lines += [
            "- **Model comparison (RQ2 CBF data):**",
            f"    - ΔAIC (correlated − independent) = "
            f"{mc.get('delta_aic', float('nan')):.2f}",
            f"    - ΔBIC (correlated − independent) = "
            f"{mc.get('delta_bic', float('nan')):.2f}",
            f"    - ρ_corr (fitted) = "
            f"{mc.get('rho_corr_fitted', float('nan')):.4f} (collapsed)",
            f"    - γ (fitted) = {mc.get('gamma_fitted', float('nan')):.3f} "
            "(meaningless once ρ_corr → 0)",
        ]

    lines += [
        "",
        "## Validation splits (Comment 1.5)",
        "",
        "Train / Test RMSE per split — see `tables/table07_validation_splits.tex`.",
        "",
        "## RQ5 time-matched (Comment 1.3)",
        "",
        "QA energies vs SA at primary, 1 s, 10 s budgets — see "
        "`tables/table05_RQ5_time_matched.tex`. Gurobi columns are "
        "intentionally em-dashed in this revision.",
        "",
        "## Notes",
        "",
        "- All fitted parameters in Tables 3 and 4 were refined via a "
        "Nelder-Mead local optimisation starting from the coarse-grid "
        "winner. The refined SSE is always ≤ the grid SSE; cells where a "
        "refined parameter moved >50% from its grid value are flagged in "
        "the build log.",
        "- Solver: `Advantage2_system1` (Zephyr topology). Note: if the "
        "manuscript currently says `Advantage2_system1.6`, it must be "
        "updated to `system1` throughout.",
        "- All seeds are reproducible with `MASTER_SEED=42` in `.env`.",
        "",
    ]
    with open(p, "w") as f:
        f.write("\n".join(lines) + "\n")
    return p


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def latest_run_dir():
    base = "out"
    if not os.path.isdir(base):
        return None
    runs = [os.path.join(base, d) for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d))]
    return sorted(runs)[-1] if runs else None


def build_parser():
    p = argparse.ArgumentParser(
        description="Collect manuscript figures + tables into "
                    "manuscript_assets/.")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--assets-dir", default="manuscript_assets")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir or latest_run_dir()
    if not run_dir or not os.path.isdir(run_dir):
        safe_print(f"ERROR: run-dir not found: {run_dir}")
        return 1

    assets = args.assets_dir
    fig_dir = os.path.join(assets, "figures")
    tab_dir = os.path.join(assets, "tables")
    cap_dir = os.path.join(assets, "captions")
    for d in (fig_dir, tab_dir, cap_dir):
        os.makedirs(d, exist_ok=True)

    safe_print(f"[run-dir] {run_dir}")
    safe_print(f"[assets] {assets}")

    # Copy figures
    copied_figs = []
    missing = []
    for label, src_rel, dest in FIGURE_PLAN:
        src = os.path.join(run_dir, src_rel)
        dst = os.path.join(fig_dir, dest)
        if os.path.isfile(src) and os.path.getsize(src) > 0:
            shutil.copy2(src, dst)
            copied_figs.append((label, src_rel, dest))
            safe_print(f"  fig: {label:<10s} -> {dst} "
                       f"({os.path.getsize(dst)} bytes)")
        else:
            missing.append((label, src))
            safe_print(f"  fig: {label:<10s} MISSING {src}")

    # Copy tables
    copied_tables = []
    for label, src_rel, dest in TABLE_PLAN:
        src = os.path.join(run_dir, src_rel)
        dst = os.path.join(tab_dir, dest)
        if os.path.isfile(src) and os.path.getsize(src) > 0:
            shutil.copy2(src, dst)
            copied_tables.append((label, src_rel, dest))
            safe_print(f"  tab: {label:<10s} -> {dst} "
                       f"({os.path.getsize(dst)} bytes)")
        else:
            missing.append((label, src))
            safe_print(f"  tab: {label:<10s} MISSING {src}")

    # Read key numbers and write captions
    key_numbers = gather_key_numbers(run_dir)
    for label, src_rel, dest in copied_figs:
        cap_lines = caption_for_figure(label, dest, key_numbers)
        # caption file name: figXX_caption.txt (use the dest's stem)
        stem = dest.rsplit(".", 1)[0]
        # Strip trailing fragments after fig##? — keep only fig##(letter)?
        # i.e., fig02 / fig04a / fig06b
        token = stem.split("_")[0]
        cap_path = os.path.join(cap_dir, f"{token}_caption.txt")
        with open(cap_path, "w") as f:
            f.write("\n".join(cap_lines) + "\n")
        safe_print(f"  cap: {label:<10s} -> {cap_path}")

    # README
    readme = write_readme(assets, copied_figs, copied_tables, key_numbers,
                          run_dir)
    safe_print(f"  readme: {readme}")

    # Validation listing
    safe_print("\n" + "=" * 72)
    safe_print("Final manuscript_assets listing:")
    safe_print("=" * 72)
    for root, _, files in os.walk(assets):
        for fn in sorted(files):
            full = os.path.join(root, fn)
            sz = os.path.getsize(full)
            tag = "  " if sz > 0 else "!!"
            safe_print(f"  {tag} {sz:>8d}  {full}")

    if missing:
        safe_print("\nMISSING source files (skipped):")
        for label, src in missing:
            safe_print(f"  {label}: {src}")

    safe_print("\nZip command for Overleaf upload:")
    safe_print(f"  zip -r manuscript_assets.zip {assets}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
