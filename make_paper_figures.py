"""make_paper_figures.py — regenerate manuscript figures from out/<ts>/.

Produces:
    Figure 2  RQ1 chain-length scaling
    Figure 4  RQ3 obs vs predicted CBF, one panel per AT (T=5,20,100,200 us)
    Figure 5  RQ4 heatmap CBF(L, k)
    Figure 7  RQ4 obs vs predicted CBF for cs in {1.0, 1.5, 2.0, 2.5}

A small caption ``.txt`` is written next to each figure with the fitted
ICE parameters (sigma_h, sigma_c, kappa) so they're easy to drop into the
manuscript.

No QPU calls — the script only reads existing CSV / NPZ artifacts under
out/<ts>/. The independent-noise ICE fit is reused from rq234_revised.

CLI
    python make_paper_figures.py --dry-run
    python make_paper_figures.py
    python make_paper_figures.py --run-dir out/<ts>
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

from rq234_revised import (
    cbf_from_lengths, fit_sigma_params, refine_sigma_params,
)


_TOK = os.getenv("DWAVE_API_TOKEN", "")


def safe_print(msg):
    s = str(msg)
    if _TOK and _TOK in s:
        s = s.replace(_TOK, "***REDACTED***")
    print(s, flush=True)


# ----------------------------------------------------------------------------
# Manuscript-matching style
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

FIGSIZE = (5.0, 3.5)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _cs_tag(cs: float) -> str:
    return "cs" + str(round(float(cs), 2)).replace(".", "p")


def load_summary_and_chains(summary_csv, raw_npz):
    """Return (L_list, cbf_obs_list, chain_lengths_dict, std_obs_list).

    ``std_obs_list`` parallels ``cbf_obs_list``; entries are NaN when the
    column is missing. The caller computes SEM from std as needed.
    Drops L rows missing chain_lengths_L<L> in the NPZ or NaN
    mean_cbf_obs in the CSV.
    """
    df = pd.read_csv(summary_csv)
    df = df.dropna(subset=["L", "mean_cbf_obs"]).copy()
    df = df.sort_values("L").reset_index(drop=True)
    chain_lengths_dict = {}
    with np.load(raw_npz, allow_pickle=False) as z:
        for L in df["L"].astype(int).unique():
            key = f"chain_lengths_L{int(L)}"
            if key in z.files:
                chain_lengths_dict[int(L)] = np.asarray(z[key], dtype=int)
    df = df[df["L"].astype(int).isin(chain_lengths_dict.keys())].reset_index(
        drop=True)
    if "std_cbf_obs" in df.columns:
        std_obs = df["std_cbf_obs"].astype(float).to_numpy()
    else:
        std_obs = np.full(len(df), np.nan, dtype=float)
    return (df["L"].astype(int).tolist(),
            df["mean_cbf_obs"].astype(float).tolist(),
            chain_lengths_dict,
            std_obs.tolist())


# Constants for SEM
N_REPS = 10  # independent QPU replicates per L value (run config)


def _sem_array(std_obs_list, n_reps=N_REPS):
    """Convert std_cbf_obs list to SEM = std / sqrt(n_reps).

    Returns ``None`` if every entry is NaN/missing (suppresses error bars)."""
    arr = np.asarray(std_obs_list, dtype=float)
    if not np.any(np.isfinite(arr)):
        return None
    sem = arr / np.sqrt(float(n_reps))
    sem[~np.isfinite(sem)] = 0.0
    return sem


# Common errorbar style for observed CBF.
ERRORBAR_KW = dict(capsize=3, ecolor="black", elinewidth=0.8)
# Coarser tick scheme for L-axis figures.
L_TICKS = [0, 20, 40, 60, 80, 100]
L_TICKS_HEATMAP = [20, 40, 60, 80, 100]


def fit_ice_for_dataset(L_list, cbf_obs, chain_lengths_dict, verbose=False):
    """Coarse grid fit followed by Nelder-Mead local refinement.

    Returns dict with both grid and refined parameter sets, plus the
    refined predictions per L (used for figures and tables).
    """
    calib = [
        {"L": int(L), "cbf_obs": float(c),
         "lengths": chain_lengths_dict[int(L)]}
        for L, c in zip(L_list, cbf_obs) if int(L) in chain_lengths_dict
    ]
    grid = fit_sigma_params(calib, verbose=verbose)

    refined = refine_sigma_params(
        L_list=L_list,
        cbf_obs_list=cbf_obs,
        chain_lengths_dict=chain_lengths_dict,
        sh0=grid["sigma_h"], sc0=grid["sigma_c"], kappa0=grid["kappa"],
        verbose=verbose,
    )

    # Sanity: refined SSE must not exceed grid SSE (the grid winner is a
    # valid starting point, so Nelder-Mead can never make it worse — if it
    # does, that's a bug in the loss / clipping).
    if refined["sse"] > grid["sse"] + 1e-9:
        raise RuntimeError(
            f"refined SSE ({refined['sse']:.6e}) > grid SSE "
            f"({grid['sse']:.6e}) — refinement bug")

    preds = {}
    for L in L_list:
        if int(L) in chain_lengths_dict:
            preds[int(L)] = float(cbf_from_lengths(
                chain_lengths_dict[int(L)],
                refined["sigma_h"], refined["sigma_c"], refined["kappa"]))
    return dict(
        grid=grid,
        refined=refined,
        # convenience aliases (refined values surfaced as the canonical fit)
        sigma_h=refined["sigma_h"],
        sigma_c=refined["sigma_c"],
        kappa=refined["kappa"],
        sse=refined["sse"],
        predictions=preds,
    )


def _moved_more_than(grid, refined, frac=0.5):
    """Return list of param names whose refined value differs from grid by
    more than ``frac`` relative change."""
    moved = []
    for k in ("sigma_h", "sigma_c", "kappa"):
        g = float(grid[k])
        r = float(refined[k])
        if g <= 0:
            continue
        if abs(r - g) / abs(g) > frac:
            moved.append(k)
    return moved


def write_caption_txt(path, lines):
    with open(path, "w") as f:
        for ln in lines:
            f.write(ln + "\n")


# ----------------------------------------------------------------------------
# Figure 2 — RQ1 chain-length scaling
# ----------------------------------------------------------------------------
def build_rq1_chainlen_figure(run_dir):
    summary_csv = os.path.join(run_dir, "rq2_at5us_cs1p0",
                                "summary_per_L_RQ2.csv")
    out_pdf = os.path.join(run_dir, "rq2_at5us_cs1p0",
                            "chainlen_scaling.pdf")
    out_txt = out_pdf.replace(".pdf", "_caption.txt")
    if not os.path.isfile(summary_csv):
        safe_print(f"[skip] Figure 2: {summary_csv} not found")
        return None
    df = pd.read_csv(summary_csv)
    df = df.dropna(subset=["L", "mean_chainlen"]).sort_values("L").reset_index(
        drop=True)
    L = df["L"].astype(int).to_numpy()
    mcl = df["mean_chainlen"].astype(float).to_numpy()
    max_cl = (df["max_chainlen"].astype(float).to_numpy()
              if "max_chainlen" in df.columns else None)

    # Linear fit y = a*L + b
    a, b = np.polyfit(L.astype(float), mcl, 1)
    fit_y = a * L + b
    ss_res = float(np.sum((mcl - fit_y) ** 2))
    ss_tot = float(np.sum((mcl - mcl.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.gca()
    ax.scatter(L, mcl, marker="o", s=36, color="#1f77b4",
               label="mean chain length", zorder=3)
    if max_cl is not None:
        ax.scatter(L, max_cl, marker="^", s=24, color="#d62728",
                   alpha=0.6, label="max chain length", zorder=2)
    Lg = np.linspace(L.min(), L.max(), 100)
    ax.plot(Lg, a * Lg + b, "-", color="#1f77b4", linewidth=1.4,
            label=fr"linear fit: slope={a:.3f}, $R^2$={r2:.4f}")
    ax.set_xlabel(r"$L$ (logical variables)")
    ax.set_ylabel("chain length")
    ax.set_title("Chain length vs logical problem size")
    ax.set_xticks(L_TICKS)
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="upper left", framealpha=0.9)
    fig.savefig(out_pdf)
    plt.close(fig)

    write_caption_txt(out_txt, [
        f"Figure 2 — RQ1 chain-length scaling",
        f"Data: {summary_csv}",
        f"N = {len(L)} L values: {L.tolist()}",
        f"Linear fit: mean_chainlen = {a:.5f} * L + {b:.5f}",
        f"R^2 = {r2:.6f}",
        f"max_chainlen plotted as triangles where present.",
    ])
    return dict(path=out_pdf, slope=a, intercept=b, r2=r2, n=len(L))


# ----------------------------------------------------------------------------
# Figure 4 — RQ3 obs vs predicted CBF, one panel per AT
# ----------------------------------------------------------------------------
def build_rq3_panel(run_dir, at_us):
    sub = f"rq3_at{int(at_us)}us_cs1p5"
    summary_csv = os.path.join(run_dir, sub,
                                f"summary_per_L_RQ3_AT{int(at_us)}us.csv")
    raw_npz = os.path.join(run_dir, sub,
                            f"raw_vectors_RQ3_AT{int(at_us)}us.npz")
    out_pdf = os.path.join(run_dir, sub, f"cbf_obs_pred_T{int(at_us)}us.pdf")
    out_txt = out_pdf.replace(".pdf", "_caption.txt")
    if not (os.path.isfile(summary_csv) and os.path.isfile(raw_npz)):
        safe_print(f"[skip] RQ3 AT={at_us}us: inputs missing "
                   f"({summary_csv}, {raw_npz})")
        return None

    L_list, cbf_obs, chain_lengths_dict, std_obs = load_summary_and_chains(
        summary_csv, raw_npz)
    if not L_list:
        safe_print(f"[skip] RQ3 AT={at_us}us: no usable rows")
        return None

    fit = fit_ice_for_dataset(L_list, cbf_obs, chain_lengths_dict,
                               verbose=False)
    cbf_pred = [fit["predictions"][int(L)] for L in L_list]
    sem = _sem_array(std_obs)

    # Paired-bar chart with bars positioned at the actual L value so the
    # x-axis tick scheme aligns with the line plots. Bars are 2-wide and
    # paired around L (-1, +1) so they don't overlap (L grid step is 5).
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.gca()
    L_arr = np.asarray(L_list, dtype=float)
    width = 2.0
    ax.bar(L_arr - width / 2, cbf_obs, width=width,
           color="#1f77b4", edgecolor="black", linewidth=0.5,
           yerr=sem, error_kw=dict(**ERRORBAR_KW),
           label="Observed")
    ax.bar(L_arr + width / 2, cbf_pred, width=width,
           facecolor="#ff7f0e", edgecolor="black", linewidth=0.5,
           hatch="///", label="Predicted (ICE)")
    ax.set_xlim(min(L_arr.min() - 5, 0), L_arr.max() + 5)
    ax.set_xticks(L_TICKS)
    ax.set_xlabel(r"$L$")
    ax.set_ylabel("mean CBF")
    ax.set_title(rf"RQ3: $T_a = {int(at_us)}\,\mu$s,  $k = 1.5$")
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    ax.legend(loc="upper left", framealpha=0.9)
    fig.savefig(out_pdf)
    plt.close(fig)

    g, r = fit["grid"], fit["refined"]
    write_caption_txt(out_txt, [
        f"Figure 4 panel — RQ3 AT={int(at_us)}us, k=1.5",
        f"Data: {summary_csv}",
        f"N = {len(L_list)} L values",
        "",
        "Fitted ICE parameters (independent Gaussian model):",
        f"  GRID   : sigma_h={g['sigma_h']:.5f}  sigma_c={g['sigma_c']:.5f}  "
        f"kappa={g['kappa']:.5f}  SSE={g['sse']:.6e}",
        f"  REFINED: sigma_h={r['sigma_h']:.5f}  sigma_c={r['sigma_c']:.5f}  "
        f"kappa={r['kappa']:.5f}  SSE={r['sse']:.6e}  "
        f"(Nelder-Mead, {r['n_iter']} iters)",
        "",
        f"Error bars: ±1 standard error of the mean (SEM = std_cbf_obs / "
        f"sqrt({N_REPS}), where {N_REPS} is the number of independent QPU "
        f"replicates per L value).",
    ])
    return dict(path=out_pdf, at_us=int(at_us),
                grid=g, refined=r, n=len(L_list))


# ----------------------------------------------------------------------------
# Figure 5 — RQ4 heatmap CBF(L, k)
# ----------------------------------------------------------------------------
def build_rq4_heatmap(run_dir):
    sweep_csv = os.path.join(run_dir, "rq4_at20us_cs_sweep",
                              "sweep_summary_per_L.csv")
    out_pdf = os.path.join(run_dir, "rq4_at20us_cs_sweep",
                            "fig_heatmap_cbf_L_k_AT20us.pdf")
    out_txt = out_pdf.replace(".pdf", "_caption.txt")
    if not os.path.isfile(sweep_csv):
        safe_print(f"[skip] Figure 5: {sweep_csv} not found")
        return None
    df = pd.read_csv(sweep_csv)
    df = df.dropna(subset=["L", "chain_strength", "mean_cbf_obs"]).copy()
    if df.empty:
        safe_print("[skip] Figure 5: sweep summary has no usable rows")
        return None

    pivot = df.pivot_table(index="L", columns="chain_strength",
                            values="mean_cbf_obs", aggfunc="mean")
    pivot = pivot.sort_index().sort_index(axis=1)
    cs_vals = np.asarray(pivot.columns, dtype=float)
    L_vals = np.asarray(pivot.index, dtype=int)
    Z = pivot.to_numpy(dtype=float)

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.gca()
    extent = [cs_vals.min() - 0.05, cs_vals.max() + 0.05,
              L_vals.min() - 2.5, L_vals.max() + 2.5]
    im = ax.imshow(Z, aspect="auto", origin="lower",
                   extent=extent, cmap="viridis",
                   interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("mean CBF")

    # White contour at CBF = 0.02
    if np.any(np.isfinite(Z)) and np.nanmin(Z) <= 0.02 <= np.nanmax(Z):
        cs_grid_2d, L_grid_2d = np.meshgrid(cs_vals, L_vals)
        cs_contour = ax.contour(cs_grid_2d, L_grid_2d, Z,
                                levels=[0.02], colors="white",
                                linewidths=1.4)
        ax.clabel(cs_contour, inline=True, fontsize=8, fmt="CBF=0.02")

    ax.set_xlabel(r"chain strength $k$")
    ax.set_ylabel(r"$L$")
    ax.set_title(r"RQ4: mean CBF vs $(L, k)$ at $T_a=20\,\mu$s")
    ax.set_yticks(L_TICKS_HEATMAP)
    fig.savefig(out_pdf)
    plt.close(fig)

    write_caption_txt(out_txt, [
        "Figure 5 — RQ4 heatmap of mean CBF vs (L, k)",
        f"Data: {sweep_csv}",
        f"L grid: {L_vals.tolist()}",
        f"k grid: {cs_vals.tolist()}",
        f"Total cells: {Z.size}, valid: {int(np.isfinite(Z).sum())}",
        "White contour: CBF = 0.02",
    ])
    return dict(path=out_pdf, n_L=len(L_vals), n_cs=len(cs_vals),
                cbf_min=float(np.nanmin(Z)), cbf_max=float(np.nanmax(Z)))


# ----------------------------------------------------------------------------
# Figure 3 — RQ2 obs vs predicted CBF (single-panel line plot, AT=5us cs=1.0)
# ----------------------------------------------------------------------------
def build_rq2_obs_vs_pred(run_dir):
    sub = "rq2_at5us_cs1p0"
    summary_csv = os.path.join(run_dir, sub, "summary_per_L_RQ2.csv")
    raw_npz = os.path.join(run_dir, sub, "raw_vectors_RQ2.npz")
    out_pdf = os.path.join(run_dir, sub, "cbf_obs_pred_RQ2.pdf")
    out_txt = out_pdf.replace(".pdf", "_caption.txt")
    if not (os.path.isfile(summary_csv) and os.path.isfile(raw_npz)):
        safe_print(f"[skip] Figure 3 (RQ2): inputs missing")
        return None
    L_list, cbf_obs, chain_lengths_dict, std_obs = load_summary_and_chains(
        summary_csv, raw_npz)
    if not L_list:
        safe_print(f"[skip] Figure 3 (RQ2): no usable rows")
        return None

    fit = fit_ice_for_dataset(L_list, cbf_obs, chain_lengths_dict,
                              verbose=False)
    cbf_pred = [fit["predictions"][int(L)] for L in L_list]
    sem = _sem_array(std_obs)

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.gca()
    ax.errorbar(L_list, cbf_obs, yerr=sem,
                fmt="-o", color="#1f77b4", markersize=5, linewidth=1.5,
                label="Observed", **ERRORBAR_KW)
    ax.plot(L_list, cbf_pred, "--s", color="#ff7f0e", markersize=4,
            linewidth=1.3, label="Predicted (ICE, refined)")
    ax.set_xlabel(r"$L$")
    ax.set_ylabel("mean CBF")
    ax.set_title(r"RQ2: $T_a = 5\,\mu$s,  $k = 1.0$")
    ax.set_xticks(L_TICKS)
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="upper left", framealpha=0.9)
    fig.savefig(out_pdf)
    plt.close(fig)

    g, r = fit["grid"], fit["refined"]
    write_caption_txt(out_txt, [
        "Figure 3 — RQ2 obs vs predicted CBF (AT=5us, k=1.0)",
        f"Data: {summary_csv}",
        f"N = {len(L_list)} L values",
        "",
        "Fitted ICE parameters (independent Gaussian model):",
        f"  GRID   : sigma_h={g['sigma_h']:.5f}  sigma_c={g['sigma_c']:.5f}  "
        f"kappa={g['kappa']:.5f}  SSE={g['sse']:.6e}",
        f"  REFINED: sigma_h={r['sigma_h']:.5f}  sigma_c={r['sigma_c']:.5f}  "
        f"kappa={r['kappa']:.5f}  SSE={r['sse']:.6e}  "
        f"(Nelder-Mead, {r['n_iter']} iters)",
        "",
        f"Error bars: ±1 standard error of the mean (SEM = std_cbf_obs / "
        f"sqrt({N_REPS}), where {N_REPS} is the number of independent QPU "
        f"replicates per L value).",
    ])
    return dict(path=out_pdf, grid=g, refined=r, n=len(L_list))


# ----------------------------------------------------------------------------
# Figure 7 — RQ4 obs vs predicted CBF for cs in {1.0, 1.5, 2.0, 2.5}
# ----------------------------------------------------------------------------
RQ4_CS_VALUES = [1.0, 1.5, 2.0, 2.5]
RQ4_CS_COLORS = {1.0: "#1f77b4", 1.5: "#2ca02c",
                  2.0: "#d62728", 2.5: "#9467bd"}


def build_rq4_multi_cs(run_dir):
    sweep_dir = os.path.join(run_dir, "rq4_at20us_cs_sweep")
    out_pdf = os.path.join(sweep_dir, "cbf_obs_pred_all_cs_AT20us.pdf")
    out_txt = out_pdf.replace(".pdf", "_caption.txt")

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.gca()
    used = []
    fits = {}
    for cs in RQ4_CS_VALUES:
        tag = _cs_tag(cs)
        summary_csv = os.path.join(sweep_dir, f"summary_per_L_RQ4_{tag}.csv")
        raw_npz = os.path.join(sweep_dir, f"raw_vectors_RQ4_{tag}.npz")
        if not (os.path.isfile(summary_csv) and os.path.isfile(raw_npz)):
            safe_print(f"  [skip cs={cs}]: inputs missing")
            continue
        L_list, cbf_obs, chain_lengths_dict, std_obs = load_summary_and_chains(
            summary_csv, raw_npz)
        if not L_list:
            safe_print(f"  [skip cs={cs}]: no usable rows")
            continue
        fit = fit_ice_for_dataset(L_list, cbf_obs, chain_lengths_dict,
                                   verbose=False)
        cbf_pred = [fit["predictions"][int(L)] for L in L_list]
        sem = _sem_array(std_obs)
        color = RQ4_CS_COLORS.get(cs, None)
        ax.errorbar(L_list, cbf_obs, yerr=sem,
                    fmt="-o", color=color, markersize=4, linewidth=1.4,
                    label=f"obs k={cs}", **ERRORBAR_KW)
        ax.plot(L_list, cbf_pred, "--s", color=color, markersize=4,
                linewidth=1.2, alpha=0.8, label=f"pred k={cs}")
        fits[cs] = fit
        used.append(cs)

    if not used:
        plt.close(fig)
        safe_print("[skip] Figure 7: no cs values with usable data")
        return None

    ax.set_xlabel(r"$L$")
    ax.set_ylabel("mean CBF")
    ax.set_title(r"RQ4: observed vs predicted CBF, $T_a=20\,\mu$s")
    ax.set_xticks(L_TICKS)
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="upper left", ncol=2, framealpha=0.9, fontsize=8)
    fig.savefig(out_pdf)
    plt.close(fig)

    cap_lines = [
        "Figure 7 — RQ4 obs vs predicted CBF for selected k values",
        f"Data dir: {sweep_dir}",
        "",
        "GRID fit (16x16x19):",
        f"{'  k':>5} {'sigma_h':>10} {'sigma_c':>10} {'kappa':>10} {'SSE':>14}",
    ]
    for cs in RQ4_CS_VALUES:
        if cs in fits:
            g = fits[cs]["grid"]
            cap_lines.append(f"{cs:>5.2f} {g['sigma_h']:>10.5f} "
                             f"{g['sigma_c']:>10.5f} {g['kappa']:>10.5f} "
                             f"{g['sse']:>14.6e}")
        else:
            cap_lines.append(f"{cs:>5.2f}     ---        ---        "
                             "---           --- (data missing)")
    cap_lines.append("")
    cap_lines.append("REFINED fit (Nelder-Mead from grid winner):")
    cap_lines.append(
        f"{'  k':>5} {'sigma_h':>10} {'sigma_c':>10} {'kappa':>10} {'SSE':>14}")
    for cs in RQ4_CS_VALUES:
        if cs in fits:
            r = fits[cs]["refined"]
            cap_lines.append(f"{cs:>5.2f} {r['sigma_h']:>10.5f} "
                             f"{r['sigma_c']:>10.5f} {r['kappa']:>10.5f} "
                             f"{r['sse']:>14.6e}")
        else:
            cap_lines.append(f"{cs:>5.2f}     ---        ---        "
                             "---           --- (data missing)")
    cap_lines += [
        "",
        f"Error bars: ±1 standard error of the mean (SEM = std_cbf_obs / "
        f"sqrt({N_REPS}), where {N_REPS} is the number of independent QPU "
        f"replicates per L value).",
    ]
    write_caption_txt(out_txt, cap_lines)

    return dict(path=out_pdf, used_cs=used,
                fits={cs: dict(grid=fits[cs]["grid"],
                                refined=fits[cs]["refined"])
                      for cs in used})


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Tables 3 / 4 — fitted parameters per AT / per k (refined)
# ----------------------------------------------------------------------------
def write_table3_RQ3(run_dir, rq3_results):
    """booktabs LaTeX for Table 3: RQ3 fitted parameters per anneal time."""
    out_dir = os.path.join(run_dir, "manuscript_tables")
    os.makedirs(out_dir, exist_ok=True)
    out_tex = os.path.join(out_dir, "table03_RQ3_fitted_params.tex")
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Fitted parameters $(\sigma_h,\sigma_c,\kappa)$ and fit "
        r"error (SSE) for different annealing times $T$ at fixed $k=1.5$. "
        r"Parameters refined by Nelder-Mead local optimisation starting "
        r"from a coarse grid winner.}",
        r"\label{tab:fit_params_T}",
        r"\begin{tabular}{ccccc}",
        r"\hline",
        r"$T$ [$\mu$s] & $\sigma_h$ & $\sigma_c$ & $\kappa$ & SSE \\",
        r"\hline",
    ]
    for r in rq3_results:
        if r is None:
            continue
        ref = r["refined"]
        sse_mant, sse_exp = _scientific(ref["sse"])
        lines.append(
            f"{r['at_us']} & {ref['sigma_h']:.4f} & "
            f"{ref['sigma_c']:.4f} & {ref['kappa']:.4f} & "
            f"${sse_mant:.3f} \\times 10^{{{sse_exp}}}$ \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    with open(out_tex, "w") as f:
        f.write("\n".join(lines) + "\n")
    return out_tex


def write_table4_RQ4(run_dir, rq4_multi_cs_result):
    """booktabs LaTeX for Table 4: RQ4 fitted parameters per chain strength."""
    out_dir = os.path.join(run_dir, "manuscript_tables")
    os.makedirs(out_dir, exist_ok=True)
    out_tex = os.path.join(out_dir, "table04_RQ4_fitted_params.tex")
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Fitted parameters $(\sigma_h,\sigma_c,\kappa)$ and fit "
        r"error (SSE) for different chain strengths $k$ at fixed "
        r"$T=20\,\mu$s. Parameters refined by Nelder-Mead local "
        r"optimisation starting from a coarse grid winner.}",
        r"\label{tab:fit_params_k}",
        r"\begin{tabular}{ccccc}",
        r"\hline",
        r"$k$ & $\sigma_h$ & $\sigma_c$ & $\kappa$ & SSE \\",
        r"\hline",
    ]
    if rq4_multi_cs_result and rq4_multi_cs_result.get("fits"):
        for cs in RQ4_CS_VALUES:
            if cs not in rq4_multi_cs_result["fits"]:
                continue
            ref = rq4_multi_cs_result["fits"][cs]["refined"]
            sse_mant, sse_exp = _scientific(ref["sse"])
            lines.append(
                f"{cs:.1f} & {ref['sigma_h']:.4f} & "
                f"{ref['sigma_c']:.4f} & {ref['kappa']:.4f} & "
                f"${sse_mant:.3f} \\times 10^{{{sse_exp}}}$ \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    with open(out_tex, "w") as f:
        f.write("\n".join(lines) + "\n")
    return out_tex


def _scientific(x):
    """Return (mantissa, exponent) for x written as mantissa * 10^exponent."""
    if x == 0 or not np.isfinite(x):
        return 0.0, 0
    exp = int(np.floor(np.log10(abs(x))))
    mant = x / (10 ** exp)
    return float(mant), int(exp)


def print_grid_vs_refined(rq3_results, rq4_multi_cs):
    """Print before/after comparison and flag any param that moved >50%."""
    safe_print("\n" + "=" * 92)
    safe_print("Table 3 — RQ3 fitted parameters per AT (k=1.5)")
    safe_print("-" * 92)
    safe_print(f"  {'T (us)':>7} | {'σ_h grid':>8} {'σ_h ref':>8} | "
               f"{'σ_c grid':>8} {'σ_c ref':>8} | "
               f"{'κ grid':>8} {'κ ref':>8} | "
               f"{'SSE grid':>10} {'SSE ref':>10} | flag")
    flags = []
    for r in rq3_results:
        if r is None:
            continue
        g, ref = r["grid"], r["refined"]
        moved = _moved_more_than(g, ref, 0.5)
        flag = (",".join(moved) if moved else "ok")
        if moved:
            flags.append(f"RQ3 T={r['at_us']}us: {moved}")
        safe_print(
            f"  {r['at_us']:>7} | {g['sigma_h']:>8.4f} {ref['sigma_h']:>8.4f} | "
            f"{g['sigma_c']:>8.4f} {ref['sigma_c']:>8.4f} | "
            f"{g['kappa']:>8.4f} {ref['kappa']:>8.4f} | "
            f"{g['sse']:>10.3e} {ref['sse']:>10.3e} | {flag}")

    safe_print("\n" + "=" * 92)
    safe_print("Table 4 — RQ4 fitted parameters per chain strength k (T=20us)")
    safe_print("-" * 92)
    safe_print(f"  {'k':>5} | {'σ_h grid':>8} {'σ_h ref':>8} | "
               f"{'σ_c grid':>8} {'σ_c ref':>8} | "
               f"{'κ grid':>8} {'κ ref':>8} | "
               f"{'SSE grid':>10} {'SSE ref':>10} | flag")
    if rq4_multi_cs and rq4_multi_cs.get("fits"):
        for cs in RQ4_CS_VALUES:
            if cs not in rq4_multi_cs["fits"]:
                continue
            g = rq4_multi_cs["fits"][cs]["grid"]
            ref = rq4_multi_cs["fits"][cs]["refined"]
            moved = _moved_more_than(g, ref, 0.5)
            flag = (",".join(moved) if moved else "ok")
            if moved:
                flags.append(f"RQ4 k={cs}: {moved}")
            safe_print(
                f"  {cs:>5.2f} | {g['sigma_h']:>8.4f} {ref['sigma_h']:>8.4f} | "
                f"{g['sigma_c']:>8.4f} {ref['sigma_c']:>8.4f} | "
                f"{g['kappa']:>8.4f} {ref['kappa']:>8.4f} | "
                f"{g['sse']:>10.3e} {ref['sse']:>10.3e} | {flag}")
    if flags:
        safe_print("\n*** Cells where refinement moved a param by >50% "
                   "(suggests grid was too coarse): ***")
        for f in flags:
            safe_print(f"   {f}")


def latest_run_dir():
    base = "out"
    if not os.path.isdir(base):
        return None
    runs = [os.path.join(base, d) for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d))]
    if not runs:
        return None
    return sorted(runs)[-1]


def build_parser():
    p = argparse.ArgumentParser(
        description="Regenerate manuscript figures from existing CSV/NPZ.")
    p.add_argument("--run-dir", default=None,
                   help="out/<ts>/ directory. Defaults to most recent under "
                        "out/.")
    p.add_argument("--dry-run", action="store_true")
    return p


PLAN = [
    ("Figure 2", "{run}/rq2_at5us_cs1p0/summary_per_L_RQ2.csv",
     "{run}/rq2_at5us_cs1p0/chainlen_scaling.pdf"),
    ("Figure 4 a", "{run}/rq3_at5us_cs1p5/summary_per_L_RQ3_AT5us.csv",
     "{run}/rq3_at5us_cs1p5/cbf_obs_pred_T5us.pdf"),
    ("Figure 4 b", "{run}/rq3_at20us_cs1p5/summary_per_L_RQ3_AT20us.csv",
     "{run}/rq3_at20us_cs1p5/cbf_obs_pred_T20us.pdf"),
    ("Figure 4 c", "{run}/rq3_at100us_cs1p5/summary_per_L_RQ3_AT100us.csv",
     "{run}/rq3_at100us_cs1p5/cbf_obs_pred_T100us.pdf"),
    ("Figure 4 d", "{run}/rq3_at200us_cs1p5/summary_per_L_RQ3_AT200us.csv",
     "{run}/rq3_at200us_cs1p5/cbf_obs_pred_T200us.pdf"),
    ("Figure 5", "{run}/rq4_at20us_cs_sweep/sweep_summary_per_L.csv",
     "{run}/rq4_at20us_cs_sweep/fig_heatmap_cbf_L_k_AT20us.pdf"),
    ("Figure 7", "{run}/rq4_at20us_cs_sweep/summary_per_L_RQ4_cs<1p0|1p5|2p0|2p5>.csv",
     "{run}/rq4_at20us_cs_sweep/cbf_obs_pred_all_cs_AT20us.pdf"),
]


def print_plan(run_dir):
    safe_print(f"=== make_paper_figures dry-run ===")
    safe_print(f"  run-dir: {run_dir}")
    for name, src, dst in PLAN:
        safe_print(f"  {name}")
        safe_print(f"    in : {src.format(run=run_dir)}")
        safe_print(f"    out: {dst.format(run=run_dir)}")
    safe_print("\n--dry-run: no files written.")


def print_summary(run_dir, rq1_res, rq3_results, rq4_heatmap, rq4_multi_cs):
    safe_print("\n" + "=" * 64)
    safe_print("Generated figures")
    safe_print("=" * 64)
    items = []
    if rq1_res:
        items.append(rq1_res["path"])
    for r in rq3_results:
        if r:
            items.append(r["path"])
    if rq4_heatmap:
        items.append(rq4_heatmap["path"])
    if rq4_multi_cs:
        items.append(rq4_multi_cs["path"])
    for p in items:
        if os.path.isfile(p):
            sz = os.path.getsize(p)
            safe_print(f"  {p}  ({sz:>7d} bytes)")
        else:
            safe_print(f"  {p}  (NOT WRITTEN)")

    # RQ3 fitted-params table (refined values)
    safe_print("\nRQ3 — fitted ICE parameters per anneal time (k=1.5, refined):")
    safe_print(f"  {'T (us)':>7}  {'sigma_h':>10}  {'sigma_c':>10}  "
               f"{'kappa':>10}  {'SSE':>14}")
    for r in rq3_results:
        if r:
            ref = r["refined"]
            safe_print(f"  {r['at_us']:>7d}  {ref['sigma_h']:>10.5f}  "
                       f"{ref['sigma_c']:>10.5f}  {ref['kappa']:>10.5f}  "
                       f"{ref['sse']:>14.6e}")

    # RQ4 fitted-params table (refined values)
    safe_print("\nRQ4 — fitted ICE parameters per chain strength (T=20us, refined):")
    safe_print(f"  {'k':>5}  {'sigma_h':>10}  {'sigma_c':>10}  "
               f"{'kappa':>10}  {'SSE':>14}")
    if rq4_multi_cs:
        for cs in RQ4_CS_VALUES:
            if cs in rq4_multi_cs["fits"]:
                ref = rq4_multi_cs["fits"][cs]["refined"]
                safe_print(f"  {cs:>5.2f}  {ref['sigma_h']:>10.5f}  "
                           f"{ref['sigma_c']:>10.5f}  {ref['kappa']:>10.5f}  "
                           f"{ref['sse']:>14.6e}")
            else:
                safe_print(f"  {cs:>5.2f}    (data missing — skipped)")

    if rq1_res:
        safe_print("\nRQ1 — chain length scaling:")
        safe_print(f"  slope = {rq1_res['slope']:.5f}  "
                   f"intercept = {rq1_res['intercept']:.5f}  "
                   f"R^2 = {rq1_res['r2']:.6f}  "
                   f"N = {rq1_res['n']}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir or latest_run_dir()
    if not run_dir:
        safe_print("ERROR: no run-dir specified and no out/<ts>/ exists.")
        return 1
    if not os.path.isdir(run_dir):
        safe_print(f"ERROR: run-dir {run_dir} does not exist.")
        return 1

    if args.dry_run:
        print_plan(run_dir)
        return 0

    safe_print(f"[run-dir] {run_dir}")

    safe_print("\n[Figure 2] RQ1 chain-length scaling")
    rq1_res = build_rq1_chainlen_figure(run_dir)

    safe_print("\n[Figure 3] RQ2 obs vs predicted CBF (AT=5us, k=1.0)")
    rq2_res = build_rq2_obs_vs_pred(run_dir)

    safe_print("\n[Figure 4] RQ3 obs vs predicted CBF (one panel per AT)")
    rq3_results = []
    for at_us in (5, 20, 100, 200):
        r = build_rq3_panel(run_dir, at_us)
        rq3_results.append(r)

    safe_print("\n[Figure 5] RQ4 heatmap")
    rq4_heatmap = build_rq4_heatmap(run_dir)

    safe_print("\n[Figure 7] RQ4 obs vs predicted CBF for selected cs values")
    rq4_multi_cs = build_rq4_multi_cs(run_dir)

    # Tables 3 & 4 from refined fits
    safe_print("\n[Tables 3 & 4] writing booktabs LaTeX from refined fits")
    t3_path = write_table3_RQ3(run_dir, rq3_results)
    t4_path = write_table4_RQ4(run_dir, rq4_multi_cs)
    safe_print(f"  saved: {t3_path}")
    safe_print(f"  saved: {t4_path}")

    print_summary(run_dir, rq1_res, rq3_results, rq4_heatmap, rq4_multi_cs)
    print_grid_vs_refined(rq3_results, rq4_multi_cs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
