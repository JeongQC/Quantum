"""qa_utils.py — shared utilities for QA experiments.

Extracted from RQ234.ipynb during Phase 1 of the paper revision so the same
helpers can be imported by rq234_revised.py and (later) rq5_time_matched.py.

Contents
--------
- Env-var configuration: ENDPOINT / TOKEN / SOLVER  (TASK A)
- Seed helpers
- BQM generation: make_random_bqm
- D-Wave sampler / embedding helpers: get_qpu, get_clique_sampler,
  get_L_list_from_clique, find_clique_embedding, find_bqm_embedding,
  chain_lengths_from_embedding, chain_lengths_from_sampleset
- Solver-param negotiation (anneal_time vs anneal_schedule): apply_anneal_params
- Occurrence-aware summarizers (TASK C):
    get_occurrences, expand_by_occurrences,
    summarize_cbf, summarize_energy
- QPU timing extraction (TASK D): extract_qpu_timing
- Optional simulated-annealing sampler import (for Phase 3 / RQ5)

Never hard-codes a D-Wave token. All credentials come from environment
variables. See .env.example at the repo root.
"""

from __future__ import annotations

import datetime
import os
import random
import time

import numpy as np


# -----------------------------------------------------------------------------
# Optional dependency wrapping
# -----------------------------------------------------------------------------
# D-Wave Ocean SDK is required for any QPU/embedding code path, but qa_utils
# itself can be imported (and unit-tested) without it. Functions that touch
# Ocean classes raise a RuntimeError if the SDK is missing.
try:
    import dimod
    import networkx as nx
    import minorminer
    from dwave.system import (
        DWaveCliqueSampler,
        DWaveSampler,
        FixedEmbeddingComposite,
    )
    from dwave.embedding.chain_strength import uniform_torque_compensation
    DWAVE_AVAILABLE = True
    DWAVE_IMPORT_ERROR = None
except Exception as _e:  # pragma: no cover - depends on environment
    DWAVE_AVAILABLE = False
    DWAVE_IMPORT_ERROR = repr(_e)
    dimod = None
    nx = None
    minorminer = None
    DWaveCliqueSampler = None
    DWaveSampler = None
    FixedEmbeddingComposite = None
    uniform_torque_compensation = None

# tqdm is purely cosmetic — fall back to a no-op identity wrapper.
try:
    from tqdm.auto import tqdm  # noqa: F401
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

# Simulated-annealing sampler — optional, only used by Phase 3 (RQ5). Prefer
# dwave-neal when present; otherwise fall back to dwave.samplers.
SimulatedAnnealingSampler = None
SA_BACKEND = None
try:
    from neal import SimulatedAnnealingSampler as _NealSA
    SimulatedAnnealingSampler = _NealSA
    SA_BACKEND = "neal"
except Exception:
    try:
        from dwave.samplers import SimulatedAnnealingSampler as _DwSA
        SimulatedAnnealingSampler = _DwSA
        SA_BACKEND = "dwave.samplers"
    except Exception:
        SimulatedAnnealingSampler = None
        SA_BACKEND = None


# -----------------------------------------------------------------------------
# Environment-variable configuration  (TASK A)
# -----------------------------------------------------------------------------
ENDPOINT = os.getenv("DWAVE_API_ENDPOINT", "https://cloud.dwavesys.com/sapi")
TOKEN = os.getenv("DWAVE_API_TOKEN", None)
SOLVER = os.getenv("DWAVE_SOLVER", "Advantage2_system1")

MASTER_SEED = int(os.getenv("MASTER_SEED", "42"))


def seed_everything(seed: int | None = None) -> int:
    """Seed numpy and the stdlib `random` module. Returns the seed used."""
    if seed is None:
        seed = MASTER_SEED
    np.random.seed(int(seed))
    random.seed(int(seed))
    return int(seed)


def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _require_dwave():
    if not DWAVE_AVAILABLE:
        raise RuntimeError(
            f"D-Wave Ocean SDK is not importable: {DWAVE_IMPORT_ERROR}. "
            "Install dwave-ocean-sdk to use this code path."
        )


# -----------------------------------------------------------------------------
# Occurrence-aware helpers  (TASK C)
# -----------------------------------------------------------------------------
def get_occurrences(sampleset) -> np.ndarray:
    """Return num_occurrences as an int array of length len(record).

    Falls back to all-ones if the field is absent (e.g. some mocked samplers).
    """
    rec = sampleset.record
    if rec.dtype.names is not None and "num_occurrences" in rec.dtype.names:
        return np.asarray(rec["num_occurrences"], dtype=int)
    return np.ones(len(rec), dtype=int)


def expand_by_occurrences(values, occurrences) -> np.ndarray:
    return np.repeat(np.asarray(values), np.asarray(occurrences, dtype=int))


def summarize_cbf(sampleset) -> dict:
    """Read-level (occurrence-weighted) chain-break-fraction stats.

    Returns dict with mean, std, prob_break, n, vec — where ``vec`` is the
    chain-break-fraction expanded so that each unique sample contributes
    ``num_occurrences`` entries (one per anneal read).
    """
    occ = get_occurrences(sampleset)
    rec = sampleset.record

    if rec.dtype.names is not None and "chain_break_fraction" in rec.dtype.names:
        raw = np.asarray(rec["chain_break_fraction"], dtype=float)
    else:
        info = getattr(sampleset, "info", {}) or {}
        emb_ctx = info.get("embedding_context", {}) or {}
        if "chain_break_fraction" in emb_ctx:
            raw = np.asarray(emb_ctx["chain_break_fraction"], dtype=float)
            if len(raw) != len(rec):
                raw = np.resize(raw, len(rec))
        else:
            raw = np.full(len(rec), np.nan, dtype=float)

    finite = np.isfinite(raw)
    if not finite.any():
        return dict(mean=0.0, std=0.0, prob_break=0.0, n=0, vec=np.array([], dtype=float))

    raw_f = raw[finite]
    occ_f = occ[finite]
    vec = expand_by_occurrences(raw_f, occ_f)
    n = int(vec.size)
    if n == 0:
        return dict(mean=0.0, std=0.0, prob_break=0.0, n=0, vec=vec)
    mean = float(np.mean(vec))
    std = float(np.std(vec, ddof=1)) if n > 1 else 0.0
    prob_break = float(np.mean(vec > 0.0))
    return dict(mean=mean, std=std, prob_break=prob_break, n=n, vec=vec)


def summarize_energy(sampleset) -> dict:
    """Read-level (occurrence-weighted) energy stats.

    ``best`` is the global minimum over unique samples (occurrence weighting
    cannot lower it). ``mean`` and ``std`` are computed over the expanded
    read-level vector. ``vec`` is the expanded vector itself.
    """
    rec = sampleset.record
    raw = np.asarray(rec["energy"], dtype=float)
    occ = get_occurrences(sampleset)
    vec = expand_by_occurrences(raw, occ)
    n = int(vec.size)
    if n == 0:
        return dict(mean=0.0, std=0.0, best=0.0, n=0, vec=vec)
    mean = float(np.mean(vec))
    std = float(np.std(vec, ddof=1)) if n > 1 else 0.0
    best = float(np.min(raw))
    return dict(mean=mean, std=std, best=best, n=n, vec=vec)


# -----------------------------------------------------------------------------
# QPU timing extraction  (TASK D)
# -----------------------------------------------------------------------------
TIMING_FIELDS = (
    "qpu_sampling_time_us",
    "qpu_anneal_time_per_sample_us",
    "qpu_readout_time_per_sample_us",
    "qpu_delay_time_per_sample_us",
    "qpu_access_time_us",
    "qpu_access_overhead_time_us",
    "qpu_programming_time_us",
    "total_post_processing_time_us",
    "post_processing_overhead_time_us",
)


def extract_qpu_timing(sampleset) -> dict:
    """Pull QPU timing fields out of sampleset.info['timing'].

    Missing or non-numeric fields become NaN; this function never raises on
    a malformed/empty timing block.
    """
    timing = (getattr(sampleset, "info", {}) or {}).get("timing", {}) or {}

    def get(name):
        val = timing.get(name, np.nan)
        try:
            return float(val)
        except Exception:
            return np.nan

    return {
        "qpu_sampling_time_us":             get("qpu_sampling_time"),
        "qpu_anneal_time_per_sample_us":    get("qpu_anneal_time_per_sample"),
        "qpu_readout_time_per_sample_us":   get("qpu_readout_time_per_sample"),
        "qpu_delay_time_per_sample_us":     get("qpu_delay_time_per_sample"),
        "qpu_access_time_us":               get("qpu_access_time"),
        "qpu_access_overhead_time_us":      get("qpu_access_overhead_time"),
        "qpu_programming_time_us":          get("qpu_programming_time"),
        "total_post_processing_time_us":    get("total_post_processing_time"),
        "post_processing_overhead_time_us": get("post_processing_overhead_time"),
    }


# -----------------------------------------------------------------------------
# BQM generation
# -----------------------------------------------------------------------------
def make_random_bqm(L, h_range=(-1.0, 1.0), J_range=(-1.0, 1.0),
                    density=1.0, seed=0):
    """Random fully-connected (or sparse) QUBO → BQM.

    The seed is the QUBO seed used in CSV/NPZ metadata; passing the same seed
    twice yields the same QUBO.
    """
    _require_dwave()
    rng = np.random.default_rng(seed)
    Q = {}
    for i in range(L):
        Q[(i, i)] = rng.uniform(*h_range)
    for i in range(L):
        for j in range(i + 1, L):
            if rng.random() < density:
                Q[(i, j)] = rng.uniform(*J_range)
    return dimod.BinaryQuadraticModel.from_qubo(Q)


# -----------------------------------------------------------------------------
# D-Wave sampler / embedding helpers
# -----------------------------------------------------------------------------
def get_qpu(solver: str | None = None):
    _require_dwave()
    solver = solver or SOLVER
    if not TOKEN:
        log("WARNING: DWAVE_API_TOKEN not set; DWaveSampler may fail to auth.")
    return DWaveSampler(endpoint=ENDPOINT, token=TOKEN, solver=solver)


def get_clique_sampler(solver: str | None = None):
    _require_dwave()
    solver = solver or SOLVER
    if not TOKEN:
        log("WARNING: DWAVE_API_TOKEN not set; CliqueSampler may fail to auth.")
    return DWaveCliqueSampler(endpoint=ENDPOINT, token=TOKEN, solver=solver)


def get_L_list_from_clique(L_min=8, step=5, solver: str | None = None):
    cs = get_clique_sampler(solver=solver)
    L_max = getattr(cs, "largest_clique_size", None)
    if not isinstance(L_max, int) or L_max <= 0:
        raise RuntimeError("largest_clique_size unavailable from CliqueSampler")
    if L_min > L_max:
        L_min = L_max
    L_list = list(range(L_min, L_max + 1, step)) or [L_max]
    log(f"[L_max] {L_max}  |  [L_list] {L_list} (step={step})")
    return L_list, L_max


def find_clique_embedding(qpu, L, seed=0, verbose=True):
    """Embed a complete graph K_L into the hardware graph."""
    _require_dwave()
    hw = qpu.to_networkx_graph()
    K = nx.complete_graph(L)
    emb = minorminer.find_embedding(K.edges(), hw.edges(), random_seed=seed)
    if not emb:
        raise RuntimeError(f"Embedding failed for K_{L}")
    if verbose:
        lens = [len(c) for c in emb.values()]
        log(f"[Embedding-KL] L={L} chains={len(emb)} "
            f"meanCL={np.mean(lens):.2f} maxCL={np.max(lens)}")
    return emb


def find_bqm_embedding(qpu, bqm, seed=0, verbose=True):
    """Sparsity-aware embedding (uses BQM's quadratic graph instead of K_L)."""
    _require_dwave()
    hw = qpu.to_networkx_graph()
    G = nx.Graph()
    G.add_nodes_from(bqm.variables)
    G.add_edges_from(bqm.quadratic.keys())
    emb = minorminer.find_embedding(G.edges(), hw.edges(), random_seed=seed)
    if not emb or any(v not in emb for v in bqm.variables):
        raise RuntimeError("BQM-based embedding failed")
    if verbose:
        lens = [len(c) for c in emb.values()]
        log(f"[Embedding-BQM] n={len(emb)} "
            f"meanCL={np.mean(lens):.2f} maxCL={np.max(lens)}")
    return emb


def chain_lengths_from_embedding(emb) -> np.ndarray:
    return np.array([len(c) for c in emb.values()], dtype=int)


def chain_lengths_from_sampleset(sampleset) -> np.ndarray:
    """Recover chain-length vector from sampleset.info['embedding_context']."""
    info = getattr(sampleset, "info", {}) or {}
    emb_ctx = info.get("embedding_context") or {}
    emb = emb_ctx.get("embedding") or {}
    return np.array([len(c) for c in emb.values()], dtype=int)


# -----------------------------------------------------------------------------
# Anneal-time / schedule param negotiation
# -----------------------------------------------------------------------------
def _solver_allowed_params(qpu) -> set:
    try:
        return set(qpu.parameters.keys())
    except Exception:
        return set()


_RATE_LIMIT_KEYWORDS = (
    "rate", "limit", "quota", "429", "too many requests", "throttle",
    # macOS thread-exhaustion (defense in depth — subprocess isolation in
    # rerun_missing_cs.py / run_one_cs.py is the primary fix).
    "thread", "can't start new thread",
)


def _looks_like_rate_limit(exc) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in _RATE_LIMIT_KEYWORDS)


def sample_with_retry(sample_callable, *args,
                      max_retries: int = 3, base_wait: float = 5.0,
                      **kwargs):
    """Wrap a SDK ``sampler.sample(...)`` call with retry on rate-limit errors.

    Detects rate-limit errors by exception message content (keywords:
    ``rate``, ``limit``, ``quota``, ``429``, ``too many requests``,
    ``throttle``). Waits ``min(60, base_wait * 2**retry)`` seconds between
    attempts. Non-rate-limit exceptions are re-raised immediately. Returns
    whatever ``sample_callable(*args, **kwargs)`` returns on success.
    """
    last_exc = None
    for retry in range(max_retries + 1):
        try:
            return sample_callable(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if not _looks_like_rate_limit(e) or retry >= max_retries:
                raise
            wait = min(60.0, float(base_wait) * (2 ** retry))
            log(f"[rate-limit] attempt {retry + 1}/{max_retries + 1} hit "
                f"{e.__class__.__name__}: '{str(e)[:80]}' — sleeping "
                f"{wait:.1f}s before retry")
            time.sleep(wait)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("sample_with_retry exhausted retries without exception")


def apply_anneal_params(qpu, base_kwargs, anneal_time_us=None,
                        anneal_schedule=None, verbose=False) -> dict:
    """Map user-facing anneal options onto whatever the solver supports."""
    allowed = _solver_allowed_params(qpu)
    kw = dict(base_kwargs)
    if anneal_schedule is not None and "anneal_schedule" in allowed:
        kw["anneal_schedule"] = anneal_schedule
    elif anneal_time_us is not None:
        if "anneal_time" in allowed:
            kw["anneal_time"] = float(anneal_time_us)
        elif "anneal_schedule" in allowed:
            kw["anneal_schedule"] = [(0.0, 0.0), (float(anneal_time_us), 1.0)]
            if verbose:
                log(f"[adapt] anneal_time_us={anneal_time_us} -> linear schedule "
                    "(solver lacks 'anneal_time')")
        else:
            if verbose:
                log("[adapt] anneal_time unsupported; using default schedule.")
    return kw


__all__ = [
    "ENDPOINT", "TOKEN", "SOLVER", "MASTER_SEED",
    "DWAVE_AVAILABLE", "DWAVE_IMPORT_ERROR",
    "SimulatedAnnealingSampler", "SA_BACKEND",
    "tqdm", "log", "seed_everything",
    "get_occurrences", "expand_by_occurrences",
    "summarize_cbf", "summarize_energy",
    "TIMING_FIELDS", "extract_qpu_timing",
    "make_random_bqm",
    "get_qpu", "get_clique_sampler", "get_L_list_from_clique",
    "find_clique_embedding", "find_bqm_embedding",
    "chain_lengths_from_embedding", "chain_lengths_from_sampleset",
    "apply_anneal_params",
    "sample_with_retry",
]
