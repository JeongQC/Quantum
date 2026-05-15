# Embedding-Aware Noise Modeling of Quantum Annealing

**Seon-Geun Jeong<sup>1</sup>, Mai Dinh Cong<sup>1</sup>, Dae-Il Noh<sup>2</sup>, Quoc-Viet Pham<sup>3</sup> and Won-Joo Hwang<sup>1,4,*</sup>**

<sup>1</sup> Department of Information Convergence Engineering, Pusan National University, Busan, Republic of Korea  
<sup>2</sup> Quantum AI Lab, Korea Quantum Computing Co., Ltd, Busan, Republic of Korea  
<sup>3</sup> School of Computer Science and Statistics, Trinity College Dublin, Dublin, Ireland  
<sup>4</sup> School of Computer Science and Engineering, Pusan National University, Busan, Republic of Korea  

**Keywords:** Quantum Annealing, embedding, chain, chain strength, integrated control error model  

---

## 📝 Abstract
> Quantum annealing provides a practical realization of adiabatic quantum computation and has emerged as a promising approach for solving large-scale combinatorial optimization problems. However, current devices remain constrained by sparse hardware connectivity, which requires embedding logical variables into chains of physical qubits. This embedding overhead limits scalability and reduces reliability as longer chains are more prone to noise-induced errors. In this work, we develop an embedding-aware baseline noise framework that connects chain-length growth to accumulated integrated-control-error variance, chain break probability (CBP), chain break fraction (CBF), and chain-strength scaling. We distinguish a moment-level uncorrelated ICE baseline, which yields a linear variance law, from a Gaussian closure that gives a closed-form erfc approximation for CBP. Experiments on the D-Wave Advantage2 Zephyr processor show that the calibrated baseline model captures the dominant growth trend of CBF with embedding size. However, the empirical critical chain-strength exponent ranges from α=0.805 to α=1.006 across chain-break tolerance thresholds, with 95% bootstrap confidence intervals consistently excluding the independent-noise prediction α=0.5; a direct fit of a correlated-variance extension recovers γ≈1.8, consistent with the asymptotic relation γ=2α. These results indicate that real-hardware variance accumulation deviates from the independent-noise reference and is consistent with a correlated component. The proposed framework offers quantitative guidance for embedding-aware chain-strength tuning and motivates more general noise models incorporating correlated and non-Gaussian hardware effects.

**Preprint:** https://arxiv.org/abs/2510.04594

---

## Requirements

- Python 3.8 or higher
- D-Wave Ocean SDK
- A D-Wave Leap account and API token (https://cloud.dwavesys.com/leap/)
- Gurobi (free academic license recommended; commercial licenses also supported)

All Python dependencies are listed in `requirements.txt`.

---

## Installation

Clone this repository and install the Python dependencies:

```bash
git clone https://github.com/JeongQC/Quantum.git
cd Quantum
pip install -r requirements.txt
```

Create a `.env` file in the repository root containing your D-Wave API token:

```bash
echo "DWAVE_API_TOKEN=YOUR_TOKEN_HERE" > .env
```

> **Important:** Do not commit your `.env` file. It is excluded from version control via `.gitignore`.

---

## File overview

### Core experiment scripts

| File | Description |
| --- | --- |
| `qa_utils.py` | Common utilities for QA experiments (embedding, sampling, post-processing). |
| `rq234_revised.py` | Runs RQ2, RQ3, and RQ4 experiments (fixed-setting CBF, schedule sweep, chain-strength sweep). |
| `rq5_time_matched.py` | Runs RQ5: time-budgeted comparison among QA, SA, PuLP, and Gurobi. |
| `run_all_experiments.py` | Top-level orchestrator that runs the full experiment pipeline. |

### Analysis scripts

| File | Description |
| --- | --- |
| `analysis_kstar_correlated_followup.py` | Extracts critical chain strength `k*` from CBF data, fits the power law with bootstrap confidence intervals, and performs the 3- vs 5-parameter correlated-extension fit. |
| `analysis_validation_and_kstar.py` | Held-out validation splits (size-extrapolation and interleaved) on the fixed-setting CBF dataset. |

### Manuscript generation

| File | Description |
| --- | --- |
| `build_manuscript_assets.py` | Generates LaTeX tables and figure captions used in the manuscript. |
| `make_paper_figures.py` | Generates the paper figures (chain-length scaling, CBF curves, `k*` log-log plots, etc.). |

### Utility scripts

| File | Description |
| --- | --- |
| `phase4_finalize.py` | Final phase of the multi-phase experiment workflow. |
| `rerun_failed_experiments.py` | Re-runs experiments that failed in the main pipeline. |
| `rerun_missing_cs.py` | Fills in missing chain-strength configurations. |
| `run_one_cs.py` | Runs a single chain-strength configuration. |
| `check_thread_count.py` | Verifies single-threaded execution of classical solvers. |

### Notebooks (artifact record)

The Jupyter notebooks below were the original interactive form of the experiments. They are preserved as a frozen artifact record from the submission timeline; the maintained, refactored implementations live in the `.py` files above.

| File | Description |
| --- | --- |
| `RQ1.ipynb` | RQ1: chain-length scaling on Zephyr clique embeddings. Source of the linear chain-length-vs-`L` scaling fit. |
| `RQ234.ipynb` | RQ2 / RQ3 / RQ4: fixed-setting CBF, anneal-schedule sweep, and chain-strength sweep. Superseded for reproduction by `rq234_revised.py`. |
| `RQ5.ipynb` | RQ5: time-budgeted comparison between QA and classical baselines. Superseded for reproduction by `rq5_time_matched.py`. |

---

## How to reproduce

Once dependencies are installed and your `.env` file is configured, the main experiments can be reproduced as follows.

### Full pipeline

```bash
python run_all_experiments.py
```

### Individual research questions

```bash
# RQ2, RQ3, RQ4
python rq234_revised.py

# RQ5 (time-budgeted classical comparison)
python rq5_time_matched.py

# k* extraction, bootstrap CIs, 5-parameter correlated fit
python analysis_kstar_correlated_followup.py

# Held-out validation splits
python analysis_validation_and_kstar.py
```

### Regenerate paper figures and tables

```bash
python make_paper_figures.py
python build_manuscript_assets.py
```

---

## Experiment configuration

The main experimental settings used in the paper are summarized below:

- **QPU:** D-Wave `Advantage2_system1` (Zephyr topology)
- **Problem instances:** random fully-connected QUBOs, coefficients drawn from `U(-1, 1)`
- **Problem size:** `L = 5, 10, ..., 105`
- **Annealing time:** `T_a ∈ {5, 20, 100, 200} μs`
- **Chain strength:** `k ∈ {0.1, 0.2, ..., 2.5}`
- **Reads per data point:** `N = 2000`
- **Replicates per setting:** `n = 10`
- **Random seeds:** QUBO seed = `L`, embedding seed = `L`, classical solver seed = `42`

Classical solvers are run on a Mac mini (Apple M4 Pro, 48 GB RAM) with single-threaded execution.

---

## Citation

If you use this code, please cite the paper:

```bibtex
@article{jeong2026embedding,
  title   = {Embedding-Aware Noise Modeling of Quantum Annealing},
  author  = {Jeong, Seon-Geun and Cong, Mai Dinh and Noh, Dae-Il
             and Pham, Quoc-Viet and Hwang, Won-Joo},
  journal = {Quantum Information Processing},
  year    = {2026},
  note    = {Submitted}
}
```

---

## License

This project is released under the MIT License. See `LICENSE` for details.

---

## Contact

For questions or issues, please open a GitHub issue or contact:

- Seon-Geun Jeong (`wjdtjsrms11@pusan.ac.kr`)
- Won-Joo Hwang (corresponding author, `wjhwang@pusan.ac.kr`)

Pusan National University, South Korea
