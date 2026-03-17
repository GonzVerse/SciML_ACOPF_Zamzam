# Neural Network AC Optimal Power Flow (AC-OPF)

> Reproducing and evaluating the method from **Zamzam & Baker (2019)**: replacing iterative AC-OPF optimization with a neural network + feasibility recovery pipeline on the IEEE 57-bus system.

![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![IEEE 57-Bus](https://img.shields.io/badge/IEEE-57--Bus%20System-orange)
![Course](https://img.shields.io/badge/Johns%20Hopkins-SciML-9370DB)

---

## Project Overview

This repository contains a full implementation of a learning-based AC Optimal Power Flow workflow for the IEEE 57-bus network. The goal is to predict near-optimal operating points from load demands in milliseconds, then recover physically feasible solutions via power flow corrections.

This work was completed as a graduate project in Scientific Machine Learning at Johns Hopkins University (Fall 2025).

### Why this matters

AC-OPF is central to power grid operation, but traditional solvers can be too slow for high-frequency decision settings. This project explores how machine learning can reduce runtime while preserving engineering feasibility.

---

## Key Results

| Metric | This implementation | Zamzam & Baker (2019) |
|---|:---:|:---:|
| Reactive power violation \(\delta_q\) | **0.967 MVAr** | 1.58 MVAr |
| Optimality gap | **< 0.5%** | 0.46% |
| Neural network inference time | **0.061 ms** | ~1 ms |
| End-to-end time (NN + recovery PF) | **17.98 ms** | 211 ms |
| Power flow convergence rate | **100%** | — |

- Approx. **5.9× faster** end-to-end runtime versus the reported paper benchmark.
- Approx. **3,500× faster** than iterative OPF when comparing NN inference only.

---

## What I implemented

This is an end-to-end reproduction and analysis effort, not only model training.

- **Data generation pipeline** for 100,000 R-ACOPF samples with correlated \(\pm 70\%\) load variation.
- **\(\alpha/\beta\) output parameterization** to keep predicted generator outputs within physical bounds before recovery.
- **Algorithm 1 (power flow recovery)** with Qg clipping and PV→PQ bus switching to restore feasibility.
- **Evaluation framework** for \(\delta_q\), optimality gap, feasibility checks, and timing benchmarks.
- **Sensitivity analysis** for Qg filtering thresholds (script `06_sensitivity_analysis_qg_filtering.py`).

---

## Method at a glance

### Model
- **Inputs:** bus-level active/reactive demands \((P_d, Q_d)\)
- **Outputs:** normalized \(\alpha\) (active generation) and \(\beta\) (voltage magnitude)
- **Architecture:** fully connected network, 3 hidden layers, sigmoid activations

### Constraint-aware parameterization

\[
P_g = P_{g,min} + \alpha (P_{g,max} - P_{g,min})
\]
\[
V_m = V_{m,min} + \beta (V_{m,max} - V_{m,min})
\]

### Feasibility recovery (Algorithm 1)
1. Predict \(\alpha,\beta\) from \((P_d,Q_d)\)
2. Map to physical \(P_g,V_m\)
3. Run power flow to obtain \(Q_g,V_a\)
4. Clip violating \(Q_g\), switch affected buses from PV to PQ
5. Re-run power flow with fixed \(Q_g\) and released voltage constraints

---

## Quick Start

### 1) Install

```bash
git clone <your-repo-url>
cd SciML_ACOPF_Zamzam
pip install -r requirements.txt
```

### 2) Run the full pipeline

```bash
python run_all.py          # ~2-3 hours (includes data generation)
python run_all.py --quick  # ~30 minutes (skips data generation)
```

### 3) Run individual stages

```bash
python 00_generate_opf_data_pypower.py
python 01_train_opf_network.py
python 02_power_flow_recovery.py
python 03_evaluate_with_metrics.py
python 04_analyze_qg_correction.py
python 05_explain_correction_process.py
python 06_sensitivity_analysis_qg_filtering.py
```

For setup details and troubleshooting, see [QUICKSTART.md](QUICKSTART.md).

---

## Repository Structure

```text
SciML_ACOPF_Zamzam/
├── 00_generate_opf_data_pypower.py
├── 01_train_opf_network.py
├── 02_power_flow_recovery.py
├── 03_evaluate_with_metrics.py
├── 04_analyze_qg_correction.py
├── 05_explain_correction_process.py
├── 06_sensitivity_analysis_qg_filtering.py
├── run_all.py
├── QUICKSTART.md
├── COMPREHENSIVE_SUMMARY.md
├── requirements.txt
├── LICENSE
└── Outputs_V3/
    ├── comprehensive_evaluation.png
    ├── network_topology.png
    └── generator_limits.json
```

---

## Reproducibility Notes

- Dataset generation and model training are stochastic and may produce small metric variation across runs.
- Runtime values depend on hardware and Python environment.
- The default scripts are configured for the IEEE 57-bus case.

---

## Visual Output

![Comprehensive evaluation](Outputs_V3/comprehensive_evaluation.png)

---

## References

```bibtex
@article{zamzam2019learning,
  title={Learning Optimal Solutions for Extremely Fast AC Optimal Power Flow},
  author={Zamzam, Ahmed S and Baker, Kyri},
  journal={arXiv preprint arXiv:1910.01213},
  year={2019}
}
```

Libraries and tools: PyTorch, PyPower, scikit-learn, and MATPOWER IEEE test cases.

---

## Author

**Jose Maria Borrego Acosta**  
Graduate Student, Johns Hopkins University

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
