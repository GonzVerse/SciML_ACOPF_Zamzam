# Quick Start Guide

Get the complete OPF neural network pipeline running in 3 steps.

## Prerequisites

- Python 3.7+ (3.8+ recommended)
- CUDA-capable GPU (optional, but recommended for training)

## Installation

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd Version-3
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python verify_setup.py
```

This checks that all packages are installed correctly. You should see:
```
✓ SETUP COMPLETE - Ready to run!
```

That's it! All dependencies will be installed automatically.

## Running the Pipeline

### Option 1: Full Pipeline (Recommended for First Run)
Generates data, trains model, and produces all results (~2-3 hours):

```bash
python run_all.py
```

### Option 2: Quick Run (Skip Data Generation)
If you already have generated data in `Outputs_V3/`:

```bash
python run_all.py --quick
```

### Option 3: Step-by-Step Execution
Run scripts individually for debugging or partial execution:

```bash
# 1. Generate OPF dataset (100k samples, ~45 min)
python 00_generate_opf_data_pypower.py

# 2. Train neural network (~20 min with GPU)
python 02_train_opf_network_v3_improved.py

# 3. Run power flow recovery - Algorithm 1
python 03_power_flow_recovery.py

# 4. Comprehensive evaluation
python 04_evaluate_with_metrics.py

# 5. Analyze reactive power corrections
python 05_analyze_qg_correction.py

# 6. Visualize correction process
python 06_explain_correction_process.py

# 7. Sensitivity analysis (optional, ~2 hours)
python 07_sensitivity_analysis_qg_filtering.py
```

## Expected Output

All results are saved to `Outputs_V3/`:

### Data Files
- `opf_case57_train_v2.csv` - Training data (70k samples)
- `opf_case57_val_v2.csv` - Validation data (15k samples)  
- `opf_case57_test_v2.csv` - Test data (15k samples)

### Model Files
- `best_opf_model.pth` - Trained neural network weights
- `scaler_*.pkl` - Data normalization parameters
- `generator_limits.json` - Network configuration

### Results
- `recovery_results_v3.csv` - Power flow recovery results
- `evaluation_report.txt` - Performance metrics
- `recovered_solutions_v3.pkl` - Detailed solutions

### Visualizations
- `training_history.png` - Loss curves
- `qg_correction_analysis.png` - Reactive power analysis
- `qg_correction_walkthrough_*.png` - Step-by-step process
- `evaluation_*.png` - Performance charts
- And more...

## Performance Benchmarks

On a typical setup:
- **Data Generation**: ~45 minutes (100k samples)
- **Training**: ~20 minutes (GPU), ~2 hours (CPU)
- **Inference**: 0.061 ms per sample (single-sample mode)
- **Power Flow Recovery**: 17.92 ms per sample
- **Total Runtime**: 17.98 ms vs 2000 ms (MATPOWER OPF)
- **Speedup**: ~111× faster than traditional OPF

## Troubleshooting

### "No module named 'pypower'"
```bash
pip install PYPOWER
```

### "CUDA out of memory"
Reduce batch size in `02_train_opf_network_v3_improved.py`:
```python
BATCH_SIZE = 128  # Was 256
```

### "File not found: opf_case57_train_v2.csv"
Run data generation first:
```bash
python 00_generate_opf_data_pypower.py
```

### Slow performance
- **Data generation**: Set `N_WORKERS` lower in script (line 40)
- **Training**: Ensure PyTorch uses GPU (`torch.cuda.is_available()` should be `True`)
- **Power flow**: Already optimized with PyPower sparse solvers

### Want to use different IEEE case?
Edit `00_generate_opf_data_pypower.py`:
```python
case = case9  # Instead of case57
```
Note: May require adjusting network-specific parameters.

## Next Steps

- Check `README.md` for detailed methodology
- See `evaluation_report.txt` for full metrics
- Review visualizations in `Outputs_V3/`
- Modify hyperparameters in scripts for experimentation

## Citation

If you use this code, please cite:
```
Zamzam, A. S., & Sidiropoulos, N. D. (2019). Physics-aware neural networks 
for distribution system state estimation. IEEE Transactions on Power Systems.
```

## Support

For issues or questions:
- Check `README.md` troubleshooting section
- Review inline code documentation
- Open an issue on GitHub
