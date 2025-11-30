"""
R-ACOPF Data Generation using PyPower

"""

import numpy as np
import pandas as pd
from pypower.api import case9, case57, runopf, ppoption, makeYbus
from pypower.idx_bus import PD, QD, VM, VA, BUS_TYPE, REF
from pypower.idx_gen import PG, QG, VG, QMAX, QMIN, PMAX, PMIN, GEN_BUS
from pypower.idx_brch import PF, QF, PT, QT
from scipy.stats import truncnorm
from scipy.linalg import cholesky
import os
from tqdm import tqdm
import json
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================
print("="*60)
print("R-ACOPF Dataset Generation (PyPower)")
print("="*60)

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "Outputs_v3")
os.makedirs(output_dir, exist_ok=True)

# Dataset parameters
N_SAMPLES = 100000
LAMBDA = 0.005  # Voltage margin 7
MU = 0.7  # Maximum load deviation
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
N_WORKERS = min(20, cpu_count() - 2)  # Use 20 workers, leave 2 cores for system

np.random.seed(42)

print(f"\nConfiguration:")
print(f"  Total samples: {N_SAMPLES}")
print(f"  Voltage margin (lambda): {LAMBDA}")
print(f"  Max load deviation (mu): {MU}")
print(f"  Parallel workers: {N_WORKERS}")

# ============================================
# LOAD BASE NETWORK
# ============================================
print(f"\nLoading IEEE 57-bus network...")
ppc_base = case57()

n_bus = ppc_base['bus'].shape[0]
n_gen = ppc_base['gen'].shape[0]

# Find load buses
load_mask = ppc_base['bus'][:, PD] > 0
n_loads = np.sum(load_mask)
load_bus_indices = np.where(load_mask)[0]

nominal_pd = ppc_base['bus'][load_mask, PD].copy()
nominal_qd = ppc_base['bus'][load_mask, QD].copy()

# Find generator buses
gen_buses = ppc_base['gen'][:, GEN_BUS].astype(int)
slack_bus = np.where(ppc_base['bus'][:, BUS_TYPE] == REF)[0][0]

print(f"\nNetwork information:")
print(f"  Buses: {n_bus}")
print(f"  Generators: {n_gen}")
print(f"  Load buses: {n_loads}")
print(f"  Slack bus: {slack_bus}")
print(f"  Total base load: {nominal_pd.sum():.2f} MW")

# ============================================
# TEST BASE OPF
# ============================================
print(f"\nTesting base OPF...")
ppopt = ppoption(VERBOSE=0, OUT_ALL=0)

results_test = runopf(ppc_base, ppopt)

if results_test['success']:
    print(f"  ✓ Base OPF converged successfully!")
    print(f"    Objective: ${results_test['f']:.2f}")
else:
    print(f"  ✗ Base OPF failed - network may not be properly configured")
    raise RuntimeError("Cannot proceed if base case doesn't work")

# ============================================
# SETUP COVARIANCE MATRIX
# ============================================
print(f"\nSetting up load covariance matrix...")

def create_correlation_matrix(n, decay=5.0):
    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            corr[i, j] = np.exp(-abs(i - j) / decay)
    return corr

corr_matrix = create_correlation_matrix(n_loads, decay=5.0)
std_devs = nominal_pd * MU / 3
cov_matrix = np.outer(std_devs, std_devs) * corr_matrix

print(f"  Covariance matrix shape: {cov_matrix.shape}")

# ============================================
# HELPER FUNCTIONS
# ============================================

def sample_truncated_multivariate_normal(mean, cov, lower, upper, size=1):
    samples = []
    n_dim = len(mean)
    
    try:
        L = cholesky(cov, lower=True)
    except:
        cov_reg = cov + np.eye(n_dim) * 1e-6
        L = cholesky(cov_reg, lower=True)
    
    attempts = 0
    max_attempts = size * 100
    
    while len(samples) < size and attempts < max_attempts:
        z = np.random.randn(n_dim)
        x = mean + L @ z
        
        if np.all(x >= lower) and np.all(x <= upper):
            samples.append(x)
        
        attempts += 1
    
    if len(samples) < size:
        samples = [np.random.uniform(lower, upper) for _ in range(size - len(samples))]
    
    return np.array(samples)


def compute_alpha_beta(pg, vm, gen_limits, vm_limits):
    """
    Convert physical values to normalized parameters α, β ∈ [0,1]
    
    Following Zamzam & Baker (2019):
    - α: active power (exclude slack)
    - β: voltage magnitude (all generators)
    - Qg is NOT parameterized - it will be determined by power flow
    """
    n_gen = len(pg)
    
    # Alpha (exclude slack generator at index 0)
    alphas = []
    for i in range(1, n_gen):
        p_min, p_max = gen_limits[i]
        if p_max > p_min:
            alpha = (pg[i] - p_min) / (p_max - p_min)
        else:
            alpha = 0.5
        alphas.append(np.clip(alpha, 0, 1))
    
    # Beta (all generators)
    betas = []
    for i in range(n_gen):
        v_min, v_max = vm_limits[i]
        if v_max > v_min:
            beta = (vm[i] - v_min) / (v_max - v_min)
        else:
            beta = 0.5
        betas.append(np.clip(beta, 0, 1))
    
    return alphas, betas


# ============================================
# COLLECT LIMITS
# ============================================
print(f"\nCollecting generator limits...")

gen_pg_min = ppc_base['gen'][:, PMIN]
gen_pg_max = ppc_base['gen'][:, PMAX]
gen_qg_min = ppc_base['gen'][:, QMIN]
gen_qg_max = ppc_base['gen'][:, QMAX]

# Voltage limits - FIXED: Column 12 is VMIN, Column 11 is VMAX!
vm_min_base = ppc_base['bus'][:, 12]  # VMIN column (index 12, constant VMIN=12)
vm_max_base = ppc_base['bus'][:, 11]  # VMAX column (index 11, constant VMAX=11)

gen_limits = [(gen_pg_min[i], gen_pg_max[i]) for i in range(n_gen)]

# Voltage limits for generator buses
vm_limits_gen = [(vm_min_base[int(gen_buses[i])] + LAMBDA, 
                  vm_max_base[int(gen_buses[i])] - LAMBDA)
                 for i in range(n_gen)]

# Reactive power limits for all generators (NEW!)
qg_limits = [(gen_qg_min[i], gen_qg_max[i]) for i in range(n_gen)]

print(f"  Generator limits: {n_gen}")
print(f"  Voltage range: [{vm_min_base.min():.3f}, {vm_max_base.max():.3f}] p.u.")
print(f"  Qg range: [{gen_qg_min.min():.1f}, {gen_qg_max.max():.1f}] MVAr")

# ============================================
# SETUP BOUNDS (needed by worker processes)
# ============================================
lower_bound = (1 - MU) * nominal_pd
upper_bound = (1 + MU) * nominal_pd

# ============================================
# PARALLEL SAMPLE GENERATION FUNCTION
# ============================================

def generate_single_sample(sample_idx, seed_offset=0):
    """
    Generate a single R-ACOPF sample (designed for parallel execution)
    
    Returns:
        dict or None: Sample data if successful, None if failed
    """
    # Set unique seed for this worker
    np.random.seed(42 + seed_offset + sample_idx)
    
    # Sample loads
    pd_sample = sample_truncated_multivariate_normal(
        mean=nominal_pd,
        cov=cov_matrix,
        lower=lower_bound,
        upper=upper_bound,
        size=1
    )[0]
    
    # Sample power factors
    power_factors = np.random.uniform(0.8, 1.0, n_loads)
    qd_sample = pd_sample * np.tan(np.arccos(power_factors))
    
    # Create modified case
    ppc = ppc_base.copy()
    ppc['bus'] = ppc_base['bus'].copy()
    ppc['gen'] = ppc_base['gen'].copy()
    ppc['branch'] = ppc_base['branch'].copy()
    
    # Apply sampled loads
    ppc['bus'][load_bus_indices, PD] = pd_sample
    ppc['bus'][load_bus_indices, QD] = qd_sample
    
    # Apply restricted voltage bounds - FIXED: 11=VMAX, 12=VMIN
    # Apply voltage constraints ONLY to generator buses
    for i, gen_bus_idx in enumerate(gen_buses):
        v_min_restricted = vm_min_base[gen_bus_idx] + LAMBDA
        v_max_restricted = vm_max_base[gen_bus_idx] - LAMBDA
        
        ppc['bus'][gen_bus_idx, 12] = v_min_restricted  # VMIN
        ppc['bus'][gen_bus_idx, 11] = v_max_restricted  # VMAX
    
    # Load buses keep original voltage limits (no λ margin applied)
    
    try:
        # Run OPF
        ppopt = ppoption(VERBOSE=0, OUT_ALL=0)
        results = runopf(ppc, ppopt)
        
        if not results['success']:
            return None
        
        # Extract solution
        pg = results['gen'][:, PG]
        qg = results['gen'][:, QG]
        vm = results['gen'][:, VG]
        objective = results['f']

        # Compute alpha/beta (no gamma - Qg determined by power flow)
        alphas, betas = compute_alpha_beta(
            pg, vm, gen_limits, vm_limits_gen
        )

        # Store sample
        sample_data = {
            'loads_pd': pd_sample.tolist(),
            'loads_qd': qd_sample.tolist(),
            'alphas': alphas,
            'betas': betas,
            'pg_all': pg.tolist(),
            'vm_all': vm.tolist(),
            'qg_all': qg.tolist(),  # Keep for validation/comparison
            'objective': float(objective)
        }
        
        return sample_data
        
    except Exception as e:
        return None


# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == '__main__':
    print(f"\n" + "="*60)
    print("Generating R-ACOPF Samples (Parallel)")
    print("="*60)

    # Parallel execution
    with Pool(processes=N_WORKERS) as pool:
        # Use imap_unordered for progress bar
        results = list(tqdm(
            pool.imap_unordered(generate_single_sample, range(N_SAMPLES)),
            total=N_SAMPLES,
            desc="Generating samples"
        ))
    
    # Filter out failed samples
    samples = [r for r in results if r is not None]
    failed_count = N_SAMPLES - len(samples)

    print(f"\n" + "="*60)
    print("Generation Complete!")
    print(f"  Successful: {len(samples)}")
    print(f"  Failed: {failed_count}")
    print(f"  Success rate: {100*len(samples)/N_SAMPLES:.1f}%")
    print("="*60)

    if len(samples) == 0:
        raise ValueError("No successful samples! Check network configuration.")

    # ============================================
    # PREPARE DATAFRAME
    # ============================================
    print(f"\nPreparing dataset...")

    load_pd_cols = [f'pd_{i}' for i in range(n_loads)]
    load_qd_cols = [f'qd_{i}' for i in range(n_loads)]
    alpha_cols = [f'alpha_{i}' for i in range(1, n_gen)]  # Exclude slack
    beta_cols = [f'beta_{i}' for i in range(n_gen)]
    qg_cols = [f'qg_{i}' for i in range(n_gen)]  # For validation only

    all_cols = load_pd_cols + load_qd_cols + alpha_cols + beta_cols + qg_cols + ['objective']

    data_rows = []
    for sample in samples:
        row = (
            sample['loads_pd'] +
            sample['loads_qd'] +
            sample['alphas'] +
            sample['betas'] +
            sample['qg_all'] +  # Keep for validation
            [sample['objective']]
        )
        data_rows.append(row)

    df = pd.DataFrame(data_rows, columns=all_cols)

    print(f"  Dataset shape: {df.shape}")
    print(f"  Input features: {len(load_pd_cols) + len(load_qd_cols)}")
    print(f"  Output features: {len(alpha_cols) + len(beta_cols)} (α, β)")

    # Verify bounds
    alpha_values = df[alpha_cols].values
    beta_values = df[beta_cols].values

    print(f"\nParameterization verification:")
    print(f"  Alpha range: [{alpha_values.min():.4f}, {alpha_values.max():.4f}]")
    print(f"  Beta range: [{beta_values.min():.4f}, {beta_values.max():.4f}]")
    print(f"  Qg stored for validation (not used in training)")

    # ============================================
    # SPLIT AND SAVE
    # ============================================
    print(f"\nSplitting data...")

    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    n_train = int(len(df_shuffled) * TRAIN_SPLIT)
    n_val = int(len(df_shuffled) * VAL_SPLIT)

    train_df = df_shuffled.iloc[:n_train]
    val_df = df_shuffled.iloc[n_train:n_train+n_val]
    test_df = df_shuffled.iloc[n_train+n_val:]

    print(f"  Train: {len(train_df)} samples")
    print(f"  Val: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")

    # Save
    train_df.to_csv(os.path.join(output_dir, 'opf_case57_train_v3.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'opf_case57_val_v3.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'opf_case57_test_v3.csv'), index=False)

    # Save limits
    limits_data = {
        'n_gens': int(n_gen),  # Convert numpy int to Python int
        'n_loads': int(n_loads),  # Convert numpy int to Python int
        'pg_limits': gen_limits,
        'qg_limits': qg_limits,  # NEW! Reactive power limits
        'vm_min_orig': vm_min_base.tolist(),
        'vm_max_orig': vm_max_base.tolist(),
        'lambda': float(LAMBDA),  # Ensure it's a Python float
    }

    with open(os.path.join(output_dir, 'generator_limits.json'), 'w') as f:
        json.dump(limits_data, f, indent=2)

    print(f"\n✅ Data saved to: {output_dir}")
    print(f"\n" + "="*60)
    print("✅ R-ACOPF Data Generation Complete (PyPower)!")
    print("="*60)
    print(f"\nUsing PyPower for better OPF convergence!")
    print(f"Network: IEEE 57-bus")
    print(f"Parameters: λ={LAMBDA}, μ={MU}")
