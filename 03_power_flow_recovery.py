"""
Power Flow Recovery - Implementing Algorithm 1 from Zamzam et al. 2019

This script implements the feasibility recovery procedure:
1. NN predicts α, β, and γ
2. Convert α/β/γ → pg, vm, qg
3. Clip qg to limits (should be rare with γ prediction!)
4. Solve power flow equations with fixed qg → get va
5. If violated: re-clip and re-solve
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pypower.api import case57, runpf, ppoption
from pypower.idx_bus import PD, QD, VM, VA, BUS_TYPE, PQ, PV, REF
from pypower.idx_gen import PG, QG, VG, QMAX, QMIN, PMAX, PMIN, GEN_BUS
import matplotlib.pyplot as plt
import os
import json
import time
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "Outputs_V3")
os.makedirs(output_dir, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print("="*60)
print("Power Flow Recovery - Algorithm 1 (Zamzam 2019)")
print(f"Run Time: {timestamp}")
print("="*60)

# ============================================
# LOAD NETWORK AND LIMITS
# ============================================
print(f"\nLoading network configuration...")

# Load generator limits
with open(os.path.join(output_dir, 'generator_limits.json'), 'r') as f:
    limits_data = json.load(f)

n_gens = limits_data['n_gens']
n_loads = limits_data['n_loads']
pg_limits = limits_data['pg_limits']
vm_min_orig = np.array(limits_data['vm_min_orig'])
vm_max_orig = np.array(limits_data['vm_max_orig'])
lambda_margin = limits_data['lambda']

print(f"  Generators: {n_gens}")
print(f"  Loads: {n_loads}")

# Load base network (PyPower format)
ppc_base = case57()
n_bus = ppc_base['bus'].shape[0]
gen_buses = ppc_base['gen'][:, GEN_BUS].astype(int)
slack_bus = gen_buses[0]  # First generator is typically slack

# Get reactive power limits from loaded data or from network
if 'qg_limits' in limits_data:
    qg_limits = limits_data['qg_limits']
    gen_qg_min = np.array([qg_limits[i][0] for i in range(n_gens)])
    gen_qg_max = np.array([qg_limits[i][1] for i in range(n_gens)])
    print(f"  ✓ Qg limits loaded from file")
else:
    # Fallback: extract from PyPower network (for old data files)
    gen_qg_min = ppc_base['gen'][:, QMIN]
    gen_qg_max = ppc_base['gen'][:, QMAX]
    qg_limits = [(gen_qg_min[i], gen_qg_max[i]) for i in range(n_gens)]
    print(f"  ⚠️ Qg limits not in file, extracted from PyPower network")

print(f"  Network loaded: IEEE 57-bus (PyPower)")
print(f"  Buses: {n_bus}, Generators: {n_gens}")
print(f"  Slack bus: {slack_bus}")
print(f"  Qg range: [{gen_qg_min.min():.1f}, {gen_qg_max.max():.1f}] MVAr")

# Get load bus indices
load_bus_mask = ppc_base['bus'][:, PD] > 0
load_bus_indices = np.where(load_bus_mask)[0]
print(f"  Load buses: {len(load_bus_indices)}")

# ============================================
# LOAD MODEL
# ============================================
class OPF_NN_Zamzam(nn.Module):
    """Neural Network for AC OPF following Zamzam et al. 2019"""
    def __init__(self, input_size, output_size):
        super(OPF_NN_Zamzam, self).__init__()
        hidden1_size = input_size
        hidden2_size = input_size
        hidden3_size = output_size

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden1_size),
            nn.Sigmoid(),
            nn.Linear(hidden1_size, hidden2_size),
            nn.Sigmoid(),
            nn.Linear(hidden2_size, hidden3_size),
            nn.Sigmoid(),
            nn.Linear(hidden3_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

print(f"\nLoading trained model...")
checkpoint_path = os.path.join(output_dir, 'best_opf_model_v3_improved.pth')
checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

input_size = checkpoint['input_size']
output_size = checkpoint['output_size']

model = OPF_NN_Zamzam(input_size, output_size).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

input_scaler = checkpoint['input_scaler']
# No output_scaler - outputs (α, β, γ) are already in [0,1]!

# Extract training metadata
training_time_seconds = checkpoint.get('training_time_seconds', None)
training_epoch = checkpoint.get('epoch', 'N/A')
num_training_samples = checkpoint.get('num_samples', 'N/A')

print(f"  Model loaded successfully")
if training_time_seconds is not None:
    training_time_minutes = training_time_seconds / 60
    print(f"  Training time: {training_time_minutes:.2f} minutes ({training_time_seconds:.1f} seconds)")
    print(f"  Training epoch: {training_epoch}")
    print(f"  Training samples: {num_training_samples}")
else:
    print(f"  ⚠️ Training time not available in checkpoint")

# ============================================
# LOAD TEST DATA
# ============================================
print(f"\nLoading test data...")
test_df = pd.read_csv(os.path.join(output_dir, 'opf_case57_test_v3.csv'))

# Extract inputs (pd and qd) - updated column names
load_pd_cols = [col for col in test_df.columns if col.startswith('pd_')]
load_qd_cols = [col for col in test_df.columns if col.startswith('qd_')]
qg_cols = [col for col in test_df.columns if col.startswith('qg_')]

X_test = test_df[load_pd_cols + load_qd_cols].values
objectives_true = test_df['objective'].values

# Extract true Qg from OPF (for comparison with predictions)
if len(qg_cols) > 0:
    qg_true_all = test_df[qg_cols].values  # [n_samples, n_gens]
    print(f"  Test samples: {len(X_test)}")
    print(f"  ✓ Loaded true Qg from OPF for comparison")
else:
    qg_true_all = None
    print(f"  Test samples: {len(X_test)}")
    print(f"  ⚠️ No Qg columns found - will use power flow Qg for comparison")

# ============================================
# HELPER FUNCTIONS
# ============================================

def alpha_beta_to_pg_vm(alphas, betas, pg_limits, vm_min, vm_max):
    """
    Convert α/β parameterization to actual pg and vm values

    Args:
        alphas: α values for non-slack generators [n_gens-1]
        betas: β values for all generator buses [n_gens]
        pg_limits: List of (min, max) tuples for all generators
        vm_min, vm_max: Voltage limits

    Returns:
        pg_all: Active power for all generators [n_gens]
        vm_all: Voltage magnitudes for all generator buses [n_gens]
    """
    # Convert alphas to pg (excluding slack, will be determined by power balance)
    pg_non_slack = []
    for i, alpha in enumerate(alphas):
        pg_min, pg_max = pg_limits[i+1]  # Skip slack (index 0)
        pg = pg_min + alpha * (pg_max - pg_min)
        pg_non_slack.append(pg)

    # Convert betas to vm (all generator buses)
    vm_all = []
    for i, beta in enumerate(betas):
        # For generator buses, use restricted limits
        vm = vm_min + beta * (vm_max - vm_min)
        vm_all.append(vm)

    return pg_non_slack, vm_all


def solve_power_flow(ppc, load_pd, load_qd, pg_non_slack, vm_all, load_bus_indices):
    """
    Solve power flow equations given loads, pg (non-slack), and vm

    This is equation (4) from the paper:
    find v, pg, qg
    subject to:
        |vn| = |vn^o|  for all generator buses
        pg,n = pg,n^o  for all non-slack generators
        h(v, pg, qg) = 0  (power flow equations)

    Returns:
        success: Boolean
        qg_all: Reactive power for all generators
        va_all: Voltage angles for all buses
        pg_slack: Active power from slack bus
    """
    # Create a copy of the network
    ppc_copy = ppc.copy()
    ppc_copy['bus'] = ppc['bus'].copy()
    ppc_copy['gen'] = ppc['gen'].copy()
    ppc_copy['branch'] = ppc['branch'].copy()

    # Set loads
    ppc_copy['bus'][load_bus_indices, PD] = load_pd
    ppc_copy['bus'][load_bus_indices, QD] = load_qd

    # Set generator active power (non-slack generators, indices 1 to n_gens-1)
    for i, pg in enumerate(pg_non_slack):
        ppc_copy['gen'][i+1, PG] = pg

    # Set voltage magnitudes for all generators
    for i in range(len(vm_all)):
        ppc_copy['gen'][i, VG] = vm_all[i]

    # Run power flow
    ppopt = ppoption(VERBOSE=0, OUT_ALL=0)
    try:
        results = runpf(ppc_copy, ppopt)

        if not results[0]['success']:
            return False, None, None, None

        # Extract results
        qg_all = results[0]['gen'][:, QG]
        pg_slack = results[0]['gen'][0, PG]
        va_all = results[0]['bus'][:, VA]

        return True, qg_all, va_all, pg_slack

    except Exception as e:
        return False, None, None, None


def check_reactive_limits(qg_all, qg_min, qg_max):
    """
    Check if reactive power violates limits
    Returns: violated_indices, clipped_qg
    """
    violated = []
    clipped_qg = qg_all.copy()

    for i in range(len(qg_all)):
        if qg_all[i] < qg_min[i]:
            violated.append(i)
            clipped_qg[i] = qg_min[i]
        elif qg_all[i] > qg_max[i]:
            violated.append(i)
            clipped_qg[i] = qg_max[i]

    return violated, clipped_qg


def solve_modified_power_flow(ppc, load_pd, load_qd, pg_non_slack, vm_all, qg_fixed, load_bus_indices):
    """
    Solve modified power flow with fixed reactive power at violated generators.

    This is equation (6) from the paper:
    find v, pg, qg
    subject to:
        |vn| = |vn^o|  for non-violated generator buses
        pg,n = pg,n^o  for all non-slack generators
        qg,n = qg,n^r  for violated generators
        h(v, pg, qg) = 0
    
    CRITICAL IMPLEMENTATION:
    - Violated generators: Change from PV bus (fixed Vm) to PQ bus (fixed Pg, Qg)
      This allows Vm to adjust while Qg stays at the clipped limit
    - Non-violated generators: Keep as PV bus (fixed Vm, solved Qg)
    """
    # Create a copy of the network
    ppc_copy = ppc.copy()
    ppc_copy['bus'] = ppc['bus'].copy()
    ppc_copy['gen'] = ppc['gen'].copy()
    ppc_copy['branch'] = ppc['branch'].copy()

    # Set loads
    ppc_copy['bus'][load_bus_indices, PD] = load_pd
    ppc_copy['bus'][load_bus_indices, QD] = load_qd

    # Set generator active power for non-slack generators
    for i, pg in enumerate(pg_non_slack):
        ppc_copy['gen'][i+1, PG] = pg

    # Determine which generators have violated Qg limits
    # qg_fixed will be different from original only for violated generators
    qg_original = ppc['gen'][:, QG]  # Get original Qg from base case
    violated_indices = []
    for i in range(len(qg_fixed)):
        # A generator is violated if qg_fixed differs from limits or original pattern
        qmin = ppc['gen'][i, QMIN]
        qmax = ppc['gen'][i, QMAX]
        # Check if qg_fixed is at a limit (within tolerance)
        if abs(qg_fixed[i] - qmin) < 0.1 or abs(qg_fixed[i] - qmax) < 0.1:
            violated_indices.append(i)

    # CRITICAL: Change bus types for violated generators
    for i in range(len(vm_all)):
        gen_bus_idx = int(ppc_copy['gen'][i, GEN_BUS]) - 1  # Convert to 0-indexed
        
        if i in violated_indices:
            # Violated generator: Switch to PQ bus type
            # This makes Vm a solved variable while Pg and Qg are fixed
            ppc_copy['bus'][gen_bus_idx, BUS_TYPE] = PQ
            ppc_copy['gen'][i, QG] = qg_fixed[i]  # Fix Qg at clipped value
            ppc_copy['gen'][i, VG] = vm_all[i]  # Initial guess, will be solved
            
            # Tighten bounds to enforce fixed Qg
            ppc_copy['gen'][i, QMIN] = qg_fixed[i]
            ppc_copy['gen'][i, QMAX] = qg_fixed[i]
        else:
            # Non-violated generator: Keep as PV bus (default for generators)
            # Vm is fixed, Qg is solved
            if i == 0:  # Slack bus
                ppc_copy['bus'][gen_bus_idx, BUS_TYPE] = REF
            else:
                ppc_copy['bus'][gen_bus_idx, BUS_TYPE] = PV
            ppc_copy['gen'][i, VG] = vm_all[i]  # Fix Vm

    ppopt = ppoption(VERBOSE=0, OUT_ALL=0)
    try:
        results = runpf(ppc_copy, ppopt)

        if not results[0]['success']:
            return False, None, None, None, None

        qg_all = results[0]['gen'][:, QG]
        pg_slack = results[0]['gen'][0, PG]
        va_all = results[0]['bus'][:, VA]
        
        # CRITICAL: Extract updated Vm from bus voltages
        # For violated generators (PQ bus), Vm has adjusted
        # For non-violated generators (PV bus), Vm should match setpoint
        gen_buses_idx = ppc['gen'][:, GEN_BUS].astype(int) - 1
        vm_all_updated = results[0]['bus'][gen_buses_idx, VM]

        return True, qg_all, va_all, pg_slack, vm_all_updated

    except Exception as e:
        return False, None, None, None, None


# ============================================
# ALGORITHM 1: POWER FLOW RECOVERY
# ============================================

def recover_full_solution(alpha_beta_pred, load_pd, load_qd, ppc_base, load_bus_indices):
    """
    Implement Algorithm 1 from Zamzam et al. 2019

    Input: alpha_beta_pred from neural network (α and β only)
    Output: full AC OPF solution (pg, qg, vm, va) or None if failed
    
    Note: Qg is NOT predicted - it's determined by power flow equations
    """
    # Step 1: Convert α/β to pg and vm
    n_alphas = n_gens - 1  # Exclude slack
    n_betas = n_gens

    alphas = alpha_beta_pred[:n_alphas]
    betas = alpha_beta_pred[n_alphas:n_alphas + n_betas]
    
    # Step 1b: Convert α/β to pg and vm
    pg_non_slack, vm_all = alpha_beta_to_pg_vm(
        alphas, betas, pg_limits,
        vm_min_orig.min(), vm_max_orig.max()
    )

    # Step 2: Solve power flow equations (Equation 4)
    # This determines Qg naturally through power balance
    success, qg_all, va_all, pg_slack = solve_power_flow(
        ppc_base, load_pd, load_qd, pg_non_slack, vm_all, load_bus_indices
    )

    if not success:
        return None, "power_flow_failed"

    # Step 3: Check reactive power limits
    # Power flow determined Qg - check if it respects limits
    violated_indices, qg_clipped = check_reactive_limits(
        qg_all, gen_qg_min, gen_qg_max
    )

    # Step 4: If violated, solve modified power flow (Equation 6)
    if len(violated_indices) > 0:
        success, qg_all_corrected, va_all_corrected, pg_slack_corrected, vm_all_corrected = solve_modified_power_flow(
            ppc_base, load_pd, load_qd, pg_non_slack, vm_all, qg_clipped, load_bus_indices
        )

        if not success:
            return None, "modified_pf_failed"
        
        # Update with corrected values (including updated Vm from PQ bus adjustments)
        qg_all = qg_all_corrected
        va_all = va_all_corrected
        pg_slack = pg_slack_corrected
        vm_all = vm_all_corrected  # CRITICAL: Use updated Vm from modified power flow

    # Construct full solution
    pg_all = np.concatenate([[pg_slack], pg_non_slack])

    solution = {
        'pg': pg_all,
        'qg': qg_all,
        'vm': vm_all,
        'va': va_all,
        'violations': len(violated_indices),
        'violated_indices': violated_indices
    }

    return solution, "success"


# ============================================
# MEASURE BASELINE: PYPOWER OPF TIME
# ============================================
print(f"\n" + "="*60)
print("Measuring Baseline: PyPower OPF Time")
print("="*60)

# Run OPF on a few test samples to get accurate baseline timing
from pypower.api import runopf

ppopt = ppoption(VERBOSE=0, OUT_ALL=0)
opf_times = []
num_opf_samples = min(10, len(X_test))  # Measure 10 samples

print(f"Running PyPower OPF on {num_opf_samples} samples...")
for i in range(num_opf_samples):
    # Create a copy of the base case
    ppc_test = ppc_base.copy()
    ppc_test['bus'] = ppc_base['bus'].copy()
    ppc_test['gen'] = ppc_base['gen'].copy()
    ppc_test['branch'] = ppc_base['branch'].copy()
    
    # Set loads from test sample
    load_pd = X_test[i, :n_loads]
    load_qd = X_test[i, n_loads:2*n_loads]
    
    for bus_idx, load_idx in enumerate(load_bus_indices):
        ppc_test['bus'][load_idx, PD] = load_pd[bus_idx]
        ppc_test['bus'][load_idx, QD] = load_qd[bus_idx]
    
    # Time the OPF solve
    start = time.time()
    result = runopf(ppc_test, ppopt)
    opf_time = (time.time() - start) * 1000  # ms
    
    if result['success']:
        opf_times.append(opf_time)

if opf_times:
    pypower_opf_time = np.mean(opf_times)
    print(f"✓ PyPower OPF average time: {pypower_opf_time:.2f} ms/sample (measured)")
    print(f"  Range: [{np.min(opf_times):.2f}, {np.max(opf_times):.2f}] ms")
else:
    # Fallback to estimate if measurement fails
    pypower_opf_time = 100
    print(f"⚠ Could not measure OPF time, using estimate: {pypower_opf_time} ms")


# ============================================
# RUN RECOVERY ON TEST SET
# ============================================
print(f"\n" + "="*60)
print("Running Power Flow Recovery on Test Set")
print("="*60)

# Make predictions
X_test_norm = input_scaler.transform(X_test)
X_test_tensor = torch.FloatTensor(X_test_norm).to(DEVICE)

# ============================================
# ACCURATE INFERENCE TIMING
# ============================================
# Problem: Batch processing entire test set gives unrealistic speedup
# Solution: Measure per-sample inference time properly
print(f"\nMeasuring accurate inference time...")

# Warmup GPU (important for accurate timing)
with torch.no_grad():
    _ = model(X_test_tensor[:10])

# Method 1: Single sample inference (most realistic for real-time deployment)
single_sample_times = []
num_timing_samples = min(100, len(X_test))  # Time 100 samples
for i in range(num_timing_samples):
    sample = X_test_tensor[i:i+1]  # Single sample as batch of 1
    start = time.time()
    with torch.no_grad():
        _ = model(sample)
    single_sample_times.append((time.time() - start) * 1000)  # ms

avg_single_inference_time = np.mean(single_sample_times)

# Method 2: Small batch inference (batch_size=32, more realistic than full test set)
batch_size = 32
batch_times = []
num_batches = min(10, len(X_test) // batch_size)  # Time 10 batches
for i in range(num_batches):
    batch = X_test_tensor[i*batch_size:(i+1)*batch_size]
    start = time.time()
    with torch.no_grad():
        _ = model(batch)
    batch_time = (time.time() - start) * 1000  # ms
    batch_times.append(batch_time / batch_size)  # ms per sample

avg_batch_inference_time = np.mean(batch_times)

# Now run full inference for all predictions (we still need the actual predictions!)
start_time = time.time()
with torch.no_grad():
    alpha_beta_pred = model(X_test_tensor).cpu().numpy()
full_batch_time = time.time() - start_time
full_batch_per_sample = (full_batch_time * 1000) / len(X_test)

# No denormalization needed - outputs are already in [0,1]!
# (α, β are designed to be in [0,1] range)

print(f"\n✓ NN Inference complete")
print(f"\nInference timing comparison:")
print(f"  Single-sample mode: {avg_single_inference_time:.3f} ms/sample (most realistic)")
print(f"  Batch-32 mode: {avg_batch_inference_time:.3f} ms/sample (moderate batching)")
print(f"  Full-batch mode: {full_batch_per_sample:.3f} ms/sample (unrealistic speedup)")
print(f"\n  → Using single-sample timing for speedup calculations")
if training_time_seconds is not None:
    print(f"  Training time (from checkpoint): {training_time_minutes:.2f} minutes ({training_time_seconds:.1f} seconds)")

# Use the most realistic (single-sample) timing
inference_time_per_sample = avg_single_inference_time  # ms

# Recover full solutions
print(f"\nRecovering full AC OPF solutions...")
results = []
recovery_times = []
pf_failures = 0
modified_pf_failures = 0
qg_violations = 0

for i in tqdm(range(len(X_test)), desc="Processing samples"):
    load_pd = X_test[i, :n_loads]
    load_qd = X_test[i, n_loads:2*n_loads]

    start = time.time()
    solution, status = recover_full_solution(
        alpha_beta_pred[i], load_pd, load_qd, ppc_base, load_bus_indices
    )
    recovery_time = (time.time() - start) * 1000  # ms
    recovery_times.append(recovery_time)

    if status == "success":
        results.append(solution)
        if solution['violations'] > 0:
            qg_violations += 1
    elif status == "power_flow_failed":
        pf_failures += 1
        results.append(None)
    elif status == "modified_pf_failed":
        modified_pf_failures += 1
        results.append(None)

# ============================================
# ANALYSIS - Following Zamzam et al. 2019 Metrics
# ============================================
print(f"\n" + "="*60)
print("Recovery Results")
print("="*60)

successful = sum([1 for r in results if r is not None])
# Calculate total time per sample using accurate inference timing
total_time_per_sample = inference_time_per_sample + np.mean(recovery_times)  # ms

print(f"\nSuccess Metrics:")
print(f"  Total samples: {len(X_test)}")
print(f"  Successful recovery: {successful}/{len(X_test)} ({100*successful/len(X_test):.1f}%)")
print(f"  Power flow failures: {pf_failures}")
print(f"  Modified PF failures: {modified_pf_failures}")
print(f"  Samples with Qg violations: {qg_violations}/{successful} ({100*qg_violations/successful:.1f}%)")

# ============================================
# ZAMZAM METRIC: δ_q (Average Qg violation magnitude)
# ============================================
# From Zamzam paper Section IV.C:
# δ_q = (1/T) Σ (1/|G|) ||ξ_q,t||₂
# where ξ_q,t,n = max{Qg_min - Qg, 0} + max{Qg - Qg_max, 0}

total_qg_violation = 0.0  # Sum of all violations across all samples
violation_details = []  # For detailed analysis

for i, result in enumerate(results):
    if result is not None:
        # Get the Qg from initial power flow (before clipping)
        # We need to re-run initial PF to get the uncorrected Qg
        load_pd = X_test[i, :n_loads]
        load_qd = X_test[i, n_loads:2*n_loads]
        
        # Convert α/β to pg and vm
        n_alphas = n_gens - 1
        n_betas = n_gens
        alphas = alpha_beta_pred[i, :n_alphas]
        betas = alpha_beta_pred[i, n_alphas:n_alphas + n_betas]
        pg_non_slack, vm_all = alpha_beta_to_pg_vm(
            alphas, betas, pg_limits,
            vm_min_orig.min(), vm_max_orig.max()
        )
        
        # Run initial power flow to get uncorrected Qg
        success_pf, qg_initial, _, _ = solve_power_flow(
            ppc_base, load_pd, load_qd, pg_non_slack, vm_all, load_bus_indices
        )
        
        if success_pf:
            # Calculate ξ_q for this sample (Zamzam Eq. after Eq. 7)
            xi_q = np.zeros(n_gens)
            for gen_idx in range(n_gens):
                qg = qg_initial[gen_idx]
                qg_min = gen_qg_min[gen_idx]
                qg_max = gen_qg_max[gen_idx]
                
                # Violation below min
                below_min = max(qg_min - qg, 0.0)
                # Violation above max
                above_max = max(qg - qg_max, 0.0)
                
                xi_q[gen_idx] = below_min + above_max
            
            # L2 norm of violations for this sample
            xi_q_norm = np.linalg.norm(xi_q, ord=2)
            
            # Add to total (divided by |G| as per paper)
            total_qg_violation += xi_q_norm / n_gens
            
            violation_details.append({
                'sample': i,
                'xi_q_norm': xi_q_norm,
                'num_violated': np.sum(xi_q > 0),
                'max_violation': np.max(xi_q),
                'violations_per_gen': xi_q
            })

# Calculate δ_q (average across all samples)
delta_q = total_qg_violation / successful if successful > 0 else 0

print(f"\nZamzam Metric - Reactive Power Violations:")
print(f"  δ_q (average violation): {delta_q:.3f} MVAr")
print(f"  Compare to Zamzam Table III (IEEE 57-bus, λ=0.005): δ_q = 1.58 MVAr")
if delta_q < 2.0:
    print(f"  ✓ Within expected range for this method!")
elif delta_q < 5.0:
    print(f"  ~ Slightly higher than Zamzam's results")
else:
    print(f"  ⚠ Higher than expected - may need tuning")

print(f"\nTiming (per sample) - ACCURATE MEASUREMENT:")
print(f"  NN inference: {inference_time_per_sample:.3f} ms")
print(f"  Power flow recovery: {np.mean(recovery_times):.2f} ms")
print(f"  Total (inference + recovery): {total_time_per_sample:.3f} ms")
print(f"\nNote: Training was a one-time cost of {training_time_minutes:.2f} minutes.")
print(f"      Inference time ({total_time_per_sample:.3f} ms/sample) is what matters for deployment.")

print(f"\nSpeedup Analysis:")

# Baselines for comparison (pypower_opf_time was measured earlier)
# pypower_opf_time already measured above
matpower_opf_time = 2000  # ms (Zamzam's MATPOWER baseline from paper)
zamzam_nn_time = 211      # ms (Zamzam's NN+recovery time from Table III, using MATPOWER for PF)

our_time = total_time_per_sample  # Use accurate per-sample timing

# Calculate speedups
speedup_vs_our_opf = pypower_opf_time / our_time
speedup_vs_matpower = matpower_opf_time / our_time

print(f"  Our NN+recovery time: {our_time:.3f} ms")
print(f"\n  Speedup vs our PyPower OPF (~{pypower_opf_time:.2f} ms): {speedup_vs_our_opf:.2f}x")
print(f"  Speedup vs MATPOWER OPF (~{matpower_opf_time} ms): {speedup_vs_matpower:.1f}x")

print(f"\n  COMPARISON TO ZAMZAM PAPER:")
print(f"    • Zamzam NN+recovery: ~211 ms (using MATPOWER for power flow)")
print(f"    • Our NN+recovery: {our_time:.2f} ms (using PyPower for power flow)")
print(f"    • Difference explained by: PyPower is {matpower_opf_time/pypower_opf_time:.1f}x faster than MATPOWER")
print(f"\n    ➜ If we used MATPOWER like Zamzam, our time would be ~{inference_time_per_sample + np.mean(recovery_times)*(matpower_opf_time/pypower_opf_time):.1f} ms")
print(f"       (NN inference + recovery scaled to MATPOWER speed)")
print(f"\n  Zamzam paper (IEEE 57-bus): SF = 9.49x (vs MATPOWER)")
print(f"  Our result (IEEE 57-bus): SF = {speedup_vs_matpower:.1f}x (vs MATPOWER)")

if speedup_vs_matpower > 100:
    print(f"\n  ⚠ Our speedup seems too high - likely due to:")
    print(f"     1. PyPower being much faster than MATPOWER ({pypower_opf_time:.1f} vs 2000 ms)")
    print(f"     2. Power flow recovery is the bottleneck, not NN inference")
    print(f"     3. Algorithm is identical to Zamzam, just different PF solver")
elif speedup_vs_our_opf < 1:
    print(f"\n  ⚠ NN is SLOWER than PyPower OPF - expected for small networks")
else:
    print(f"\n  ✓ Reasonable speedup given PyPower's efficiency")

# ============================================
# SAVE RESULTS
# ============================================
print(f"\n" + "="*60)
print("Saving Results")
print("="*60)

# Save recovered solutions
recovered_data = {
    'sample_idx': [],
    'success': [],
    'pg_slack': [],
    'qg_violations': [],
    'qg_violation_magnitude': [],  # Add δ_q per sample
    'recovery_time_ms': []
}

# Create lookup for violation magnitudes
violation_lookup = {v['sample']: v['xi_q_norm'] / n_gens for v in violation_details}

for i, (result, rec_time) in enumerate(zip(results, recovery_times)):
    recovered_data['sample_idx'].append(i)
    recovered_data['success'].append(result is not None)
    recovered_data['pg_slack'].append(result['pg'][0] if result else np.nan)
    recovered_data['qg_violations'].append(result['violations'] if result else np.nan)
    recovered_data['qg_violation_magnitude'].append(violation_lookup.get(i, 0.0))
    recovered_data['recovery_time_ms'].append(rec_time)

results_df = pd.DataFrame(recovered_data)
results_path = os.path.join(output_dir, 'recovery_results_v3.csv')
results_df.to_csv(results_path, index=False)
print(f"  ✓ Saved recovery results to: {results_path}")

# Save full solutions for successful cases (for optimality analysis)
full_solutions = []
for i, result in enumerate(results):
    if result is not None:
        # Convert to list (handles both numpy arrays and lists)
        pg = result['pg'].tolist() if hasattr(result['pg'], 'tolist') else result['pg']
        qg = result['qg'].tolist() if hasattr(result['qg'], 'tolist') else result['qg']
        vm = result['vm'].tolist() if hasattr(result['vm'], 'tolist') else result['vm']
        va = result['va'].tolist() if hasattr(result['va'], 'tolist') else result['va']

        full_solutions.append({
            'sample_idx': i,
            'pg': pg,
            'qg': qg,
            'vm': vm,
            'va': va
        })

import pickle
solutions_path = os.path.join(output_dir, 'recovered_solutions_v3.pkl')
with open(solutions_path, 'wb') as f:
    pickle.dump(full_solutions, f)
print(f"  ✓ Saved full solutions to: {solutions_path}")

# ============================================
# VISUALIZATION
# ============================================
print(f"\nGenerating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
title_str = f'Power Flow Recovery Analysis (V3 Model)\nGenerated: {timestamp}'
if training_time_seconds is not None:
    title_str += f' | Training: {training_time_minutes:.1f} min'
fig.suptitle(title_str, fontsize=16, fontweight='bold')

# Plot 1: Success rate
ax = axes[0, 0]
categories = ['Successful', 'PF Failed', 'Modified\nPF Failed']
counts = [successful, pf_failures, modified_pf_failures]
colors = ['green', 'orange', 'red']
bars = ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')

for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{count}\n({100*count/len(X_test):.1f}%)',
            ha='center', va='bottom', fontweight='bold')

ax.set_ylabel('Number of Samples')
ax.set_title('Recovery Success Rate')
ax.grid(True, alpha=0.3, axis='y')

# Plot 2: Recovery time distribution
ax = axes[0, 1]
ax.hist(recovery_times, bins=20, alpha=0.7, edgecolor='black')
ax.axvline(np.mean(recovery_times), color='r', linestyle='--',
           linewidth=2, label=f'Mean: {np.mean(recovery_times):.1f} ms')
ax.set_xlabel('Recovery Time (ms)')
ax.set_ylabel('Frequency')
ax.set_title('Power Flow Recovery Time Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: δ_q metric (Zamzam's violation magnitude metric)
ax = axes[1, 0]
if len(violation_details) > 0:
    # Extract per-sample violation magnitudes
    sample_violations = [v['xi_q_norm'] / n_gens for v in violation_details]
    
    ax.hist(sample_violations, bins=20, alpha=0.7, color='orange', edgecolor='black')
    ax.axvline(delta_q, color='r', linestyle='--', linewidth=2, 
               label=f'Average δ_q: {delta_q:.2f} MVAr')
    ax.axvline(1.58, color='g', linestyle=':', linewidth=2,
               label=f'Zamzam (λ=0.005): 1.58 MVAr')
    ax.set_xlabel('Qg Violation Magnitude (MVAr per sample)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Reactive Power Violation Distribution\n(δ_q metric from Zamzam 2019)')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Plot 4: Speedup comparison
ax = axes[1, 1]
# Note: Zamzam used MATPOWER for power flow, we use PyPower (much faster)
# So we show both our actual time and what it would be with MATPOWER
our_time_with_matpower = inference_time_per_sample + np.mean(recovery_times)*(matpower_opf_time/pypower_opf_time)
methods = ['MATPOWER\nOPF', 'Zamzam NN\n(MATPOWER PF)', 'Our NN\n(PyPower PF)']
zamzam_nn_time = 211  # From paper
times_comparison = [matpower_opf_time, zamzam_nn_time, total_time_per_sample]
colors_bar = ['red', 'orange', 'green']

bars = ax.bar(methods, times_comparison, color=colors_bar, alpha=0.7, edgecolor='black')

# Add value labels on bars
for bar, time_val in zip(bars, times_comparison):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{time_val:.1f} ms',
            ha='center', va='bottom', fontweight='bold')

ax.set_ylabel('Time per Sample (ms)')
ax.set_title(f'Speed Comparison\n(Our: {speedup_vs_our_opf:.2f}× vs PyPower, {speedup_vs_matpower:.1f}× vs MATPOWER)')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, axis='y', which='both')

plt.tight_layout()
plot_path = os.path.join(output_dir, 'recovery_analysis_v3.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"  ✓ Saved plots to: {plot_path}")

# Try to display the plot
try:
    print(f"\n  Displaying plot... (close window to continue)")
    plt.show(block=True)
except Exception as e:
    print(f"  ⚠ Could not display plot interactively: {e}")
    print(f"     Plot saved to: {plot_path}")

print("\n" + "="*60)
print("✅ Power Flow Recovery Complete!")
print("="*60)
print(f"\n📊 PERFORMANCE SUMMARY:")
print(f"\n1. MODEL TRAINING (one-time cost):")
if training_time_seconds is not None:
    print(f"   • Training time: {training_time_minutes:.2f} minutes ({training_time_seconds:.1f} seconds)")
    print(f"   • Training samples: {num_training_samples}")
    print(f"   • Best epoch: {training_epoch}")
else:
    print(f"   • Training time: Not available in checkpoint")
print(f"\n2. INFERENCE PERFORMANCE (per sample - deployment speed):")
print(f"   • Test samples evaluated: {len(X_test)}")
print(f"   • NN inference: {inference_time_per_sample:.3f} ms/sample")
print(f"   • Power flow recovery: {np.mean(recovery_times):.2f} ms/sample")
print(f"   • Total per-sample time: {total_time_per_sample:.3f} ms/sample")
print(f"\n   ⚡ Speedup: {speedup_vs_our_opf:.2f}× vs PyPower OPF ({pypower_opf_time:.2f} ms)")
print(f"\n3. RESULTS SAVED:")
print(f"   • Recovery results: recovery_results_v3.csv")
print(f"   • Full solutions: recovered_solutions_v3.pkl")
print(f"   • Visualization: recovery_analysis_v3.png")
print(f"\nNext step: Compute optimality gap (04_evaluate_with_metrics.py)")

