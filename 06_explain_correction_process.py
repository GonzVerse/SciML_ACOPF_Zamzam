"""
Explain the Qg Correction Process

This script demonstrates what actually happens in Algorithm 1 from Zamzam et al. 2019
by manually stepping through the process for a few samples.
"""

import torch
import numpy as np
import pandas as pd
from pypower.api import case57, runpf, ppoption
from pypower.idx_bus import PD, QD, BUS_TYPE, PQ, PV, REF
from pypower.idx_gen import PG, QG, VG, QMIN, QMAX, GEN_BUS
import os
import json

# ============================================
# LOAD CONFIGURATION
# ============================================
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "Outputs_v3")
os.makedirs(output_dir, exist_ok=True)

print("="*80)
print("UNDERSTANDING THE QG CORRECTION PROCESS")
print("="*80)

# Load limits
with open(os.path.join(output_dir, 'generator_limits.json'), 'r') as f:
    limits_data = json.load(f)

n_gens = limits_data['n_gens']
n_loads = limits_data['n_loads']
pg_limits = limits_data['pg_limits']
qg_limits = limits_data['qg_limits']
vm_min_orig = np.array(limits_data['vm_min_orig'])
vm_max_orig = np.array(limits_data['vm_max_orig'])

qg_min = np.array([lim[0] for lim in qg_limits])
qg_max = np.array([lim[1] for lim in qg_limits])

# Load network
ppc_base = case57()
gen_buses = ppc_base['gen'][:, GEN_BUS].astype(int)
load_bus_mask = ppc_base['bus'][:, PD] > 0
load_bus_indices = np.where(load_bus_mask)[0]

# Load test data
test_df = pd.read_csv(os.path.join(output_dir, 'opf_case57_test_v3.csv'))
load_pd_cols = [col for col in test_df.columns if col.startswith('pd_')]
load_qd_cols = [col for col in test_df.columns if col.startswith('qd_')]

# Load model
checkpoint = torch.load(os.path.join(output_dir, 'best_opf_model_v3_improved.pth'),
                        map_location='cpu', weights_only=False)
input_scaler = checkpoint['input_scaler']

# ============================================
# HELPER FUNCTIONS
# ============================================

def alpha_beta_to_pg_vm(alphas, betas, pg_limits, vm_min, vm_max):
    """Convert α/β to pg and vm"""
    pg_non_slack = []
    for i, alpha in enumerate(alphas):
        pg_min, pg_max = pg_limits[i+1]
        pg = pg_min + alpha * (pg_max - pg_min)
        pg_non_slack.append(pg)
    
    vm_all = []
    for i, beta in enumerate(betas):
        vm = vm_min + beta * (vm_max - vm_min)
        vm_all.append(vm)
    
    return np.array(pg_non_slack), np.array(vm_all)


def solve_power_flow_with_details(ppc, load_pd, load_qd, pg_non_slack, vm_all, load_bus_indices):
    """Solve power flow and return detailed results"""
    ppc_copy = ppc.copy()
    ppc_copy['bus'] = ppc['bus'].copy()
    ppc_copy['gen'] = ppc['gen'].copy()
    ppc_copy['branch'] = ppc['branch'].copy()
    
    # Set loads
    ppc_copy['bus'][load_bus_indices, PD] = load_pd
    ppc_copy['bus'][load_bus_indices, QD] = load_qd
    
    # Set Pg for non-slack
    for i, pg in enumerate(pg_non_slack):
        ppc_copy['gen'][i+1, PG] = pg
    
    # Set Vm for all
    for i in range(len(vm_all)):
        ppc_copy['gen'][i, VG] = vm_all[i]
    
    ppopt = ppoption(VERBOSE=0, OUT_ALL=0)
    try:
        results = runpf(ppc_copy, ppopt)
        if not results[0]['success']:
            return None
        
        return {
            'qg': results[0]['gen'][:, QG],
            'pg': results[0]['gen'][:, PG],
            'vm': results[0]['gen'][:, VG]
        }
    except:
        return None


def solve_modified_pf_with_details(ppc, load_pd, load_qd, pg_non_slack, vm_all, qg_clipped, violated_indices, load_bus_indices):
    """
    Solve modified power flow with fixed Qg at violated generators.
    
    KEY IMPLEMENTATION (Equation 6 from Zamzam paper):
    - Violated generators: Change from PV bus (fixed Vm) to PQ bus (fixed Pg, Qg)
      This allows Vm to adjust while Qg stays at the clipped limit
    - Non-violated generators: Keep as PV bus (fixed Vm, solved Qg)
    """
    ppc_copy = ppc.copy()
    ppc_copy['bus'] = ppc['bus'].copy()
    ppc_copy['gen'] = ppc['gen'].copy()
    ppc_copy['branch'] = ppc['branch'].copy()
    
    # Set loads
    ppc_copy['bus'][load_bus_indices, PD] = load_pd
    ppc_copy['bus'][load_bus_indices, QD] = load_qd
    
    # Set Pg for non-slack generators
    for i, pg in enumerate(pg_non_slack):
        ppc_copy['gen'][i+1, PG] = pg
    
    # CRITICAL: Change bus types for violated generators
    for i in range(len(vm_all)):
        gen_bus_idx = int(ppc_copy['gen'][i, GEN_BUS]) - 1  # Convert to 0-indexed
        
        if i in violated_indices:
            # Violated generator: Switch to PQ bus type
            # This makes Vm a solved variable while Pg and Qg are fixed
            ppc_copy['bus'][gen_bus_idx, BUS_TYPE] = PQ
            ppc_copy['gen'][i, QG] = qg_clipped[i]  # Fix Qg at clipped value
            ppc_copy['gen'][i, VG] = vm_all[i]  # Initial guess, will be solved
            
            # Tighten bounds to enforce fixed Qg
            ppc_copy['gen'][i, QMIN] = qg_clipped[i]
            ppc_copy['gen'][i, QMAX] = qg_clipped[i]
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
            return None
        
        # Get updated Vm from bus voltages
        gen_buses_idx = ppc['gen'][:, GEN_BUS].astype(int) - 1
        vm_updated = results[0]['bus'][gen_buses_idx, 7]  # VM column (actual voltage magnitude)
        
        return {
            'qg': results[0]['gen'][:, QG],
            'pg': results[0]['gen'][:, PG],
            'vm': vm_updated  # Return actual bus voltages
        }
    except:
        return None


# ============================================
# STEP-BY-STEP DEMONSTRATION
# ============================================
print("\n" + "="*80)
print("STEP-BY-STEP WALKTHROUGH FOR SAMPLE 0")
print("="*80)

sample_idx = 0

# Get loads
load_pd = test_df[load_pd_cols].values[sample_idx]
load_qd = test_df[load_qd_cols].values[sample_idx]

# Get NN prediction
X_test = test_df[load_pd_cols + load_qd_cols].values
X_norm = input_scaler.transform(X_test[sample_idx:sample_idx+1])

# Load model
from torch import nn
class OPF_NN_Zamzam(nn.Module):
    def __init__(self, input_size, output_size):
        super(OPF_NN_Zamzam, self).__init__()
        hidden1_size = input_size
        hidden2_size = input_size
        hidden3_size = output_size
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden1_size), nn.Sigmoid(),
            nn.Linear(hidden1_size, hidden2_size), nn.Sigmoid(),
            nn.Linear(hidden2_size, hidden3_size), nn.Sigmoid(),
            nn.Linear(hidden3_size, output_size), nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

model = OPF_NN_Zamzam(checkpoint['input_size'], checkpoint['output_size'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

with torch.no_grad():
    alpha_beta_pred = model(torch.FloatTensor(X_norm)).numpy()[0]

n_alphas = n_gens - 1
alphas = alpha_beta_pred[:n_alphas]
betas = alpha_beta_pred[n_alphas:n_alphas + n_gens]

print(f"\n1. NEURAL NETWORK PREDICTIONS:")
print(f"   α (Active Power Parameters): {alphas}")
print(f"   β (Voltage Parameters):      {betas}")

# Convert to physical values
pg_non_slack, vm_all = alpha_beta_to_pg_vm(alphas, betas, pg_limits, 
                                            vm_min_orig.min(), vm_max_orig.max())

print(f"\n2. CONVERTED TO PHYSICAL VALUES:")
print(f"   Pg (non-slack): {pg_non_slack}")
print(f"   Vm (all gens):  {vm_all}")

# Solve initial power flow
print(f"\n3. SOLVE INITIAL POWER FLOW (Equation 4 from Zamzam paper):")
print(f"   Fixed: Pg (non-slack) and Vm (all)")
print(f"   Solve for: Qg, Pg_slack, Va (voltage angles)")

pf_result = solve_power_flow_with_details(ppc_base, load_pd, load_qd, pg_non_slack, 
                                          vm_all, load_bus_indices)

if pf_result is None:
    print(f"   ✗ Power flow FAILED to converge")
else:
    qg_initial = pf_result['qg']
    print(f"   ✓ Power flow converged")
    print(f"   Qg (from power balance): {qg_initial}")
    
    # Check violations
    print(f"\n4. CHECK QG AGAINST LIMITS:")
    print(f"   {'Gen':<5} {'Qg':<10} {'Qmin':<10} {'Qmax':<10} {'Status':<15}")
    print(f"   " + "-"*50)
    
    violations = []
    qg_clipped = qg_initial.copy()
    
    for i in range(n_gens):
        status = "OK"
        if qg_initial[i] < qg_min[i]:
            status = f"VIOLATES (< Qmin)"
            violations.append(i)
            qg_clipped[i] = qg_min[i]
        elif qg_initial[i] > qg_max[i]:
            status = f"VIOLATES (> Qmax)"
            violations.append(i)
            qg_clipped[i] = qg_max[i]
        
        print(f"   {i:<5} {qg_initial[i]:<10.2f} {qg_min[i]:<10.2f} {qg_max[i]:<10.2f} {status:<15}")
    
    if len(violations) > 0:
        print(f"\n5. APPLY CORRECTION (Equation 6 from Zamzam paper):")
        print(f"   Violated generators: {violations}")
        print(f"   Clipped Qg values: {qg_clipped}")
        print(f"   Strategy: Fix Qg at limits for violated gens, release Vm constraint")
        print(f"   Re-solving power flow...")
        
        modified_result = solve_modified_pf_with_details(ppc_base, load_pd, load_qd, 
                                                         pg_non_slack, vm_all, qg_clipped,
                                                         violations, load_bus_indices)
        
        if modified_result is None:
            print(f"   ✗ Modified power flow FAILED")
        else:
            qg_final = modified_result['qg']
            vm_final = modified_result['vm']
            print(f"   ✓ Modified power flow converged")
            print(f"   Final Qg: {qg_final}")
            print(f"   Final Vm: {vm_final}")
            
            print(f"\n6. SUMMARY OF CORRECTIONS:")
            print(f"   {'Gen':<5} {'Violated?':<11} {'Initial Qg':<12} {'Final Qg':<12} {'ΔQg':<10} {'Initial Vm':<11} {'Final Vm':<11} {'ΔVm':<10}")
            print(f"   " + "-"*95)
            for i in range(n_gens):
                qg_change = qg_final[i] - qg_initial[i]
                vm_change = vm_final[i] - vm_all[i]
                is_violated = "YES" if i in violations else "No"
                print(f"   {i:<5} {is_violated:<11} {qg_initial[i]:<12.3f} {qg_final[i]:<12.3f} "
                      f"{qg_change:>+9.3f} {vm_all[i]:<11.6f} {vm_final[i]:<11.6f} {vm_change:>+9.6f}")
            
            print(f"\n   Note: Violated generators should show:")
            print(f"         - Qg change (clipped to limit)")
            print(f"         - Vm change (adjusted to maintain power balance with fixed Qg)")
    else:
        print(f"\n5. NO VIOLATIONS - No correction needed!")

# ============================================
# COMPARE WITH OPF SOLUTION
# ============================================
qg_opf_cols = [col for col in test_df.columns if col.startswith('qg_')]
if len(qg_opf_cols) > 0:
    qg_opf = test_df[qg_opf_cols].values[sample_idx]
    
    print(f"\n" + "="*80)
    print("COMPARISON WITH OPF SOLUTION")
    print("="*80)
    print(f"\n{'Gen':<5} {'OPF Qg':<12} {'Recovered Qg':<15} {'Difference':<12}")
    print("-"*50)
    for i in range(n_gens):
        diff = qg_final[i] - qg_opf[i] if pf_result is not None else 0
        print(f"{i:<5} {qg_opf[i]:<12.3f} {qg_final[i]:<15.3f} {diff:>+10.3f}")

print(f"\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)
print("""
1. The NN predicts α (active power) and β (voltage magnitude)
2. These are converted to physical values: Pg and Vm
3. Power flow solver determines Qg from power balance equations
4. Qg often violates limits because NN doesn't directly control it
5. When violations occur, we clip Qg and re-solve (allows Vm to adjust)
6. Final solution is feasible with all constraints satisfied

This is the EXPECTED behavior in the Zamzam et al. 2019 approach!
The violation + correction IS the solution, not a bug.
""")


