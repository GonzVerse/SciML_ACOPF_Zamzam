"""
Deep Qg Correction Analysis

This script analyzes:
1. Which generators violate and why
2. How much Qg changes after correction
3. How voltage magnitudes change after correction
4. The relationship between predicted voltages and Qg violations
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pypower.api import case57
from pypower.idx_gen import GEN_BUS, QMIN, QMAX
import os
import json

# ============================================
# CONFIGURATION
# ============================================
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "Outputs_V3")
os.makedirs(output_dir, exist_ok=True)

print("="*70)
print("QG CORRECTION DEEP ANALYSIS")
print("="*70)

# ============================================
# LOAD DATA
# ============================================
print("\nLoading data...")

# Load recovered solutions
with open(os.path.join(output_dir, 'recovered_solutions_v3.pkl'), 'rb') as f:
    recovered_solutions = pickle.load(f)

# Load test data with true OPF solutions
test_df = pd.read_csv(os.path.join(output_dir, 'opf_case57_test_v2.csv'))

# Load generator limits
with open(os.path.join(output_dir, 'generator_limits.json'), 'r') as f:
    limits_data = json.load(f)

n_gens = limits_data['n_gens']
qg_limits = limits_data['qg_limits']
qg_min = np.array([lim[0] for lim in qg_limits])
qg_max = np.array([lim[1] for lim in qg_limits])

# Load base network
ppc = case57()
gen_buses = ppc['gen'][:, GEN_BUS].astype(int)

print(f"  Loaded {len(recovered_solutions)} recovered solutions")
print(f"  Generators: {n_gens}")
print(f"  Test samples: {len(test_df)}")

# ============================================
# EXTRACT QG DATA
# ============================================
print("\nExtracting Qg data...")

# True Qg from OPF
qg_cols = [f'qg_{i}' for i in range(n_gens)]
qg_opf = test_df[qg_cols].values  # Shape: (n_samples, n_gens)

# Recovered Qg (after correction)
qg_recovered = np.array([sol['qg'] for sol in recovered_solutions])  # Shape: (n_samples, n_gens)

# Recovered Vm
vm_recovered = np.array([sol['vm'] for sol in recovered_solutions])  # Shape: (n_samples, n_gens)

print(f"  Qg OPF shape: {qg_opf.shape}")
print(f"  Qg recovered shape: {qg_recovered.shape}")

# ============================================
# ANALYZE VIOLATIONS
# ============================================
print("\n" + "="*70)
print("VIOLATION ANALYSIS")
print("="*70)

# Check which samples/generators violate in OPF solution
violations_opf = np.zeros((len(qg_opf), n_gens), dtype=bool)
for i in range(n_gens):
    violations_opf[:, i] = (qg_opf[:, i] < qg_min[i]) | (qg_opf[:, i] > qg_max[i])

# Check violations in recovered solution (should be 0 after correction!)
violations_recovered = np.zeros((len(qg_recovered), n_gens), dtype=bool)
for i in range(n_gens):
    violations_recovered[:, i] = (qg_recovered[:, i] < qg_min[i] - 0.01) | (qg_recovered[:, i] > qg_max[i] + 0.01)

print("\nViolation Rates:")
print(f"{'Generator':<12} {'OPF Violations':<20} {'Recovered Violations':<20} {'Qg Limits':<30}")
print("-"*90)

for i in range(n_gens):
    opf_viol = np.sum(violations_opf[:, i])
    rec_viol = np.sum(violations_recovered[:, i])
    print(f"Gen {i:<8} {opf_viol}/{len(qg_opf)} ({100*opf_viol/len(qg_opf):>5.1f}%)    "
          f"{rec_viol}/{len(qg_recovered)} ({100*rec_viol/len(qg_recovered):>5.1f}%)    "
          f"[{qg_min[i]:>6.1f}, {qg_max[i]:>6.1f}]")

total_opf_viols = np.sum(violations_opf)
total_rec_viols = np.sum(violations_recovered)
print(f"\n{'TOTAL':<12} {total_opf_viols:<20} {total_rec_viols:<20}")

# ============================================
# ANALYZE CORRECTIONS
# ============================================
print("\n" + "="*70)
print("CORRECTION MAGNITUDE ANALYSIS")
print("="*70)

# Compute differences (this shows how Qg changed)
qg_diff = qg_recovered - qg_opf

print("\nQg Changes (Recovered - OPF):")
print(f"{'Generator':<12} {'Mean Change':<15} {'Max Change':<15} {'Std Dev':<15}")
print("-"*60)

for i in range(n_gens):
    mean_change = np.mean(qg_diff[:, i])
    max_change = np.max(np.abs(qg_diff[:, i]))
    std_change = np.std(qg_diff[:, i])
    print(f"Gen {i:<8} {mean_change:>10.3f} MVAr   {max_change:>10.3f} MVAr   {std_change:>10.3f} MVAr")

# ============================================
# ANALYZE WHICH DIRECTION VIOLATIONS OCCUR
# ============================================
print("\n" + "="*70)
print("VIOLATION DIRECTION ANALYSIS")
print("="*70)

print("\nOPF Solution Violations:")
print(f"{'Generator':<12} {'Below Qmin':<15} {'Above Qmax':<15} {'At Qmin':<15} {'At Qmax':<15}")
print("-"*75)

for i in range(n_gens):
    below = np.sum(qg_opf[:, i] < qg_min[i])
    above = np.sum(qg_opf[:, i] > qg_max[i])
    at_min = np.sum(np.abs(qg_opf[:, i] - qg_min[i]) < 0.01)
    at_max = np.sum(np.abs(qg_opf[:, i] - qg_max[i]) < 0.01)
    
    print(f"Gen {i:<8} {below:<15} {above:<15} {at_min:<15} {at_max:<15}")

# ============================================
# RELATIONSHIP BETWEEN Vm AND Qg
# ============================================
print("\n" + "="*70)
print("VOLTAGE-REACTIVE POWER RELATIONSHIP")
print("="*70)

# For each generator, compute correlation between Vm and Qg violations
print("\nCorrelation between Vm and Qg:")
print(f"{'Generator':<12} {'Corr(Vm, Qg_OPF)':<20} {'Avg Vm':<15}")
print("-"*50)

for i in range(n_gens):
    corr = np.corrcoef(vm_recovered[:, i], qg_opf[:, i])[0, 1]
    avg_vm = np.mean(vm_recovered[:, i])
    print(f"Gen {i:<8} {corr:>15.3f}       {avg_vm:>10.4f} p.u.")

# ============================================
# SAMPLE-BY-SAMPLE ANALYSIS
# ============================================
print("\n" + "="*70)
print("SAMPLE-BY-SAMPLE STATISTICS")
print("="*70)

violations_per_sample = np.sum(violations_opf, axis=1)

print(f"\nViolations per sample:")
print(f"  Min: {violations_per_sample.min()}")
print(f"  Max: {violations_per_sample.max()}")
print(f"  Mean: {violations_per_sample.mean():.2f}")
print(f"  Median: {np.median(violations_per_sample):.0f}")

# Which samples have the most violations?
worst_samples = np.argsort(violations_per_sample)[-5:][::-1]
print(f"\nTop 5 samples with most violations:")
for rank, idx in enumerate(worst_samples, 1):
    n_viols = violations_per_sample[idx]
    which_gens = np.where(violations_opf[idx])[0]
    print(f"  {rank}. Sample {idx}: {n_viols} violations (Gens: {which_gens})")

# ============================================
# DETAILED VISUALIZATION
# ============================================
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.35)

# Color scheme
colors = plt.cm.Set3(np.linspace(0, 1, n_gens))

# Plot 1: Qg OPF vs Recovered (scatter for each generator)
ax1 = fig.add_subplot(gs[0, :2])
for i in range(n_gens):
    ax1.scatter(qg_opf[:, i], qg_recovered[:, i], alpha=0.5, s=20, 
               label=f'Gen {i}', color=colors[i])
ax1.plot([qg_opf.min(), qg_opf.max()], [qg_opf.min(), qg_opf.max()], 
         'k--', linewidth=2, label='Perfect Match')
ax1.set_xlabel('Qg from OPF (MVAr)', fontsize=12)
ax1.set_ylabel('Qg Recovered (MVAr)', fontsize=12)
ax1.set_title('OPF vs Recovered Qg', fontsize=14, fontweight='bold')
ax1.legend(fontsize=8, ncol=2)
ax1.grid(True, alpha=0.3)

# Plot 2: Qg differences histogram
ax2 = fig.add_subplot(gs[0, 2:])
qg_diff_flat = qg_diff.flatten()
ax2.hist(qg_diff_flat, bins=50, alpha=0.7, edgecolor='black')
ax2.axvline(0, color='r', linestyle='--', linewidth=2, label='No Change')
ax2.set_xlabel('Qg Change (Recovered - OPF) [MVAr]', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('Distribution of Qg Corrections', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3-9: Individual generator Qg trajectories
for i in range(n_gens):
    row = 1 + i // 4
    col = i % 4
    ax = fig.add_subplot(gs[row, col])
    
    # Plot first 30 samples
    n_plot = min(30, len(qg_opf))
    x = np.arange(n_plot)
    
    ax.plot(x, qg_opf[:n_plot, i], 'o-', label='OPF', markersize=4, alpha=0.7)
    ax.plot(x, qg_recovered[:n_plot, i], 's-', label='Recovered', markersize=4, alpha=0.7)
    ax.axhline(qg_max[i], color='r', linestyle='--', linewidth=1, label=f'Qmax')
    ax.axhline(qg_min[i], color='b', linestyle='--', linewidth=1, label=f'Qmin')
    
    # Highlight violations
    viols = violations_opf[:n_plot, i]
    if np.any(viols):
        ax.scatter(x[viols], qg_opf[viols, i], color='red', s=100, 
                  marker='x', linewidths=3, label='Violation', zorder=5)
    
    ax.set_xlabel('Sample', fontsize=9)
    ax.set_ylabel('Qg (MVAr)', fontsize=9)
    ax.set_title(f'Generator {i} (Bus {gen_buses[i]})', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

# Plot: Vm vs Qg relationship (last row)
ax_vm = fig.add_subplot(gs[3, :2])
for i in range(n_gens):
    ax_vm.scatter(vm_recovered[:, i], qg_opf[:, i], alpha=0.5, s=20,
                 label=f'Gen {i}', color=colors[i])
ax_vm.set_xlabel('Voltage Magnitude (p.u.)', fontsize=12)
ax_vm.set_ylabel('Qg from OPF (MVAr)', fontsize=12)
ax_vm.set_title('Voltage vs Reactive Power Relationship', fontsize=14, fontweight='bold')
ax_vm.legend(fontsize=8, ncol=2)
ax_vm.grid(True, alpha=0.3)

# Plot: Violation heatmap
ax_heat = fig.add_subplot(gs[3, 2:])
im = ax_heat.imshow(violations_opf.T.astype(int), cmap='RdYlGn_r', aspect='auto', interpolation='nearest')
ax_heat.set_yticks(range(n_gens))
ax_heat.set_yticklabels([f'Gen {i}' for i in range(n_gens)])
ax_heat.set_xlabel('Sample Index', fontsize=12)
ax_heat.set_ylabel('Generator', fontsize=12)
ax_heat.set_title('Violation Pattern Across Samples', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax_heat, label='Violation')

plt.suptitle('Detailed Qg Correction Analysis', fontsize=18, fontweight='bold', y=0.995)

plot_path = os.path.join(output_dir, 'qg_correction_detailed_analysis.png')
plt.savefig(plot_path, dpi=200, bbox_inches='tight')
print(f"\n✓ Saved detailed visualization: {plot_path}")

# ============================================
# SUMMARY STATISTICS TABLE
# ============================================
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

summary_data = []
for i in range(n_gens):
    summary_data.append({
        'Generator': i,
        'Bus': int(gen_buses[i]),
        'Qmin': qg_min[i],
        'Qmax': qg_max[i],
        'OPF_Violations': np.sum(violations_opf[:, i]),
        'Violation_Rate_%': 100 * np.sum(violations_opf[:, i]) / len(qg_opf),
        'Below_Qmin': np.sum(qg_opf[:, i] < qg_min[i]),
        'Above_Qmax': np.sum(qg_opf[:, i] > qg_max[i]),
        'Mean_Qg_OPF': np.mean(qg_opf[:, i]),
        'Mean_Qg_Recovered': np.mean(qg_recovered[:, i]),
        'Mean_Correction': np.mean(qg_diff[:, i]),
        'Max_Correction': np.max(np.abs(qg_diff[:, i])),
        'Avg_Vm': np.mean(vm_recovered[:, i]),
        'Corr_Vm_Qg': np.corrcoef(vm_recovered[:, i], qg_opf[:, i])[0, 1]
    })

summary_df = pd.DataFrame(summary_data)
summary_path = os.path.join(output_dir, 'qg_correction_summary.csv')
summary_df.to_csv(summary_path, index=False)
print(f"\n✓ Saved summary statistics: {summary_path}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
print(f"\nKey Findings:")
print(f"  1. Total OPF violations: {total_opf_viols} (before correction)")
print(f"  2. Total recovered violations: {total_rec_viols} (after correction)")
print(f"  3. Correction success rate: {100*(1-total_rec_viols/max(total_opf_viols,1)):.1f}%")
print(f"  4. Most problematic generator: Gen {np.argmax(np.sum(violations_opf, axis=0))}")
print(f"  5. Average violations per sample: {violations_per_sample.mean():.2f}")

plt.show()
