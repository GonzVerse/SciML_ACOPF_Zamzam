"""
Sensitivity Analysis: Qg Filtering Threshold vs Violation Rates

Tests different filtering thresholds (0%, 10%, 30%, 50%, 70%) to determine
optimal balance between dataset size and constraint satisfaction quality.

For each threshold:
1. Filter training data based on % of generators at Qg limits
2. Train neural network
3. Run power flow recovery on test set
4. Measure δ_q (reactive power violation magnitude)
5. Compare results
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
import os
import json
from pypower.api import case57, runpf, ppoption
from pypower.idx_gen import PG, QG, VG, GEN_BUS
from pypower.idx_bus import PD, QD, VM, VA
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "Outputs_V3")
os.makedirs(output_dir, exist_ok=True)

# Thresholds to test (% of generators allowed at limits before filtering)
# 0.10 = remove if >10% gens at limits, 1.0 = no filtering
FILTER_THRESHOLDS = [0.10, 0.30, 0.50, 0.70, 1.0]

# Training parameters
BATCH_SIZE = 256
LEARNING_RATE = 0.001
NUM_EPOCHS = 300  # Match v3_improved for fair comparison
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*70)
print("SENSITIVITY ANALYSIS: Qg Filtering Threshold vs Violation Rates")
print("="*70)
print(f"Device: {DEVICE}")
print(f"Testing thresholds: {[f'{t*100:.0f}%' for t in FILTER_THRESHOLDS]}")
print(f"Epochs per run: {NUM_EPOCHS}")
print("="*70)

# ============================================
# LOAD GENERATOR LIMITS AND BASE NETWORK
# ============================================
limits_path = os.path.join(output_dir, 'generator_limits.json')
with open(limits_path, 'r') as f:
    limits_data = json.load(f)

n_gens = limits_data['n_gens']
n_loads = limits_data['n_loads']
pg_limits = limits_data['pg_limits']
qg_limits = limits_data['qg_limits']
vm_min_orig = np.array(limits_data['vm_min_orig'])
vm_max_orig = np.array(limits_data['vm_max_orig'])

ppc_base = case57()
gen_buses = ppc_base['gen'][:, GEN_BUS].astype(int)

# ============================================
# DATASET CLASS WITH CONFIGURABLE FILTERING
# ============================================
class OPFDataset_Filtered(Dataset):
    def __init__(self, csv_path, input_scaler=None, qg_limits=None, filter_threshold=0.5):
        """
        Args:
            filter_threshold: Remove samples where >threshold fraction of gens are at limits
                            0.0 = remove all samples with ANY gen at limits
                            0.5 = remove samples with >50% gens at limits
                            1.0 = no filtering (keep all samples)
        """
        df = pd.read_csv(csv_path)
        
        # Apply quality filter
        if qg_limits is not None and filter_threshold < 1.0:
            df = self._filter_overly_constrained(df, qg_limits, filter_threshold)
        
        # Check if filtering removed all samples
        if len(df) == 0:
            raise ValueError(f"Filtering with threshold {filter_threshold} removed all samples!")
        
        # Separate inputs and outputs
        load_pd_cols = [col for col in df.columns if col.startswith('pd_')]
        load_qd_cols = [col for col in df.columns if col.startswith('qd_')]
        alpha_cols = [col for col in df.columns if col.startswith('alpha_')]
        beta_cols = [col for col in df.columns if col.startswith('beta_')]
        
        self.X = df[load_pd_cols + load_qd_cols].values
        self.y = df[alpha_cols + beta_cols].values
        
        # Normalize
        if input_scaler is None:
            self.input_scaler = StandardScaler()
            self.X = self.input_scaler.fit_transform(self.X)
        else:
            self.input_scaler = input_scaler
            self.X = self.input_scaler.transform(self.X)
        
        self.X = torch.FloatTensor(self.X)
        self.y = torch.FloatTensor(self.y)
        self.num_samples = len(self.X)
    
    def _filter_overly_constrained(self, df, qg_limits, threshold):
        """Remove samples where >threshold of generators are at Qg limits"""
        qg_cols = [col for col in df.columns if col.startswith('qg_')]
        if not qg_cols:
            return df
        
        qg_data = df[qg_cols].values
        n_gens = len(qg_cols)
        
        # Count generators at limits for each sample
        at_limits_count = np.zeros(len(df))
        
        for i, (qg_min, qg_max) in enumerate(qg_limits):
            at_min = np.abs(qg_data[:, i] - qg_min) < 0.01
            at_max = np.abs(qg_data[:, i] - qg_max) < 0.01
            at_limits_count += (at_min | at_max)
        
        # Keep samples where fraction at limits < threshold
        pct_at_limits = at_limits_count / n_gens
        keep_mask = pct_at_limits < threshold
        filtered_df = df[keep_mask].copy()
        
        removed = len(df) - len(filtered_df)
        return filtered_df
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================
# NEURAL NETWORK MODEL
# ============================================
class OPF_NN(nn.Module):
    def __init__(self, input_size, output_size):
        super(OPF_NN, self).__init__()
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

# ============================================
# POWER FLOW RECOVERY FUNCTION
# ============================================
def recover_power_flow(alpha, beta, pg_limits, vm_min_orig, vm_max_orig, gen_buses, lambda_margin, ppc_template, loads_pd, loads_qd):
    """
    Recover full power flow solution from α, β predictions
    Returns: (success, pg_slack, qg_violations, qg_violation_magnitude)
    """
    ppc = ppc_template.copy()
    ppc['bus'] = ppc_template['bus'].copy()
    ppc['gen'] = ppc_template['gen'].copy()
    ppc['branch'] = ppc_template['branch'].copy()
    
    # Set loads
    load_mask = ppc_template['bus'][:, PD] > 0
    load_indices = np.where(load_mask)[0]
    ppc['bus'][load_indices, PD] = loads_pd
    ppc['bus'][load_indices, QD] = loads_qd
    
    # Convert α → Pg (excluding slack)
    pg = np.zeros(n_gens)
    for i in range(1, n_gens):
        p_min, p_max = pg_limits[i]
        pg[i] = p_min + alpha[i-1] * (p_max - p_min)
    
    # Convert β → Vm (all generators)
    vm = np.zeros(n_gens)
    for i in range(n_gens):
        v_min = vm_min_orig[gen_buses[i]] + lambda_margin
        v_max = vm_max_orig[gen_buses[i]] - lambda_margin
        vm[i] = v_min + beta[i] * (v_max - v_min)
    
    # Set generator outputs (except slack Pg)
    ppc['gen'][1:, PG] = pg[1:]
    ppc['gen'][:, VG] = vm
    
    # Run power flow
    ppopt = ppoption(VERBOSE=0, OUT_ALL=0)
    results = runpf(ppc, ppopt)
    
    # runpf returns (results_dict, success_flag)
    if isinstance(results, tuple):
        results, success = results
    else:
        success = results.get('success', False)
    
    if not success:
        return False, None, None, None
    
    # Extract results
    pg_slack = results['gen'][0, PG]
    qg_all = results['gen'][:, QG]
    
    # Check Qg violations
    violations = []
    violation_mags = []
    
    for i in range(n_gens):
        qg_min, qg_max = qg_limits[i]
        if qg_all[i] < qg_min:
            violations.append(i)
            violation_mags.append(qg_min - qg_all[i])
        elif qg_all[i] > qg_max:
            violations.append(i)
            violation_mags.append(qg_all[i] - qg_max)
    
    qg_violation_magnitude = np.sum(violation_mags) if violation_mags else 0.0
    
    return True, pg_slack, len(violations), qg_violation_magnitude

# ============================================
# TRAINING FUNCTION
# ============================================
def train_model(train_loader, val_loader, input_size, output_size, epochs):
    """Train model and return best validation loss"""
    model = OPF_NN(input_size, output_size).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(DEVICE)
            batch_targets = batch_targets.to(DEVICE)
            
            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_inputs, batch_targets in val_loader:
                batch_inputs = batch_inputs.to(DEVICE)
                batch_targets = batch_targets.to(DEVICE)
                predictions = model(batch_inputs)
                loss = criterion(predictions, batch_targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        # Print progress every 50 epochs
        if (epoch + 1) % 50 == 0:
            print(f"    Epoch [{epoch+1}/{epochs}] - Val Loss: {val_loss:.6f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model, best_val_loss

# ============================================
# EVALUATION FUNCTION
# ============================================
def evaluate_model(model, test_dataset, ppc_base, test_file):
    """Evaluate model on test set with power flow recovery"""
    model.eval()
    
    results = {
        'success_count': 0,
        'qg_violations': [],
        'qg_violation_magnitudes': []
    }
    
    # Get test data
    test_df = pd.read_csv(test_file)
    
    load_pd_cols = [col for col in test_df.columns if col.startswith('pd_')]
    load_qd_cols = [col for col in test_df.columns if col.startswith('qd_')]
    
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            # Print progress every 20 samples
            if (idx + 1) % 20 == 0:
                print(f"    Evaluated {idx+1}/{len(test_dataset)} samples...", end='\r')
            
            inputs = test_dataset.X[idx:idx+1].to(DEVICE)
            predictions = model(inputs).cpu().numpy()[0]
            
            # Get loads
            loads_pd = test_df.iloc[idx][load_pd_cols].values
            loads_qd = test_df.iloc[idx][load_qd_cols].values
            
            # Split predictions
            alpha_pred = predictions[:n_gens-1]
            beta_pred = predictions[n_gens-1:]
            
            # Recover power flow
            success, pg_slack, qg_viols, qg_viol_mag = recover_power_flow(
                alpha_pred, beta_pred, pg_limits, vm_min_orig, vm_max_orig,
                gen_buses, limits_data['lambda'], ppc_base, loads_pd, loads_qd
            )
            
            if success:
                results['success_count'] += 1
                results['qg_violations'].append(qg_viols)
                results['qg_violation_magnitudes'].append(qg_viol_mag)
    
    print()  # New line after progress updates
    
    # Compute metrics
    success_rate = 100 * results['success_count'] / len(test_dataset)
    avg_qg_violations = np.mean(results['qg_violations'])
    avg_delta_q = np.mean(results['qg_violation_magnitudes'])
    
    return {
        'success_rate': success_rate,
        'avg_qg_violations': avg_qg_violations,
        'avg_delta_q': avg_delta_q,
        'num_samples': results['success_count']
    }

# ============================================
# MAIN SENSITIVITY ANALYSIS
# ============================================
if __name__ == '__main__':
    train_file = os.path.join(output_dir, 'opf_case57_train_v3.csv')
    val_file = os.path.join(output_dir, 'opf_case57_val_v3.csv')
    test_file = os.path.join(output_dir, 'opf_case57_test_v3.csv')
    
    results_summary = []
    
    for threshold_idx, threshold in enumerate(FILTER_THRESHOLDS):
        print(f"\n{'='*70}")
        print(f"TESTING THRESHOLD {threshold_idx+1}/{len(FILTER_THRESHOLDS)}: {threshold*100:.0f}% (filter if >{threshold*100:.0f}% gens at limits)")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        # Load datasets with current threshold
        print(f"\nLoading datasets with threshold={threshold:.2f}...")
        try:
            train_dataset = OPFDataset_Filtered(
                train_file, qg_limits=qg_limits, filter_threshold=threshold
            )
            val_dataset = OPFDataset_Filtered(
                val_file, input_scaler=train_dataset.input_scaler,
                qg_limits=qg_limits, filter_threshold=threshold
            )
            test_dataset = OPFDataset_Filtered(
                test_file, input_scaler=train_dataset.input_scaler,
                qg_limits=qg_limits, filter_threshold=1.0  # Don't filter test
            )
        except ValueError as e:
            print(f"  ⚠️ SKIPPING: {e}")
            continue
        
        print(f"  Training samples: {train_dataset.num_samples}")
        print(f"  Validation samples: {val_dataset.num_samples}")
        print(f"  Test samples: {test_dataset.num_samples}")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Train model
        print(f"\nTraining model for {NUM_EPOCHS} epochs...")
        input_size = train_dataset.X.shape[1]
        output_size = train_dataset.y.shape[1]
        
        model, best_val_loss = train_model(train_loader, val_loader, input_size, output_size, NUM_EPOCHS)
        print(f"  Best validation loss: {best_val_loss:.6f}")
        
        # Evaluate on test set
        print(f"\nEvaluating on test set...")
        eval_results = evaluate_model(model, test_dataset, ppc_base, test_file)
        
        elapsed_time = time.time() - start_time
        
        # Store results
        result = {
            'threshold': threshold,
            'threshold_pct': threshold * 100,
            'train_samples': train_dataset.num_samples,
            'val_loss': best_val_loss,
            'success_rate': eval_results['success_rate'],
            'avg_qg_violations': eval_results['avg_qg_violations'],
            'avg_delta_q': eval_results['avg_delta_q'],
            'time_seconds': elapsed_time
        }
        results_summary.append(result)
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"RESULTS FOR THRESHOLD {threshold*100:.0f}%:")
        print(f"  Training samples: {result['train_samples']}")
        print(f"  Validation loss: {result['val_loss']:.6f}")
        print(f"  Success rate: {result['success_rate']:.1f}%")
        print(f"  Avg Qg violations per sample: {result['avg_qg_violations']:.2f}")
        print(f"  Avg δ_q (MVAr): {result['avg_delta_q']:.3f}")
        print(f"  Time: {result['time_seconds']:.1f}s")
        print(f"{'='*70}")
    
    # ============================================
    # SAVE RESULTS
    # ============================================
    results_df = pd.DataFrame(results_summary)
    results_path = os.path.join(output_dir, 'sensitivity_analysis_qg_filtering.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n✅ Results saved to: {results_path}")
    
    # ============================================
    # PLOT RESULTS
    # ============================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    thresholds_pct = results_df['threshold_pct'].values
    
    # Plot 1: Training samples vs threshold
    ax1 = axes[0, 0]
    ax1.plot(thresholds_pct, results_df['train_samples'], 'o-', linewidth=2, markersize=8, color='blue')
    ax1.set_xlabel('Filter Threshold (%)', fontsize=12)
    ax1.set_ylabel('Training Samples', fontsize=12)
    ax1.set_title('Dataset Size vs Filter Threshold', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: δ_q vs threshold
    ax2 = axes[0, 1]
    ax2.plot(thresholds_pct, results_df['avg_delta_q'], 'o-', linewidth=2, markersize=8, color='red')
    ax2.axhline(y=1.58, color='green', linestyle='--', label='Zamzam Benchmark (1.58 MVAr)')
    ax2.set_xlabel('Filter Threshold (%)', fontsize=12)
    ax2.set_ylabel('Avg δ_q (MVAr)', fontsize=12)
    ax2.set_title('Reactive Power Violation vs Filter Threshold', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Avg Qg violations vs threshold
    ax3 = axes[1, 0]
    ax3.plot(thresholds_pct, results_df['avg_qg_violations'], 'o-', linewidth=2, markersize=8, color='orange')
    ax3.set_xlabel('Filter Threshold (%)', fontsize=12)
    ax3.set_ylabel('Avg Qg Violations per Sample', fontsize=12)
    ax3.set_title('Constraint Violations vs Filter Threshold', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Validation loss vs threshold
    ax4 = axes[1, 1]
    ax4.plot(thresholds_pct, results_df['val_loss'], 'o-', linewidth=2, markersize=8, color='purple')
    ax4.set_xlabel('Filter Threshold (%)', fontsize=12)
    ax4.set_ylabel('Validation MSE Loss', fontsize=12)
    ax4.set_title('Model Performance vs Filter Threshold', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'sensitivity_analysis_plots.png')
    plt.savefig(plot_path, dpi=150)
    print(f"✅ Plots saved to: {plot_path}")
    
    # ============================================
    # SUMMARY TABLE
    # ============================================
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(results_df.to_string(index=False))
    print(f"{'='*70}")
    
    # Find optimal threshold
    optimal_idx = results_df['avg_delta_q'].idxmin()
    optimal_row = results_df.iloc[optimal_idx]
    
    print(f"\n🎯 OPTIMAL THRESHOLD: {optimal_row['threshold_pct']:.0f}%")
    print(f"   Training samples: {optimal_row['train_samples']}")
    print(f"   δ_q: {optimal_row['avg_delta_q']:.3f} MVAr")
    print(f"   Qg violations: {optimal_row['avg_qg_violations']:.2f} per sample")
    print(f"   Success rate: {optimal_row['success_rate']:.1f}%")
    
    plt.show()

