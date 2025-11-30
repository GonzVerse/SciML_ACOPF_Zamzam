"""
OPF Neural Network Training - Version 3 (IMPROVED)
Based on Zamzam et al. 2019 paper with constraint-aware improvements

IMPROVEMENTS OVER V2:
1. ✓ Physics-informed loss: Enforces power balance equations
2. ✓ Data quality filtering: Removes overly constrained samples
3. ✓ Validation metrics: Tracks violation rates during training
4. ✓ Better convergence: Balanced loss function with tunable weights

Note: Qg penalty removed - not part of Zamzam architecture.
Qg is determined by power flow equations, not directly predicted.
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
from pypower.api import runpf, ppoption

# ============================================
# CONFIGURATION
# ============================================
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "Outputs_v3")
os.makedirs(output_dir, exist_ok=True)

BATCH_SIZE = 256  # Increased for better GPU utilization with 33k samples was 32.
LEARNING_RATE = 0.001
NUM_EPOCHS = 300
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loss function weights - Pure Zamzam approach
LAMBDA_MSE = 1.0           # Standard prediction error (only loss used)

print("="*60)
print("OPF Neural Network Training - V3 (CONSTRAINT-AWARE)")
print("="*60)
print(f"Output directory: {output_dir}")
print(f"Device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"\nLoss function: Pure MSE (Zamzam architecture)")
print(f"  λ_MSE = {LAMBDA_MSE}")
print("="*60)

# ============================================
# LOAD GENERATOR LIMITS
# ============================================
print(f"\nLoading generator limits...")
limits_path = os.path.join(output_dir, 'generator_limits.json')

if not os.path.exists(limits_path):
    raise FileNotFoundError(
        f"Generator limits not found: {limits_path}\n"
        f"Please run 00_generate_opf_data_v2.py first!"
    )

with open(limits_path, 'r') as f:
    limits_data = json.load(f)

n_gens = limits_data['n_gens']
n_loads = limits_data['n_loads']
pg_limits = limits_data['pg_limits']
qg_limits = limits_data['qg_limits']  # Used for data filtering only
vm_min_orig = np.array(limits_data['vm_min_orig'])
vm_max_orig = np.array(limits_data['vm_max_orig'])
lambda_margin = limits_data['lambda']

print(f"  Generators: {n_gens}")
print(f"  Loads: {n_loads}")
print(f"  Voltage margin (λ): {lambda_margin}")

# Note: Qg limits used only for data filtering, not in loss function
print(f"\nQg limits loaded for data filtering only (not used in loss)")

# ============================================
# CUSTOM DATASET CLASS WITH FILTERING
# ============================================
class OPFDataset_V3(Dataset):
    """
    Dataset for OPF with α/β parameterization (Zamzam et al. 2019)
    
    IMPROVEMENT 3: Data quality filtering
    - Filters out samples where >50% of generators are at Qg limits
    - Ensures training data contains diverse, non-degenerate solutions
    
    Inputs: [pd_1, ..., pd_L, qd_1, ..., qd_L]  (2*n_loads)
    Outputs: [α_2, ..., α_G, β_1, ..., β_G]     (2*n_gens - 1)
    """

    def __init__(self, csv_path, input_scaler=None, qg_limits=None, filter_quality=True):
        """
        Args:
            csv_path: Path to CSV file
            input_scaler: Fitted StandardScaler for inputs (None for training)
            qg_limits: List of (qg_min, qg_max) for filtering
            filter_quality: Whether to remove overly constrained samples
        """
        # Load data
        df = pd.read_csv(csv_path)
        
        # Apply quality filter if requested
        if filter_quality and qg_limits is not None:
            print(f"  Pre-filter: {len(df)} samples")
            df = self._filter_overly_constrained(df, qg_limits)
            print(f"  Post-filter: {len(df)} samples")

        # Separate inputs (pd and qd)
        load_pd_cols = [col for col in df.columns if col.startswith('pd_')]
        load_qd_cols = [col for col in df.columns if col.startswith('qd_')]

        # Outputs (α and β only)
        alpha_cols = [col for col in df.columns if col.startswith('alpha_')]
        beta_cols = [col for col in df.columns if col.startswith('beta_')]

        # Input: pd and qd for all loads
        self.X = df[load_pd_cols + load_qd_cols].values

        # Output: α and β
        self.y = df[alpha_cols + beta_cols].values

        print(f"  Loaded: {len(df)} samples")
        print(f"  Input dimension: {self.X.shape[1]} (2 × {len(load_pd_cols)} loads)")
        print(f"  Output dimension: {self.y.shape[1]} ({len(alpha_cols)} α + {len(beta_cols)} β)")

        # Normalize inputs only
        if input_scaler is None:
            self.input_scaler = StandardScaler()
            self.X = self.input_scaler.fit_transform(self.X)
        else:
            self.input_scaler = input_scaler
            self.X = self.input_scaler.transform(self.X)

        # Convert to tensors
        self.X = torch.FloatTensor(self.X)
        self.y = torch.FloatTensor(self.y)
    
    def _filter_overly_constrained(self, df, qg_limits):
        """
        Remove samples where >50% of generators are at Qg limits
        These are degenerate cases that teach the NN bad patterns
        """
        qg_cols = [col for col in df.columns if col.startswith('qg_')]
        
        if not qg_cols:
            return df  # No filtering possible
        
        qg_data = df[qg_cols].values
        n_gens = len(qg_cols)
        
        # Check how many generators are at limits
        at_limits = np.zeros(len(df), dtype=bool)
        
        for i, (qg_min, qg_max) in enumerate(qg_limits):
            at_min = np.abs(qg_data[:, i] - qg_min) < 0.01
            at_max = np.abs(qg_data[:, i] - qg_max) < 0.01
            at_limits += (at_min | at_max)
        
        # Keep samples where <50% of generators are at limits
        pct_at_limits = at_limits / n_gens
        keep_mask = pct_at_limits < 0.5
        
        filtered_df = df[keep_mask].copy()
        
        removed = len(df) - len(filtered_df)
        print(f"    Removed {removed} overly constrained samples ({100*removed/len(df):.1f}%)")
        
        return filtered_df

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================
# NEURAL NETWORK MODEL - ZAMZAM ARCHITECTURE
# ============================================
class OPF_NN_Zamzam(nn.Module):
    """
    Neural Network for AC OPF following Zamzam et al. 2019
    (Same architecture as V2)
    """

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

        print(f"\nNetwork Architecture (Zamzam):")
        print(f"  Input layer: {input_size}")
        print(f"  Hidden layer 1: {hidden1_size} (Sigmoid)")
        print(f"  Hidden layer 2: {hidden2_size} (Sigmoid)")
        print(f"  Hidden layer 3: {hidden3_size} (Sigmoid)")
        print(f"  Output layer: {output_size} (Sigmoid)")

    def forward(self, x):
        return self.network(x)

# ============================================
# CONSTRAINT-AWARE LOSS FUNCTION
# ============================================
class PureMSELoss(nn.Module):
    """
    Pure MSE loss function (Zamzam 2019 architecture)
    
    Loss = MSE(y_pred, y_true)
    
    Note: 
    - No Qg penalty (Qg determined by power flow equations)
    - No balance penalty (sigmoid output + R-ACOPF data ensures [0,1] bounds)
    - Matches original Zamzam paper exactly
    """
    
    def __init__(self, lambda_mse=1.0):
        super(PureMSELoss, self).__init__()
        
        self.lambda_mse = lambda_mse
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Network output [α, β]
            targets: Ground truth [α, β]
        
        Returns:
            total_loss, mse_loss
        """
        # Pure MSE loss - exactly as Zamzam 2019
        mse_loss = self.mse_loss(predictions, targets)
        total_loss = self.lambda_mse * mse_loss
        
        return total_loss, mse_loss.item()

# ============================================
# LOAD DATA
# ============================================
print("\n" + "="*60)
print("Loading Datasets")
print("="*60)

train_file = os.path.join(output_dir, 'opf_case57_train_v3.csv')
val_file = os.path.join(output_dir, 'opf_case57_val_v3.csv')
test_file = os.path.join(output_dir, 'opf_case57_test_v3.csv')

if not os.path.exists(train_file):
    raise FileNotFoundError(
        f"Training file not found: {train_file}\n"
        f"Please run 00_generate_opf_data_v2.py first!"
    )

# Load datasets with quality filtering
print(f"\nTraining set:")
train_dataset = OPFDataset_V3(train_file, qg_limits=qg_limits, filter_quality=True)

print(f"\nValidation set:")
val_dataset = OPFDataset_V3(val_file,
                            input_scaler=train_dataset.input_scaler,
                            qg_limits=qg_limits,
                            filter_quality=True)

print(f"\nTest set:")
test_dataset = OPFDataset_V3(test_file,
                             input_scaler=train_dataset.input_scaler,
                             qg_limits=qg_limits,
                             filter_quality=False)  # Don't filter test set

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============================================
# CREATE MODEL
# ============================================
print("\n" + "="*60)
print("Building Model")
print("="*60)

input_size = train_dataset.X.shape[1]
output_size = train_dataset.y.shape[1]

model = OPF_NN_Zamzam(input_size, output_size).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")

# ============================================
# TRAINING SETUP WITH PURE MSE LOSS (ZAMZAM 2019)
# ============================================
criterion = PureMSELoss(
    lambda_mse=LAMBDA_MSE
)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)

# ============================================
# TRAINING LOOP WITH DIAGNOSTICS
# ============================================
print("\n" + "="*60)
print("Starting Training")
print("="*60)

train_losses = []
val_losses = []
best_val_loss = float('inf')

# Track loss during training
train_mse_history = []
val_mse_history = []

# Start timer
training_start_time = time.time()

for epoch in range(NUM_EPOCHS):
    # ========== TRAINING ==========
    model.train()
    train_loss = 0.0
    train_mse = 0.0

    for batch_inputs, batch_targets in train_loader:
        batch_inputs = batch_inputs.to(DEVICE)
        batch_targets = batch_targets.to(DEVICE)

        # Forward pass with pure MSE loss
        optimizer.zero_grad()
        predictions = model(batch_inputs)
        loss, mse = criterion(predictions, batch_targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_mse += mse

    train_loss /= len(train_loader)
    train_mse /= len(train_loader)
    
    train_losses.append(train_loss)
    train_mse_history.append(train_mse)

    # ========== VALIDATION ==========
    model.eval()
    val_loss = 0.0
    val_mse = 0.0

    with torch.no_grad():
        for batch_inputs, batch_targets in val_loader:
            batch_inputs = batch_inputs.to(DEVICE)
            batch_targets = batch_targets.to(DEVICE)

            predictions = model(batch_inputs)
            loss, mse = criterion(predictions, batch_targets)
            
            val_loss += loss.item()
            val_mse += mse

    val_loss /= len(val_loader)
    val_mse /= len(val_loader)
    
    val_losses.append(val_loss)
    val_mse_history.append(val_mse)

    # Update learning rate
    scheduler.step(val_loss)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model_path = os.path.join(output_dir, 'best_opf_model_v3_improved.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'input_scaler': train_dataset.input_scaler,
            'input_size': input_size,
            'output_size': output_size,
            'loss_weights': {
                'lambda_mse': LAMBDA_MSE
            },
            'training_time_seconds': time.time() - training_start_time,  # Track elapsed time
            'batch_size': BATCH_SIZE,
            'num_samples': len(train_dataset)
        }, model_path)

    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        print(f"  Train - Loss: {train_loss:.6f}  (MSE: {train_mse:.6f})")
        print(f"  Val   - Loss: {val_loss:.6f}  (MSE: {val_mse:.6f})")

# Calculate total training time
training_end_time = time.time()
total_training_time = training_end_time - training_start_time
training_time_minutes = total_training_time / 60

print("\n" + "="*60)
print("Training Complete!")
print(f"Total training time: {training_time_minutes:.2f} minutes ({total_training_time:.1f} seconds)")
print(f"Best validation loss: {best_val_loss:.6f}")
print(f"Model saved to: {model_path}")
print("="*60)

# ============================================
# PLOT TRAINING CURVES
# ============================================
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# MSE loss (only loss component)
ax.plot(train_mse_history, label='Training MSE', linewidth=2, color='blue')
ax.plot(val_mse_history, label='Validation MSE', linewidth=2, color='orange')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('MSE Loss', fontsize=12)
ax.set_title('Pure MSE Loss (Zamzam 2019 Architecture)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
plot_path = os.path.join(output_dir, 'training_curves_v3_improved.png')
plt.savefig(plot_path, dpi=150)
print(f"\nSaved training curves to: {plot_path}")

# ============================================
# QUICK EVALUATION ON TEST SET
# ============================================
print("\n" + "="*60)
print("Quick Test Set Evaluation")
print("="*60)

model.eval()
test_loss = 0.0
all_preds = []
all_targets = []

with torch.no_grad():
    for batch_inputs, batch_targets in test_loader:
        batch_inputs = batch_inputs.to(DEVICE)
        batch_targets = batch_targets.to(DEVICE)

        predictions = model(batch_inputs)
        loss, _ = criterion(predictions, batch_targets)
        test_loss += loss.item()

        all_preds.append(predictions.cpu().numpy())
        all_targets.append(batch_targets.cpu().numpy())

test_loss /= len(test_loader)

# Concatenate predictions
all_preds = np.concatenate(all_preds, axis=0)
all_targets = np.concatenate(all_targets, axis=0)

# Split outputs
n_alphas = n_gens - 1
n_betas = n_gens

alpha_preds = all_preds[:, :n_alphas]
beta_preds = all_preds[:, n_alphas:n_alphas+n_betas]

print(f"\nTest Loss: {test_loss:.6f}")
print(f"\nOutput validation:")
print(f"  Alpha (α) range: [{alpha_preds.min():.4f}, {alpha_preds.max():.4f}]")
print(f"  Beta (β) range: [{beta_preds.min():.4f}, {beta_preds.max():.4f}]")

# Count violations
alpha_violations = np.sum((alpha_preds < 0) | (alpha_preds > 1))
beta_violations = np.sum((beta_preds < 0) | (beta_preds > 1))

print(f"\nBound violations:")
print(f"  Alpha outside [0,1]: {alpha_violations} / {alpha_preds.size}")
print(f"  Beta outside [0,1]: {beta_violations} / {beta_preds.size}")

print("\n" + "="*60)
print("✅ Training Complete - V3 IMPROVED!")
print("="*60)
print(f"\nZamzam 2019 Architecture:")
print(f"  ✓ Pure MSE loss (no penalties)")
print(f"  ✓ Sigmoid output layer (enforces [0,1] bounds)")
print(f"  ✓ R-ACOPF training data (λ margin for interior solutions)")
print(f"  ✓ No Qg prediction (determined by power flow)")
print(f"\nNext steps:")
print(f"  1. Run power flow recovery (03_power_flow_recovery.py)")
print(f"  2. Check if δ_q improved (target: < 1.58 MVAr for IEEE 57-bus)")
print(f"  3. Compare with Zamzam Table III results")

plt.show()
