"""
One-click script to run the entire pipeline.

Usage:
    python run_all.py           # Run full pipeline
    python run_all.py --quick   # Skip data generation (use existing)
"""

import subprocess
import sys
import os
import argparse

def run_script(script_name, description):
    """Run a Python script and handle errors."""
    print("\n" + "="*70)
    print(f"RUNNING: {description}")
    print(f"Script: {script_name}")
    print("="*70)
    
    result = subprocess.run([sys.executable, script_name], 
                          capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n❌ ERROR: {script_name} failed with exit code {result.returncode}")
        sys.exit(1)
    
    print(f"\n✓ {description} completed successfully")

def main():
    parser = argparse.ArgumentParser(description="Run OPF Neural Network Pipeline")
    parser.add_argument('--quick', action='store_true', 
                       help='Skip data generation (must have existing data)')
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("="*70)
    print("OPF NEURAL NETWORK - COMPLETE PIPELINE")
    print("Based on Zamzam et al. 2019")
    print("="*70)
    
    scripts = []
    
    if not args.quick:
        scripts.append(("00_generate_opf_data_pypower.py", 
                       "Data Generation (100k samples, ~45 min)"))
    
    scripts.extend([
        ("02_train_opf_network_v3_improved.py", 
         "Neural Network Training (~20 min)"),
        ("03_power_flow_recovery.py", 
         "Power Flow Recovery - Algorithm 1"),
        ("04_evaluate_with_metrics.py", 
         "Comprehensive Evaluation & Metrics"),
        ("05_analyze_qg_correction.py", 
         "Reactive Power Correction Analysis"),
        ("06_explain_correction_process.py", 
         "Correction Process Visualization"),
        ("07_sensitivity_analysis_qg_filtering.py", 
         "Sensitivity Analysis (~2 hours)")
    ])
    
    total = len(scripts)
    for i, (script, desc) in enumerate(scripts, 1):
        print(f"\n[Step {i}/{total}]")
        run_script(script, desc)
    
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETE!")
    print("="*70)
    print("\nResults saved to: Outputs_V3/")
    print("\nKey outputs:")
    print("  - opf_case57_train_v2.csv, test_v2.csv, val_v2.csv")
    print("  - best_opf_model.pth")
    print("  - recovery_results_v3.csv")
    print("  - evaluation_report.txt")
    print("  - Multiple visualization PNG files")
    print("\nFor quick re-runs: python run_all.py --quick")
    print("="*70)

if __name__ == "__main__":
    main()
