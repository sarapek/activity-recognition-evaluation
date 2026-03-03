"""
Sensitivity Analysis for Evaluation Parameters
===============================================

Runs experiments varying one parameter at a time to determine which settings
have the most impact on segment-level performance.

Baseline settings:
- overlap_threshold = 0.5
- start_time_threshold = 30.0s
- max_gap_seconds = 60.0s
- confidence_threshold = 0.5

Usage:
    python scripts/run_sensitivity_analysis.py

Results will be saved to results/sensitivity/
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime

# Add evaluation folder to path
sys.path.append('../evaluation')

# ============================================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================================

BASELINE = {
    'name': 'baseline',
    'overlap_threshold': 0.5,
    'start_time_threshold': 30.0,
    'max_gap_seconds': 60.0,
    'confidence_threshold': 0.5,
    'description': 'Baseline configuration (already completed)'
}

EXPERIMENTS = [
    # Overlap threshold variations
    {
        'name': 'overlap_0.3',
        'overlap_threshold': 0.3,
        'start_time_threshold': 30.0,
        'max_gap_seconds': 60.0,
        'confidence_threshold': 0.5,
        'description': 'More lenient segment matching (overlap=0.3)'
    },
    {
        'name': 'overlap_0.7',
        'overlap_threshold': 0.7,
        'start_time_threshold': 30.0,
        'max_gap_seconds': 60.0,
        'confidence_threshold': 0.5,
        'description': 'Stricter segment matching (overlap=0.7)'
    },

    # Start time threshold variations
    {
        'name': 'start_time_0',
        'overlap_threshold': 0.5,
        'start_time_threshold': 0.0,
        'max_gap_seconds': 60.0,
        'confidence_threshold': 0.5,
        'description': 'No start time tolerance (exact timing required)'
    },
    {
        'name': 'start_time_15',
        'overlap_threshold': 0.5,
        'start_time_threshold': 15.0,
        'max_gap_seconds': 60.0,
        'confidence_threshold': 0.5,
        'description': 'Moderate start time tolerance (15s)'
    },
    {
        'name': 'start_time_60',
        'overlap_threshold': 0.5,
        'start_time_threshold': 60.0,
        'max_gap_seconds': 60.0,
        'confidence_threshold': 0.5,
        'description': 'Generous start time tolerance (60s)'
    },

    # Gap threshold variations
    {
        'name': 'gap_30',
        'overlap_threshold': 0.5,
        'start_time_threshold': 30.0,
        'max_gap_seconds': 30.0,
        'confidence_threshold': 0.5,
        'description': 'More fragmentation (gap=30s)'
    },
    {
        'name': 'gap_120',
        'overlap_threshold': 0.5,
        'start_time_threshold': 30.0,
        'max_gap_seconds': 120.0,
        'confidence_threshold': 0.5,
        'description': 'Less fragmentation (gap=120s)'
    },
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def run_experiment(exp_config, output_base_dir):
    """Run a single experiment with given configuration"""

    exp_name = exp_config['name']
    print(f"\n{'='*70}")
    print(f"RUNNING EXPERIMENT: {exp_name}")
    print(f"{'='*70}")
    print(f"Description: {exp_config['description']}")
    print(f"Settings:")
    print(f"  overlap_threshold:      {exp_config['overlap_threshold']}")
    print(f"  start_time_threshold:   {exp_config['start_time_threshold']}s")
    print(f"  max_gap_seconds:        {exp_config['max_gap_seconds']}s")
    print(f"  confidence_threshold:   {exp_config['confidence_threshold']}")
    print(f"{'='*70}\n")

    # Build command
    cmd = [
        'python', 'train_and_evaluate.py',
        '--overlap-threshold', str(exp_config['overlap_threshold']),
        '--start-time-threshold', str(exp_config['start_time_threshold']),
        '--max-gap-seconds', str(exp_config['max_gap_seconds']),
        '--confidence-threshold', str(exp_config['confidence_threshold'])
    ]

    # Create log file
    log_file = os.path.join(output_base_dir, f'{exp_name}.log')

    # Run experiment
    start_time = time.time()
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        for line in process.stdout:
            f.write(line)
            f.flush()
            # Print important lines to console
            if any(keyword in line for keyword in ['FOLD', 'AUC', 'Precision', 'Recall', 'F1']):
                print(line.rstrip())

        process.wait()

    elapsed = time.time() - start_time

    if process.returncode == 0:
        print(f"\n[OK] Experiment '{exp_name}' completed in {elapsed/60:.1f} minutes")
        return True
    else:
        print(f"\n[FAILED] Experiment '{exp_name}' FAILED")
        return False


def extract_results(output_dir):
    """Extract key metrics from cross_validation_summary.json"""

    # Find the most recent run directory
    runs = [d for d in os.listdir(output_dir) if d.startswith('run_')]
    if not runs:
        return None

    latest_run = sorted(runs)[-1]
    summary_file = os.path.join(output_dir, latest_run, 'cross_validation_summary.json')

    if not os.path.exists(summary_file):
        return None

    with open(summary_file, 'r') as f:
        data = json.load(f)

    # Extract macro and weighted averages
    macro = data['aggregated']['aggregate']['macro_avg']
    weighted = data['aggregated']['aggregate']['weighted_avg']

    return {
        'run_dir': latest_run,
        'macro_seg_precision': macro['segment_precision']['mean'],
        'macro_seg_recall': macro['segment_recall']['mean'],
        'macro_seg_f1': macro['segment_f1']['mean'],
        'macro_seg_auc': macro['segment_auc']['mean'],
        'weighted_seg_precision': weighted['segment_precision']['mean'],
        'weighted_seg_recall': weighted['segment_recall']['mean'],
        'weighted_seg_f1': weighted['segment_f1']['mean'],
        'weighted_seg_auc': weighted['segment_auc']['mean'],
    }


def create_comparison_table(all_results, output_file):
    """Create a comparison table of all experiments"""

    with open(output_file, 'w') as f:
        f.write("SENSITIVITY ANALYSIS RESULTS\n")
        f.write("="*100 + "\n\n")

        # Macro-average results
        f.write("MACRO-AVERAGE SEGMENT METRICS\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Experiment':<20} {'Precision':>12} {'Recall':>12} {'F1':>12} {'AUC':>12}\n")
        f.write("-"*100 + "\n")

        for exp_name, results in all_results.items():
            if results:
                f.write(f"{exp_name:<20} {results['macro_seg_precision']:>12.4f} "
                       f"{results['macro_seg_recall']:>12.4f} {results['macro_seg_f1']:>12.4f} "
                       f"{results['macro_seg_auc']:>12.4f}\n")

        f.write("\n\n")

        # Weighted-average results
        f.write("WEIGHTED-AVERAGE SEGMENT METRICS\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Experiment':<20} {'Precision':>12} {'Recall':>12} {'F1':>12} {'AUC':>12}\n")
        f.write("-"*100 + "\n")

        for exp_name, results in all_results.items():
            if results:
                f.write(f"{exp_name:<20} {results['weighted_seg_precision']:>12.4f} "
                       f"{results['weighted_seg_recall']:>12.4f} {results['weighted_seg_f1']:>12.4f} "
                       f"{results['weighted_seg_auc']:>12.4f}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all sensitivity analysis experiments"""

    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS - EVALUATION PARAMETER SETTINGS")
    print("="*70)
    print(f"\nTotal experiments: {len(EXPERIMENTS)}")
    print(f"Baseline already completed: True")
    print(f"Estimated total time: ~{len(EXPERIMENTS) * 45} minutes ({len(EXPERIMENTS) * 45 / 60:.1f} hours)")
    print("\nPress Ctrl+C to cancel...\n")

    time.sleep(3)

    # Create output directory
    output_base = '../results/sensitivity'
    os.makedirs(output_base, exist_ok=True)

    # Save experiment configurations
    config_file = os.path.join(output_base, 'experiment_configurations.json')
    with open(config_file, 'w') as f:
        json.dump({
            'baseline': BASELINE,
            'experiments': EXPERIMENTS,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"[OK] Saved experiment configurations to: {config_file}\n")

    # Run all experiments
    all_results = {}
    successful = 0
    failed = 0

    for i, exp_config in enumerate(EXPERIMENTS, 1):
        print(f"\n{'#'*70}")
        print(f"# EXPERIMENT {i}/{len(EXPERIMENTS)}")
        print(f"{'#'*70}")

        success = run_experiment(exp_config, output_base)

        if success:
            # Extract results from the most recent run
            results = extract_results('../results/training_runs')
            if results:
                all_results[exp_config['name']] = results
                successful += 1
            else:
                print(f"Warning: Could not extract results for {exp_config['name']}")
                failed += 1
        else:
            failed += 1

        # Save intermediate results
        if all_results:
            intermediate_file = os.path.join(output_base, 'intermediate_results.json')
            with open(intermediate_file, 'w') as f:
                json.dump(all_results, f, indent=2)

    # Create final comparison table
    print("\n" + "="*70)
    print("CREATING COMPARISON TABLE")
    print("="*70 + "\n")

    comparison_file = os.path.join(output_base, 'sensitivity_analysis_results.txt')
    create_comparison_table(all_results, comparison_file)

    # Save final JSON results
    final_json = os.path.join(output_base, 'final_results.json')
    with open(final_json, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nSuccessful experiments: {successful}/{len(EXPERIMENTS)}")
    print(f"Failed experiments: {failed}/{len(EXPERIMENTS)}")
    print(f"\nResults saved to:")
    print(f"  - {comparison_file}")
    print(f"  - {final_json}")
    print(f"  - Individual logs in: {output_base}/")
    print("\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExperiments cancelled by user.")
        sys.exit(1)
