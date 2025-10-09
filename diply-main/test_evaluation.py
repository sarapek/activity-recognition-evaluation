"""
Test script for evaluating all sanity check variants against the original file
"""

from segment_evaluator import SegmentEvaluator
from visualize_segments_timeline import visualize_segments_timeline
from modify_file import SanityCheckModifier
import os
import json
from datetime import datetime
import numpy as np

def convert_to_json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    else:
        return str(obj)

def run_single_test(evaluator, predicted_file, ground_truth_file, variant_name, 
                    output_file=None, per_segment=False):
    """Run evaluation on a single variant"""
    header = f"\n{'='*70}\nTesting: {variant_name.upper()}\n{'='*70}\n"
    
    # Print to terminal
    print(header)
    
    # Write to file if specified
    if output_file:
        output_file.write(header)
    
    try:
        results = evaluator.evaluate(predicted_file, ground_truth_file, per_segment=per_segment)
        
        # Print to terminal
        evaluator.print_report(results, show_per_segment=per_segment)
        
        # Write to file
        if output_file:
            write_report_to_file(results, output_file, show_per_segment=per_segment)

        # --- Visualize segments ---
        # Extract segments for visualization
        pred_segments = evaluator.extract_segments(evaluator.parse_log_file(predicted_file))
        gt_segments = evaluator.extract_segments(evaluator.parse_log_file(ground_truth_file))
        try:
            fig, axes = visualize_segments_timeline(
                predicted_segments=pred_segments,
                ground_truth_segments=gt_segments,
                title=f"Timeline: {variant_name}"
            )
            import matplotlib.pyplot as plt
            plt.show()
        except Exception as vis_err:
            print(f"[Visualization error] {vis_err}")

        return results
    except Exception as e:
        error_msg = f"ERROR: {str(e)}\n"
        print(error_msg)
        if output_file:
            output_file.write(error_msg)
        return None

def write_report_to_file(results, file_handle, show_per_segment=False):
    """Write formatted evaluation report to file"""
    file_handle.write("=" * 60 + "\n")
    file_handle.write("SEGMENT EVALUATION REPORT\n")
    file_handle.write("=" * 60 + "\n")
    
    file_handle.write("\n--- Detection Metrics ---\n")
    file_handle.write(f"F1 Score:         {results['f1_score']:.4f}\n")
    file_handle.write(f"Precision:        {results['precision']:.4f}\n")
    file_handle.write(f"Recall:           {results['recall']:.4f}\n")
    auc_val = results['auc']
    if auc_val is not None:
        file_handle.write(f"AUC:              {auc_val:.4f}\n")
    else:
        file_handle.write(f"AUC:              N/A (no confidence scores)\n")
    
    file_handle.write("\n--- Confusion Matrix ---\n")
    file_handle.write(f"True Positives:   {results['true_positives']}\n")
    file_handle.write(f"False Positives:  {results['false_positives']}\n")
    file_handle.write(f"False Negatives:  {results['false_negatives']}\n")
    file_handle.write(f"True Negatives:   {results['true_negatives']}\n")
    
    file_handle.write("\n--- Temporal Accuracy ---\n")
    if results['start_errors_mean'] is not None:
        file_handle.write(f"Start Time Error (mean): {results['start_errors_mean']:.4f}s\n")
        file_handle.write(f"Start Time Error (std):  {results['start_errors_std']:.4f}s\n")
        file_handle.write(f"End Time Error (mean):   {results['end_errors_mean']:.4f}s\n")
        file_handle.write(f"End Time Error (std):    {results['end_errors_std']:.4f}s\n")
        file_handle.write(f"Duration Error (mean):   {results['duration_errors_mean']:.4f}s\n")
    else:
        file_handle.write("No matched segments for temporal analysis\n")
    
    file_handle.write("\n--- Segment Counts ---\n")
    file_handle.write(f"Predicted Segments:     {results['num_predicted']}\n")
    file_handle.write(f"Ground Truth Segments:  {results['num_ground_truth']}\n")
    file_handle.write(f"Matched Segments:       {results['num_matched']}\n")
    
    # ADD THIS SECTION FOR PER-SEGMENT BREAKDOWN
    if 'per_segment_metrics' in results and show_per_segment:
        file_handle.write("\n" + "=" * 60 + "\n")
        file_handle.write("PER-SEGMENT BREAKDOWN\n")
        file_handle.write("=" * 60 + "\n")
        
        per_seg = results['per_segment_metrics']
        for seg_id, metrics in sorted(per_seg.items()):
            file_handle.write(f"\n--- Segment ID: {seg_id} ---\n")
            file_handle.write(f"  Predicted:     {metrics['num_predicted']}\n")
            file_handle.write(f"  Ground Truth:  {metrics['num_ground_truth']}\n")
            file_handle.write(f"  Status:        ")
            
            if metrics['detected']:
                file_handle.write("DETECTED\n")
            elif metrics['false_positive']:
                file_handle.write("FALSE POSITIVE\n")
            elif metrics['false_negative']:
                file_handle.write("MISSED\n")
            else:
                file_handle.write("TRUE NEGATIVE\n")
            
            if metrics['start_error'] is not None:
                file_handle.write(f"  Start Error:   {metrics['start_error']:.4f}s\n")
                file_handle.write(f"  End Error:     {metrics['end_error']:.4f}s\n")
                file_handle.write(f"  Duration Err:  {metrics['duration_error']:.4f}s\n")
                tolerance_str = "YES" if metrics['within_tolerance'] else "NO"
                file_handle.write(f"  Within Tol:    {tolerance_str}\n")
    
    file_handle.write("=" * 60 + "\n\n")

def run_all_sanity_tests(ground_truth_file, sanity_check_dir="./SanityCheck", per_segment=False):
    """
    Run evaluation on all sanity check variants and generate summary report
    
    Args:
        ground_truth_file: Path to original/ground truth file
        sanity_check_dir: Directory containing sanity check variants
        per_segment: If True, include per-segment breakdown for each test
    """
    
    # Initialize evaluator
    evaluator = SegmentEvaluator(time_tolerance_seconds=2.0)
    
    # Expected variants
    variants = [
        'perfect',
        'small_time_shift',
        'large_time_shift', 
        'boundary_errors',
        'missing_segments',
        'false_positives',
        'label_confusion',
        'combined_errors'
    ]
    
    # Get base filename
    base_name = os.path.splitext(os.path.basename(ground_truth_file))[0]
    
    # Store all results
    all_results = {}
    
    print("\n" + "="*70)
    print("SANITY CHECK EVALUATION TEST SUITE")
    print("="*70)
    print(f"Ground Truth: {ground_truth_file}")
    print(f"Sanity Check Directory: {sanity_check_dir}")
    print(f"Testing {len(variants)} variants...")
    print(f"Per-segment breakdown: {'Enabled' if per_segment else 'Disabled'}")
    
    # Run tests for each variant
    for variant in variants:
        predicted_file = os.path.join(sanity_check_dir, f"{base_name}_{variant}.txt")
        
        if not os.path.exists(predicted_file):
            print(f"\nWARNING: File not found: {predicted_file}")
            continue
        
        results = run_single_test(evaluator, predicted_file, ground_truth_file, 
                                 variant, per_segment=per_segment)
        
        if results:
            all_results[variant] = results
    
    # Generate summary report
    print("\n\n" + "="*70)
    print("SUMMARY REPORT - ALL VARIANTS")
    print("="*70)
    
    print(f"\n{'Variant':<20} {'F1':>8} {'AUC':>8} {'Precision':>10} {'Recall':>8} {'Time Err':>10}")
    print("-"*70)
    
    for variant, results in all_results.items():
        time_err = results.get('start_errors_mean', 0)
        time_err_str = f"{time_err:.3f}s" if time_err is not None else "N/A"
        auc_val = results['auc']
        auc_str = f"{auc_val:.4f}" if auc_val is not None else "N/A"
        
        print(f"{variant:<20} {results['f1_score']:>8.4f} {auc_str:>8} "
              f"{results['precision']:>10.4f} {results['recall']:>8.4f} {time_err_str:>10}")
    
    """ # Validation checks
    print("\n" + "="*70)
    print("VALIDATION CHECKS")
    print("="*70)
    
    checks_passed = 0
    total_checks = 0
    
    # Check 1: Perfect copy should have F1 = 1.0
    if 'perfect' in all_results:
        total_checks += 1
        perfect_f1 = all_results['perfect']['f1_score']
        if perfect_f1 >= 0.99:
            print(f"✓ Perfect copy F1 score: {perfect_f1:.4f} (PASS)")
            checks_passed += 1
        else:
            print(f"✗ Perfect copy F1 score: {perfect_f1:.4f} (FAIL - expected ~1.0)")
    
    # Check 2: Missing segments should have lower recall
    if 'perfect' in all_results and 'missing_segments' in all_results:
        total_checks += 1
        perfect_recall = all_results['perfect']['recall']
        missing_recall = all_results['missing_segments']['recall']
        if missing_recall < perfect_recall:
            print(f"✓ Missing segments recall: {missing_recall:.4f} < {perfect_recall:.4f} (PASS)")
            checks_passed += 1
        else:
            print(f"✗ Missing segments should have lower recall (FAIL)")
    
    # Check 3: False positives should have lower precision
    if 'perfect' in all_results and 'false_positives' in all_results:
        total_checks += 1
        perfect_precision = all_results['perfect']['precision']
        fp_precision = all_results['false_positives']['precision']
        if fp_precision < perfect_precision:
            print(f"✓ False positives precision: {fp_precision:.4f} < {perfect_precision:.4f} (PASS)")
            checks_passed += 1
        else:
            print(f"✗ False positives should have lower precision (FAIL)")
    
    # Check 4: Large time shifts should have larger time errors
    if 'small_time_shift' in all_results and 'large_time_shift' in all_results:
        total_checks += 1
        small_err = all_results['small_time_shift'].get('start_errors_mean')
        large_err = all_results['large_time_shift'].get('start_errors_mean')
        if small_err is not None and large_err is not None and large_err > small_err:
            print(f"✓ Large shifts have bigger errors: {large_err:.3f}s > {small_err:.3f}s (PASS)")
            checks_passed += 1
        else:
            print(f"✗ Large time shifts should have larger errors (FAIL)")
    
    print(f"\nValidation Score: {checks_passed}/{total_checks} checks passed") """
    
    # Save results to JSON
    # Save results to JSON
    output_file = os.path.join(sanity_check_dir, "evaluation_results.json")
    with open(output_file, 'w') as f:
        # Convert results to JSON-serializable format
        
        json.dump({
            'test_date': datetime.now().isoformat(),
            'ground_truth_file': ground_truth_file,
            'results': convert_to_json_serializable(all_results)
            #'validation_score': f"{checks_passed}/{total_checks}"
        }, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")
        
    return all_results

def test_with_confidence_scores(ground_truth_file, predicted_file, confidence_scores, per_segment=False):
    """
    Test evaluation with confidence scores (for SVM predictions)
    
    Args:
        ground_truth_file: Path to ground truth
        predicted_file: Path to predictions
        confidence_scores: List of confidence values for each predicted segment
        per_segment: If True, include per-segment breakdown
    """
    print("\n" + "="*70)
    print("TESTING WITH CONFIDENCE SCORES (SVM Mode)")
    print("="*70)
    
    evaluator = SegmentEvaluator(time_tolerance_seconds=2.0)
    
    results = evaluator.evaluate(
        predicted_file,
        ground_truth_file,
        confidence_scores=confidence_scores,
        per_segment=per_segment
    )
    
    evaluator.print_report(results, show_per_segment=per_segment)
    
    print(f"\nConfidence scores used: {confidence_scores}")
    auc_val = results['auc']
    if auc_val is not None:
        print(f"AUC with confidence: {auc_val:.4f}")
    else:
        print(f"AUC: N/A (not enough varied scores)")
    
    return results

def quick_test(ground_truth_file, predicted_file, per_segment=False):
    """Quick single-file test"""
    evaluator = SegmentEvaluator(time_tolerance_seconds=2.0)
    results = evaluator.evaluate(predicted_file, ground_truth_file, per_segment=per_segment)
    evaluator.print_report(results, show_per_segment=per_segment)
    return results


if __name__ == "__main__":
    
    # Configuration
    GROUND_TRUTH_FILE = "./RawData/P001.txt"
    SANITY_CHECK_DIR = "./SanityCheck"
    
    print("\n" + "="*70)
    print("SEGMENT EVALUATION TESTING SYSTEM")
    print("="*70)
    
    # Option 1: Create sanity check files first (if not already created)
    create_files = input("\nCreate sanity check files? (y/n): ").lower().strip()
    if create_files == 'y':
        print("\nCreating sanity check files...")
        modifier = SanityCheckModifier(seed=42)
        modifier.create_sanity_check_variants(GROUND_TRUTH_FILE, SANITY_CHECK_DIR)
    
    # Option 2: Ask about per-segment breakdown
    per_segment_enabled = input("\nEnable per-segment breakdown? (y/n): ").lower().strip() == 'y'
    
    # Option 3: Run full test suite
    print("\n" + "="*70)
    print("Choose test mode:")
    print("1. Run full sanity check test suite")
    print("2. Quick test single file")
    print("3. Test with confidence scores (SVM mode)")
    print("="*70)
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == '1':
        # Full test suite
        all_results = run_all_sanity_tests(GROUND_TRUTH_FILE, SANITY_CHECK_DIR, 
                                          per_segment=per_segment_enabled)
        
    elif choice == '2':
        # Quick single file test
        predicted = input("Enter predicted file path: ").strip()
        quick_test(GROUND_TRUTH_FILE, predicted, per_segment=per_segment_enabled)
        
    elif choice == '3':
        # Test with confidence scores
        predicted = input("Enter predicted file path: ").strip()
        scores_input = input("Enter confidence scores (comma-separated, e.g., 0.95,0.87,0.92): ").strip()
        confidence_scores = [float(x.strip()) for x in scores_input.split(',')]
        test_with_confidence_scores(GROUND_TRUTH_FILE, predicted, confidence_scores, 
                                   per_segment=per_segment_enabled)
    
    else:
        print("Invalid choice. Exiting.")
    
    print("\n" + "="*70)
    print("Testing complete!")
    print("="*70)