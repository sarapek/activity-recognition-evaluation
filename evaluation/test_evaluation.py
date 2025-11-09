"""
Test script for evaluating all sanity check variants against the original file
"""

from statistics import variance
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
                    output_file=None, timeline_visualisation=False, show_roc=False,
                    activity_filter=None):
    """Run evaluation on a single variant"""
    header = f"\n{'='*70}\nTesting: {variant_name.upper()}\n{'='*70}\n"
    
    print(header)
    if output_file:
        output_file.write(header)
    
    try:
        # Use evaluate_with_dual_roc to get ROC curves
        if show_roc:
            results = evaluator.evaluate_with_dual_roc(predicted_file, ground_truth_file, aggregation='average')
            
            # Plot ROC curves
            try:
                evaluator.plot_dual_roc_curves(
                    results, 
                    save_path=f"roc_{variant_name}.png"
                )
            except Exception as plot_err:
                print(f"Could not plot ROC curves: {plot_err}")
        else:
            # Use regular evaluate (no ROC)
            results = evaluator.evaluate(predicted_file, ground_truth_file)
        
        # Print to terminal with activity filter
        evaluator.print_report(results, activity_filter=activity_filter)
        
        # Write to file
        if output_file:
            write_report_to_file(results, output_file)

    except Exception as e:
        error_msg = f"ERROR: {str(e)}\n"
        print(error_msg)
        if output_file:
            output_file.write(error_msg)
        return None
    
    # Visualization
    if timeline_visualisation:
        try:
            pred_segments = evaluator.extract_segments(evaluator.parse_log_file(predicted_file))
            gt_segments = evaluator.extract_segments(evaluator.parse_log_file(ground_truth_file))
            
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

def write_report_to_file(results, file_handle):
    """Write formatted evaluation report to file"""
    file_handle.write("=" * 60 + "\n")
    file_handle.write("SEGMENT EVALUATION REPORT\n")
    file_handle.write("=" * 60 + "\n")
    
    # Get metrics from micro_avg
    micro = results.get('micro_avg', {})
    macro = results.get('macro_avg', {})
    weighted = results.get('weighted_avg', {})
    
    file_handle.write("\n--- Aggregate Metrics ---\n")
    file_handle.write(f"Macro F1:    {macro.get('f1_score', 0):.4f}\n")
    file_handle.write(f"Micro F1:    {micro.get('f1_score', 0):.4f}\n")
    file_handle.write(f"Weighted F1: {weighted.get('f1_score', 0):.4f}\n")
    file_handle.write(f"Precision:   {micro.get('precision', 0):.4f}\n")
    file_handle.write(f"Recall:      {micro.get('recall', 0):.4f}\n")
    
    # Per-class metrics
    if 'per_class' in results:
        file_handle.write("\n" + "=" * 60 + "\n")
        file_handle.write("PER-CLASS METRICS\n")
        file_handle.write("=" * 60 + "\n")
        
        per_class = results['per_class']
        for activity_id, metrics in sorted(per_class.items()):
            file_handle.write(f"\n--- Activity {activity_id} ---\n")
            file_handle.write(f"  GT Duration:   {metrics['gt_total_duration']:.1f}s\n")
            file_handle.write(f"  Pred Duration: {metrics['pred_total_duration']:.1f}s\n")
            file_handle.write(f"  Precision:     {metrics['precision']:.4f}\n")
            file_handle.write(f"  Recall:        {metrics['recall']:.4f}\n")
            file_handle.write(f"  F1 Score:      {metrics['f1_score']:.4f}\n")
            file_handle.write(f"  IoU:           {metrics['iou']:.4f}\n")
            
            # Add AUC scores if available
            if 'frame_level' in results and activity_id in results['frame_level']:
                auc_val = results['frame_level'][activity_id].get('auc')
                if auc_val is not None:
                    file_handle.write(f"  Frame AUC:     {auc_val:.4f}\n")
            
            if 'segment_level' in results and activity_id in results['segment_level']:
                auc_val = results['segment_level'][activity_id].get('auc')
                if auc_val is not None:
                    file_handle.write(f"  Segment AUC:   {auc_val:.4f}\n")
    
    file_handle.write("=" * 60 + "\n\n")
    """Write formatted evaluation report to file"""
    file_handle.write("=" * 60 + "\n")
    file_handle.write("SEGMENT EVALUATION REPORT\n")
    file_handle.write("=" * 60 + "\n")
    
    # Get metrics from micro_avg
    micro = results.get('micro_avg', {})
    
    file_handle.write("\n--- Detection Metrics (Micro-Average) ---\n")
    file_handle.write(f"F1 Score:         {micro.get('f1_score', 0):.4f}\n")
    file_handle.write(f"Precision:        {micro.get('precision', 0):.4f}\n")
    file_handle.write(f"Recall (TPR):     {micro.get('recall', 0):.4f}\n")
    file_handle.write(f"FPR:              {results['fpr']:.4f}\n")
    auc_val = results.get('auc')
    if auc_val is not None:
        file_handle.write(f"AUC:              {auc_val:.4f}\n")
    else:
        file_handle.write(f"AUC:              N/A (no varied confidence scores)\n")
    
    file_handle.write("\n--- Confusion Matrix (Segments) ---\n")
    file_handle.write(f"True Positives:   {results['true_positives']} segments\n")
    file_handle.write(f"False Positives:  {results['false_positives']} segments\n")
    file_handle.write(f"False Negatives:  {results['false_negatives']} segments\n")
    file_handle.write(f"True Negatives:   {results['true_negatives']} segment IDs\n")
    
    file_handle.write("\n--- Temporal Accuracy ---\n")
    if results.get('start_errors_mean') is not None:
        file_handle.write(f"Start Time Error (mean): {results['start_errors_mean']:.4f}s\n")
        file_handle.write(f"Start Time Error (std):  {results['start_errors_std']:.4f}s\n")
        file_handle.write(f"Start Time Error (max):  {results['start_errors_max']:.4f}s\n")
        file_handle.write(f"End Time Error (mean):   {results['end_errors_mean']:.4f}s\n")
        file_handle.write(f"End Time Error (std):    {results['end_errors_std']:.4f}s\n")
        file_handle.write(f"Duration Error (mean):   {results['duration_errors_mean']:.4f}s\n")
    else:
        file_handle.write("No matched segments for temporal analysis\n")
    
    file_handle.write("\n--- Segment Counts ---\n")
    # Calculate these from the segment-based metrics
    num_predicted = results['true_positives'] + results['false_positives']
    num_ground_truth = results['true_positives'] + results['false_negatives']
    num_matched = results['true_positives']
    
    file_handle.write(f"Predicted Segments:     {num_predicted}\n")
    file_handle.write(f"Ground Truth Segments:  {num_ground_truth}\n")
    file_handle.write(f"Matched Segments:       {num_matched}\n")
    file_handle.write(f"Missed Segments:        {results['false_negatives']}\n")
    file_handle.write(f"False Alarm Segments:   {results['false_positives']}\n")
    
    # Per-segment breakdown
    if 'per_segment_metrics' in results:
        filtered_metrics = {k: v for k, v in results['per_segment_metrics'].items() if k <= 8}
        
        if filtered_metrics:
            print(f"\n--- {variance.upper()} ---")
            for seg_id in sorted(filtered_metrics.keys()):
                metrics = filtered_metrics[seg_id]
                status = "DETECTED" if metrics['detected'] else \
                        "FALSE_POS" if metrics['false_positive'] else \
                        "MISSED" if metrics['false_negative'] else "TRUE_NEG"
                print(f"  Activity {seg_id}: {status}")
                
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
                if metrics.get('too_early'):
                    file_handle.write("DETECTED (TOO EARLY)\n")
                else:
                    file_handle.write("DETECTED\n")
            elif metrics['false_positive']:
                file_handle.write("FALSE POSITIVE\n")
            elif metrics['false_negative']:
                if metrics.get('too_late'):
                    file_handle.write("MISSED (TOO LATE)\n")
                else:
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

def run_all_sanity_tests(ground_truth_file, sanity_check_dir="./SanityCheck", 
                        timeline_visualisation=False, start_time_threshold_seconds=2.0,
                        show_roc=False, activity_filter=None):
    """Run evaluation on all sanity check variants and generate summary report"""
    
    evaluator = SegmentEvaluator(timeline_resolution_seconds=5,start_time_threshold_seconds=start_time_threshold_seconds)
    
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
    
    base_name = os.path.splitext(os.path.basename(ground_truth_file))[0]
    all_results = {}
    
    print("\n" + "="*70)
    print("SANITY CHECK EVALUATION TEST SUITE")
    print("="*70)
    
    # Run tests for each variant
    for variant in variants:
        predicted_file = os.path.join(sanity_check_dir, f"{base_name}_{variant}.txt")
        
        if not os.path.exists(predicted_file):
            print(f"\nWARNING: File not found: {predicted_file}")
            continue
        
        results = run_single_test(
            evaluator, 
            predicted_file, 
            ground_truth_file, 
            variant, 
            timeline_visualisation=timeline_visualisation,
            show_roc=show_roc,
            activity_filter=activity_filter
        )
        
        if results:
            all_results[variant] = results
    
    # Generate summary report
    print("\n\n" + "="*70)
    print("SUMMARY REPORT - ALL VARIANTS")
    print("="*70)
    
    # Filter activities in summary too
    if activity_filter:
        print(f"(Filtered to activities: {', '.join(activity_filter)})")
    
    print(f"\n{'Variant':<20} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Frame AUC':>10} {'Seg AUC':>10}")
    print("-"*80)
    
    for variant_name in variants:
        pred_file = os.path.join(sanity_check_dir, f'P001_{variant_name}.txt')
        
        print(f"\nEvaluating {variant_name}...")
        results = evaluator.evaluate_with_dual_roc(pred_file, ground_truth_file, aggregation='average')
        
        # ADD DEBUG HERE FOR PERFECT VARIANT
        if variant_name == 'perfect':
            print("\n=== Perfect Variant Debug ===")
            
            pred_df = evaluator.parse_log_file(pred_file)
            gt_df = evaluator.parse_log_file(ground_truth_file)
                            
        # if variant_name in ['label_confusion', 'false_positives', 'combined_errors']:
        #    print(f"\n=== {variant_name} Confidence Check ===")
        #    print(f"  Mean confidence: {pred_df['confidence'].mean():.3f}")
        #   print(f"  Expected for {variant_name}: 0.15-0.50")
        
        # Continue with normal processing
        micro = results.get('micro_avg', {})
        # Get metrics from micro_avg
        micro = results.get('micro_avg', {})
        f1 = micro.get('f1_score', 0)
        precision = micro.get('precision', 0)
        recall = micro.get('recall', 0)
        
        # Get average AUC from frame_level if available (filtered)
        frame_auc_str = "N/A"
        if 'frame_level' in results:
            if activity_filter:
                aucs = [v.get('auc') for k, v in results['frame_level'].items() 
                       if k in activity_filter and v.get('auc') is not None]
            else:
                aucs = [v.get('auc') for v in results['frame_level'].values() 
                       if v.get('auc') is not None]
            if aucs:
                frame_auc_str = f"{np.mean(aucs):.3f}"
        
        # Get average AUC from segment_level if available (filtered)
        seg_auc_str = "N/A"
        if 'segment_level' in results:
            if activity_filter:
                aucs = [v.get('auc') for k, v in results['segment_level'].items() 
                       if k in activity_filter and v.get('auc') is not None]
            else:
                aucs = [v.get('auc') for v in results['segment_level'].values() 
                       if v.get('auc') is not None]
            if aucs:
                seg_auc_str = f"{np.mean(aucs):.3f}"
        
        print(f"{variant:<20} {f1:>8.4f} {precision:>10.4f} {recall:>8.4f} {frame_auc_str:>10} {seg_auc_str:>10}")
    
    # Save results to JSON
    output_file = os.path.join(sanity_check_dir, "evaluation_results.json")    
    with open(output_file, 'w') as f:
        json.dump({
            'test_date': datetime.now().isoformat(),
            'ground_truth_file': ground_truth_file,
            'activity_filter': activity_filter,
            'results': convert_to_json_serializable(all_results)
        }, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")
    
    return all_results

def test_with_confidence_scores(ground_truth_file, predicted_file, confidence_scores, start_time_threshold_seconds=2.0):
    """
    Test evaluation with confidence scores (for SVM predictions)
    
    Args:
        ground_truth_file: Path to ground truth
        predicted_file: Path to predictions
        confidence_scores: List of confidence values for each predicted segment
    """
    print("\n" + "="*70)
    print("TESTING WITH CONFIDENCE SCORES (SVM Mode)")
    print("="*70)
    
    evaluator = SegmentEvaluator(timeline_resolution_seconds=30.0,start_time_threshold_seconds=start_time_threshold_seconds)
    
    results = evaluator.evaluate(
        predicted_file,
        ground_truth_file,
        confidence_scores=confidence_scores
    )
    
    evaluator.print_report(results)
    
    print(f"\nConfidence scores used: {confidence_scores}")
    auc_val = results['auc']
    if auc_val is not None:
        print(f"AUC with confidence: {auc_val:.4f}")
    else:
        print(f"AUC: N/A (not enough varied scores)")
    
    return results

def quick_test(ground_truth_file, predicted_file, start_time_threshold_seconds=2.0, 
               show_roc=False, activity_filter=None):
    """Quick single-file test"""
    evaluator = SegmentEvaluator(timeline_resolution_seconds=30,start_time_threshold_seconds=start_time_threshold_seconds)
    
    if show_roc:
        results = evaluator.evaluate_with_dual_roc(predicted_file, ground_truth_file, aggregation='average')
        try:
            evaluator.plot_dual_roc_curves(results, save_path='roc_quick_test.png', activity_filter=['1', '2', '3', '4', '5', '6', '7', '8'])
        except Exception as e:
            print(f"Could not plot ROC curves: {e}")
    else:
        results = evaluator.evaluate(predicted_file, ground_truth_file)
        
    # ADD THE DEBUG CODE HERE (before print_report)
    print("\n=== Frame-Level Timeline Debug ===")
    pred_df = evaluator.parse_log_file(predicted_file)
    gt_df = evaluator.parse_log_file(ground_truth_file)

    pred_segments = evaluator.extract_segments(pred_df)
    gt_segments = evaluator.extract_segments(gt_df)

    all_times = []
    for seg in pred_segments + gt_segments:
        all_times.extend([seg['start_time'], seg['end_time']])

    start_time = min(all_times)
    end_time = max(all_times)

    # Create ground truth timeline
    gt_timeline, _ = evaluator.create_timeline(gt_segments, start_time, end_time)

    # Create frame-level prediction timeline
    pred_timeline, pred_conf_timeline = evaluator.create_true_frame_level_timeline_from_events(
        pred_df, start_time, end_time, aggregation='average'
    )

    print(f"Timeline length: {len(pred_timeline)}")
    print(f"Non-empty frames: {np.sum(pred_timeline != 'None')}")
    print(f"Frames with confidence > 0: {np.sum(pred_conf_timeline > 0)}")
    print(f"Mean confidence (non-zero): {np.mean(pred_conf_timeline[pred_conf_timeline > 0]):.3f}")
    print(f"Min confidence (non-zero): {np.min(pred_conf_timeline[pred_conf_timeline > 0]):.3f}")
    print(f"Max confidence: {np.max(pred_conf_timeline):.3f}")

    # Check for activity 1 specifically
    activity_1_gt_frames = np.sum(gt_timeline == '1')
    activity_1_pred_frames = np.sum(pred_timeline == '1')
    activity_1_conf_frames = np.sum((pred_timeline == '1') & (pred_conf_timeline > 0))

    print(f"\nActivity 1 frames:")
    print(f"  GT frames: {activity_1_gt_frames}")
    print(f"  Pred frames: {activity_1_pred_frames}")
    print(f"  Pred frames with conf>0: {activity_1_conf_frames}")
    
    evaluator.print_report(results, activity_filter=activity_filter)
    return results

if __name__ == "__main__":
    
    # Configuration
    GROUND_TRUTH_FILE = "../data/P001.txt"
    SANITY_CHECK_DIR = "../SanityCheck"
    
    # Define activity filter (only show activities 1-8)
    ACTIVITY_FILTER = ['1', '2', '3', '4', '5', '6', '7', '8']
    
    print("\n" + "="*70)
    print("SEGMENT EVALUATION TESTING SYSTEM")
    print("="*70)
    
    # Option 1: Create sanity check files first (if not already created)
    create_files = input("\nCreate sanity check files? (y/n): ").lower().strip()
    if create_files == 'y':
        print("\nCreating sanity check files...")
        modifier = SanityCheckModifier(seed=42)
        modifier.create_sanity_check_variants(GROUND_TRUTH_FILE, SANITY_CHECK_DIR)
        
    # Option 3: Ask about timeline visualisation
    timeline_visualisation_enabled = input("\nEnable segment timeline visualisation? (y/n): ").lower().strip() == 'y'
    
    show_roc = input("\nGenerate ROC curves? (y/n): ").lower().strip() == 'y'
        
    # Option 3: Ask about time tolerance
    #start_time_threshold_seconds = int(input("\nStart and end time tolerance in seconds: ").strip())
    
    # Option 4: Run full test suite
    print("\n" + "="*70)
    print("Choose test mode:")
    print("1. Run full sanity check test suite")
    print("2. Quick test single file")
    print("3. Test with confidence scores (SVM mode)")
    print("="*70)
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == '1':
        # Full test suite
        all_results = run_all_sanity_tests(
            GROUND_TRUTH_FILE, 
            SANITY_CHECK_DIR, 
            timeline_visualisation=timeline_visualisation_enabled,
            #start_time_threshold_seconds=start_time_threshold_seconds,
            show_roc=show_roc,
            activity_filter=ACTIVITY_FILTER
        )
        
    elif choice == '2':
        # Quick single file test
        predicted = input("Enter predicted file path: ").strip()
        quick_test(
            GROUND_TRUTH_FILE, 
            predicted, 
            #start_time_threshold_seconds=start_time_threshold_seconds,
            show_roc=show_roc,
            activity_filter=ACTIVITY_FILTER
        )
        
    elif choice == '3':
        # Test with confidence scores
        predicted = input("Enter predicted file path: ").strip()
        scores_input = input("Enter confidence scores (comma-separated, e.g., 0.95,0.87,0.92): ").strip()
        confidence_scores = [float(x.strip()) for x in scores_input.split(',')]
        test_with_confidence_scores(
            GROUND_TRUTH_FILE, 
            predicted, 
            confidence_scores, 
            #start_time_threshold_seconds=start_time_threshold_seconds
        )
    
    else:
        print("Invalid choice. Exiting.")
    
    print("\n" + "="*70)
    print("Testing complete!")
    print("="*70)