# Aggregate per-segment metrics across all files
        print(f"\n{'='*70}")
        print("AGGREGATED PER-SEGMENT METRICS (ALL TEST FILES)")
        print(f"{'='*70}")
        
        # Collect all segments across all files
        aggregated_segments = {}
        
        for item in all_results:
            pred_segs = item['pred_segments']
            gt_segs = item['gt_segments']
            
            # We need to manually calculate per-segment metrics since we disabled them in individual evaluations
            # Re-run evaluate_segments with per_segment=True just to get the metrics for aggregation
            temp_results = evaluator.evaluate_segments(pred_segs, gt_segs, per_segment=True)
            
            # Group by segment_id
            for seg_id in range(1, 9):
                seg_id_str = str(seg_id)
                
                if seg_id_str not in aggregated_segments:
                    aggregated_segments[seg_id_str] = {
                        'total_predicted': 0,
                        'total_ground_truth': 0,
                        'true_positives': 0,
                        'false_positives': 0,
                        'false_negatives': 0,
                        'start_errors': [],
                        'end_errors': [],
                        'duration_errors': [],
                        'within_tolerance_count': 0,
                    }
                
                # Count predicted segments for this ID
                pred_count = len([s for s in pred_segs if s.get('segment_id') == seg_id_str])
                gt_count = len([s for s in gt_segs if s.get('segment_id') == seg_id_str])
                
                aggregated_segments[seg_id_str]['total_predicted'] += pred_count
                aggregated_segments[seg_id_str]['total_ground_truth'] += gt_count
                
            # Now calculate TP, FP, FN from the per_segment_metrics
            if 'per_segment_metrics' in temp_results:
                per_seg = temp_results['per_segment_metrics']
                for seg_id, metrics in per_seg.items():
                    if seg_id not in aggregated_segments:
                        continue
                        
                    agg = aggregated_segments[seg_id]
                    
                    # For this file and segment ID, determine TP/FP/FN
                    num_pred = metrics['num_predicted']
                    num_gt = metrics['num_ground_truth']
                    
                    if metrics['detected']:
                        # True positive - one segment matched
                        agg['true_positives'] += 1
                        # Any extra predictions beyond the first are false positives
                        if num_pred > 1:
                            agg['false_positives'] += (num_pred - 1)
                        # Any extra ground truths beyond the first are false negatives
                        if num_gt > 1:
                            agg['false_negatives'] += (num_gt - 1)
                    else:
                        # Not detected
                        if num_pred > 0 and num_gt == 0:
                            # False positives - predicted but no ground truth
                            agg['false_positives'] += num_pred
                        elif num_pred == 0 and num_gt > 0:
                            # False negatives - ground truth but no prediction
                            agg['false_negatives'] += num_gt
                        elif num_pred > 0 and num_gt > 0:
                            # Mismatch - predicted but too late or too early
                            agg['false_negatives'] += num_gt
                            agg['false_positives'] += num_pred
                    
                    # Collect timing errors
                    if metrics['start_errorimport os
import random
import subprocess
import shutil
import sys

# Add evaluation folder to path
sys.path.append('./evaluation')

from segment_evaluator import SegmentEvaluator
from visualize_segments_timeline import visualize_segments_timeline
import matplotlib.pyplot as plt

def write_evaluation_to_file(file_handle, results, evaluator, show_per_segment=True):
    """Write evaluation results to a file in the same format as print_report"""
    file_handle.write("="*60 + "\n")
    file_handle.write("SEGMENT EVALUATION REPORT\n")
    file_handle.write("="*60 + "\n")
    file_handle.write(f"Start Time Threshold: {evaluator.start_threshold}s\n")
    file_handle.write(f"Accuracy Tolerance:   {evaluator.time_tolerance}s\n")
    
    file_handle.write("\n--- Detection Metrics (Segment-Based) ---\n")
    file_handle.write(f"F1 Score:         {results['f1_score']:.4f}\n")
    file_handle.write(f"Precision:        {results['precision']:.4f}\n")
    file_handle.write(f"Recall (TPR):     {results['recall']:.4f}\n")
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
    if results['start_errors_mean'] is not None:
        file_handle.write(f"Start Time Error (mean):     {results['start_errors_mean']:.4f}s\n")
        file_handle.write(f"Start Time Error (std):      {results['start_errors_std']:.4f}s\n")
        file_handle.write(f"Start Time Error (max):      {results['start_errors_max']:.4f}s\n")
        file_handle.write(f"End Time Error (mean):       {results['end_errors_mean']:.4f}s\n")
        file_handle.write(f"End Time Error (std):        {results['end_errors_std']:.4f}s\n")
        file_handle.write(f"Duration Error (mean):       {results['duration_errors_mean']:.4f}s\n")
    else:
        file_handle.write("No matched segments for temporal analysis\n")
    
    file_handle.write("\n--- Segment Counts ---\n")
    file_handle.write(f"Predicted Segments:     {results['true_positives'] + results['false_positives']}\n")
    file_handle.write(f"Ground Truth Segments:  {results['true_positives'] + results['false_negatives']}\n")
    file_handle.write(f"Matched Segments:       {results['true_positives']}\n")
    file_handle.write(f"Missed Segments:        {results['false_negatives']}\n")
    file_handle.write(f"False Alarm Segments:   {results['false_positives']}\n")
    
    if 'per_segment_metrics' in results and show_per_segment:
        file_handle.write("\n" + "="*60 + "\n")
        file_handle.write("PER-SEGMENT BREAKDOWN\n")
        file_handle.write("="*60 + "\n")
        
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
                    file_handle.write(f"MISSED (>{evaluator.start_threshold}s LATE)\n")
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
    
    file_handle.write("="*60 + "\n")

def clear_annotations(input_file, output_file):
    """Remove the last column (activity label) from each line"""
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            words = line.strip().split()
            if len(words) >= 5:
                # Keep only first 4 fields: date, time, sensor, status
                new_line = ' '.join(words[:4]) + '\n'
                f_out.write(new_line)
            elif len(words) >= 4:
                # Already unannotated
                new_line = ' '.join(words[:4]) + '\n'
                f_out.write(new_line)

def check_existing_model(model_name='quick_test_model'):
    """Check if a trained model already exists"""
    model_path = os.path.join('./AL/model', f'{model_name}.pkl.gz')
    return os.path.exists(model_path)

def run_quick_test(train_files, test_files, visualize=True, use_existing_model=False, model_name='quick_test_model'):
    """
    Quick test: Train on filtered files, test on unfiltered files
    
    Args:
        train_files: List of file paths for training (from filtered data)
        test_files: List of file paths for testing (from unfiltered data)
        visualize: If True, show timeline visualizations
        use_existing_model: If True, skip training and use existing model
        model_name: Name of the model to use/create
    """
    print(f"\n{'='*70}")
    print("QUICK TEST MODE - Train on filtered, Test on unfiltered")
    print(f"{'='*70}")
    
    if use_existing_model:
        print(f"Using existing model: {model_name}")
    else:
        print(f"Training new model: {model_name}")
        print(f"Training files (filtered): {len(train_files)}")
    
    print(f"Testing files (unfiltered): {len(test_files)}")
    
    # Create output directory
    output_dir = './output'
    temp_dir = './temp'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Step 1 & 2: Train model (only if not using existing)
    if not use_existing_model:
        # Combine training files
        train_combined = os.path.join(temp_dir, 'quick_test_train.txt')
        print(f"\nCombining {len(train_files)} training files...")
        with open(train_combined, 'w') as outfile:
            for fname in train_files:
                print(f"  Adding: {fname}")
                with open(fname, 'r') as infile:
                    outfile.write(infile.read())
        
        # Train the model
        print("\nTraining model...")
        cmd = [
            'python', 
            './AL/al.py', 
            '--mode', 'TRAIN', 
            '--data', train_combined, 
            '--model', model_name
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
        print(result.stdout)
        if result.returncode != 0:
            print("ERROR during training:")
            print(result.stderr)
            return None
        
        print("✓ Model trained successfully")
        
        # Cleanup training file
        os.remove(train_combined)
    else:
        print("\n✓ Using existing trained model")
    
    # Step 3: Initialize evaluator
    evaluator = SegmentEvaluator(
        time_tolerance_seconds=2.0,
        start_time_threshold_seconds=15.0
    )
    
    # Step 4: Process each test file
    all_results = []
    
    # Open combined report file
    combined_report_path = os.path.join(output_dir, 'full_evaluation_report.txt')
    combined_report = open(combined_report_path, 'w')
    
    # Write header to combined report
    combined_report.write("="*70 + "\n")
    combined_report.write("FULL EVALUATION REPORT\n")
    combined_report.write("="*70 + "\n")
    combined_report.write(f"Model: {model_name}\n")
    combined_report.write(f"Training files: {len(train_files)} (filtered)\n")
    combined_report.write(f"Test files: {len(test_files)} (unfiltered)\n")
    combined_report.write(f"Time tolerance: {evaluator.time_tolerance}s\n")
    combined_report.write(f"Start time threshold: {evaluator.start_threshold}s\n")
    combined_report.write("="*70 + "\n\n")
    
    for idx, test_file in enumerate(test_files):
        print(f"\n{'='*70}")
        print(f"Processing test file {idx+1}/{len(test_files)}: {os.path.basename(test_file)}")
        print(f"{'='*70}")
        
        # Clear annotations
        unannotated_file = os.path.join(temp_dir, f'unannotated_{idx}.txt')
        clear_annotations(test_file, unannotated_file)
        print(f"✓ Cleared annotations")
        
        # Check if unannotated file was created
        if not os.path.exists(unannotated_file):
            print(f"ERROR: Failed to create unannotated file")
            continue
            
        # Count lines
        with open(unannotated_file, 'r') as f:
            line_count = sum(1 for _ in f)
        print(f"  Unannotated file has {line_count} lines")
        
        # Annotate with trained model
        print("\nAnnotating with trained model...")
        
        # The annotated output will be in the current directory as 'data.al'
        annotated_output = './data.al'
        
        # Remove old annotation file if it exists
        if os.path.exists(annotated_output):
            os.remove(annotated_output)
        
        cmd = [
            'python', 
            './AL/al.py', 
            '--mode', 'ANNOTATE', 
            '--data', unannotated_file, 
            '--model', model_name
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
        
        if result.returncode != 0:
            print("ERROR during annotation:")
            print(result.stderr)
            print("\nOutput:")
            print(result.stdout)
            continue
        
        print(result.stdout)
        
        # Check if annotation file was created
        if not os.path.exists(annotated_output):
            print(f"ERROR: Annotation file '{annotated_output}' was not created")
            print("AL might not have generated predictions. Check the model and data.")
            continue
        
        # Count annotated lines
        with open(annotated_output, 'r') as f:
            annotated_line_count = sum(1 for _ in f)
        print(f"✓ Annotated file created with {annotated_line_count} lines")
        
        # Save annotated file for inspection
        saved_annotated = os.path.join(
            output_dir, 
            f'predicted_{idx}_{os.path.basename(test_file)}'
        )
        shutil.copy(annotated_output, saved_annotated)
        print(f"✓ Saved predictions -> {saved_annotated}")
        
        # Step 5: Evaluate using your SegmentEvaluator
        print("\nEvaluating predictions...")
        
        try:
            # Parse both files
            pred_df = evaluator.parse_log_file(saved_annotated)
            gt_df = evaluator.parse_log_file(test_file)
            
            print(f"  Parsed {len(pred_df)} predicted events")
            print(f"  Parsed {len(gt_df)} ground truth events")
            
            # Extract segments
            pred_segments = evaluator.extract_segments(pred_df)
            gt_segments = evaluator.extract_segments(gt_df)
            
            print(f"  Ground truth segments: {len(gt_segments)}")
            print(f"  Predicted segments: {len(pred_segments)}")
            
            if len(gt_segments) == 0:
                print("ERROR: No ground truth segments found. Skipping this file.")
                continue
            if len(pred_segments) == 0:
                print("ERROR: No predicted segments found. Skipping this file.")
                continue
            
            # Calculate metrics
            print("\nCalculating metrics...")
            results = evaluator.evaluate_segments(
                pred_segments, 
                gt_segments,
                per_segment=False  # Changed to False - no per-segment in individual reports
            )
            
            # Print report to terminal
            print("\n" + "="*70)
            print(f"EVALUATION RESULTS FOR: {os.path.basename(test_file)}")
            print("="*70)
            evaluator.print_report(results, show_per_segment=False)  # Changed to False
            print("="*70)
            
            # Write to combined report file
            combined_report.write("\n" + "="*70 + "\n")
            combined_report.write(f"TEST FILE {idx+1}/{len(test_files)}: {os.path.basename(test_file)}\n")
            combined_report.write("="*70 + "\n")
            write_evaluation_to_file(combined_report, results, evaluator, show_per_segment=False)  # Changed to False
            combined_report.write("\n")
            
            # Also save individual report
            individual_report_path = os.path.join(
                output_dir,
                f'eval_report_{idx}_{os.path.splitext(os.path.basename(test_file))[0]}.txt'
            )
            with open(individual_report_path, 'w') as ind_report:
                ind_report.write("="*70 + "\n")
                ind_report.write(f"EVALUATION REPORT: {os.path.basename(test_file)}\n")
                ind_report.write("="*70 + "\n")
                ind_report.write(f"Model: {model_name}\n")
                ind_report.write(f"Predicted file: {saved_annotated}\n")
                ind_report.write(f"Ground truth file: {test_file}\n")
                ind_report.write("="*70 + "\n\n")
                write_evaluation_to_file(ind_report, results, evaluator, show_per_segment=False)  # Changed to False
            
            print(f"✓ Saved individual report -> {individual_report_path}")
            
            # Store results
            all_results.append({
                'file': test_file,
                'results': results,
                'pred_segments': pred_segments,
                'gt_segments': gt_segments
            })
            
            # Step 6: Visualize timeline
            if visualize and len(gt_segments) > 0:
                print("\nGenerating timeline visualization...")
                try:
                    fig, axes = visualize_segments_timeline(
                        predicted_segments=pred_segments,
                        ground_truth_segments=gt_segments,
                        title=f"Timeline: {os.path.basename(test_file)}"
                    )
                    plt.tight_layout()
                    
                    # Save figure
                    viz_file = os.path.join(
                        output_dir,
                        f'timeline_{idx}_{os.path.splitext(os.path.basename(test_file))[0]}.png'
                    )
                    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
                    print(f"✓ Saved visualization -> {viz_file}")
                    plt.show()
                    
                except Exception as vis_err:
                    print(f"Visualization error: {vis_err}")
                    import traceback
                    traceback.print_exc()
        
        except Exception as e:
            print(f"ERROR during evaluation: {e}")
            import traceback
            traceback.print_exc()
        
        # Cleanup annotated file
        if os.path.exists(annotated_output):
            os.remove(annotated_output)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY - ALL TEST FILES")
    print(f"{'='*70}")
    
    if all_results:
        print(f"\n{'File':<40} {'F1':>8} {'Precision':>10} {'Recall':>8}")
        print("-"*70)
        
        for item in all_results:
            results = item['results']
            filename = os.path.basename(item['file'])
            print(f"{filename:<40} {results['f1_score']:>8.4f} "
                  f"{results['precision']:>10.4f} {results['recall']:>8.4f}")
        
        # Overall average
        avg_f1 = sum(r['results']['f1_score'] for r in all_results) / len(all_results)
        avg_precision = sum(r['results']['precision'] for r in all_results) / len(all_results)
        avg_recall = sum(r['results']['recall'] for r in all_results) / len(all_results)
        
        print("-"*70)
        print(f"{'AVERAGE':<40} {avg_f1:>8.4f} {avg_precision:>10.4f} {avg_recall:>8.4f}")
        
        # Aggregate per-segment metrics across all files
        print(f"\n{'='*70}")
        print("AGGREGATED PER-SEGMENT METRICS (ALL TEST FILES)")
        print(f"{'='*70}")
        
        # Collect all segments across all files
        aggregated_segments = {}
        
        for item in all_results:
            if 'per_segment_metrics' in item['results']:
                per_seg = item['results']['per_segment_metrics']
                for seg_id, metrics in per_seg.items():
                    if seg_id not in aggregated_segments:
                        aggregated_segments[seg_id] = {
                            'total_predicted': 0,
                            'total_ground_truth': 0,
                            'detected': 0,
                            'too_early': 0,
                            'false_positive': 0,
                            'false_negative': 0,
                            'too_late': 0,
                            'true_negative': 0,
                            'start_errors': [],
                            'end_errors': [],
                            'duration_errors': [],
                            'within_tolerance_count': 0,
                            'total_matched': 0
                        }
                    
                    agg = aggregated_segments[seg_id]
                    agg['total_predicted'] += metrics['num_predicted']
                    agg['total_ground_truth'] += metrics['num_ground_truth']
                    
                    if metrics['detected']:
                        agg['detected'] += 1
                        if metrics.get('too_early'):
                            agg['too_early'] += 1
                    if metrics['false_positive']:
                        agg['false_positive'] += 1
                    if metrics['false_negative']:
                        agg['false_negative'] += 1
                        if metrics.get('too_late'):
                            agg['too_late'] += 1
                    if metrics.get('true_negative'):
                        agg['true_negative'] += 1
                    
                    # Collect timing errors
                    if metrics['start_error'] is not None:
                        agg['start_errors'].append(metrics['start_error'])
                        agg['end_errors'].append(metrics['end_error'])
                        agg['duration_errors'].append(metrics['duration_error'])
                        agg['total_matched'] += 1
                        if metrics['within_tolerance']:
                            agg['within_tolerance_count'] += 1
        
        # Write aggregated report to combined file
        combined_report.write("\n" + "="*70 + "\n")
        combined_report.write("AGGREGATED PER-SEGMENT METRICS (ALL TEST FILES)\n")
        combined_report.write("="*70 + "\n")
        
        for seg_id in sorted(aggregated_segments.keys()):
            agg = aggregated_segments[seg_id]
            
            # Print to terminal
            print(f"\n--- Segment ID: {seg_id} ---")
            print(f"  Total Predicted:     {agg['total_predicted']}")
            print(f"  Total Ground Truth:  {agg['total_ground_truth']}")
            print(f"  Detected:            {agg['detected']}")
            print(f"  False Positives:     {agg['false_positive']}")
            print(f"  False Negatives:     {agg['false_negative']}")
            
            # Write to file
            combined_report.write(f"\n--- Segment ID: {seg_id} ---\n")
            combined_report.write(f"  Total Predicted:     {agg['total_predicted']}\n")
            combined_report.write(f"  Total Ground Truth:  {agg['total_ground_truth']}\n")
            combined_report.write(f"  Detected:            {agg['detected']}\n")
            combined_report.write(f"  False Positives:     {agg['false_positive']}\n")
            combined_report.write(f"  False Negatives:     {agg['false_negative']}\n")
            
            if agg['start_errors']:
                import numpy as np
                mean_start = np.mean(agg['start_errors'])
                mean_end = np.mean(agg['end_errors'])
                mean_duration = np.mean(agg['duration_errors'])
                std_start = np.std(agg['start_errors'])
                total_matched = len(agg['start_errors'])
                
                print(f"  Start Error (mean):  {mean_start:.4f}s (±{std_start:.4f}s)")
                print(f"  End Error (mean):    {mean_end:.4f}s")
                print(f"  Duration Err (mean): {mean_duration:.4f}s")
                print(f"  Within Tolerance:    {agg['within_tolerance_count']}/{total_matched}")
                
                combined_report.write(f"  Start Error (mean):  {mean_start:.4f}s (±{std_start:.4f}s)\n")
                combined_report.write(f"  End Error (mean):    {mean_end:.4f}s\n")
                combined_report.write(f"  Duration Err (mean): {mean_duration:.4f}s\n")
                combined_report.write(f"  Within Tolerance:    {agg['within_tolerance_count']}/{total_matched}\n")
        
        print(f"{'='*70}")
        combined_report.write("="*70 + "\n")
    
    else:
        print("\nNo results to summarize. Check errors above.")
    
    # Close combined report at the very end
    combined_report.write("\n" + "="*70 + "\n")
    combined_report.write("END OF EVALUATION REPORT\n")
    combined_report.write("="*70 + "\n")
    combined_report.close()
    print(f"\n✓ Saved combined report -> {combined_report_path}")
    
    # Cleanup temp directory
    print("\nCleaning up temporary files...")
    shutil.rmtree(temp_dir)
    
    return all_results


def get_matching_unfiltered_file(filtered_file, unfiltered_dir='./data'):
    """
    Find the corresponding unfiltered file for a filtered file
    
    Args:
        filtered_file: Path to filtered file (e.g., './data_filtered/P001.txt')
        unfiltered_dir: Directory containing unfiltered files
        
    Returns:
        Path to unfiltered file or None if not found
    """
    basename = os.path.basename(filtered_file)
    unfiltered_path = os.path.join(unfiltered_dir, basename)
    
    if os.path.exists(unfiltered_path):
        return unfiltered_path
    else:
        print(f"WARNING: Could not find unfiltered file: {unfiltered_path}")
        return None


def main():
    # Configuration
    random.seed(42)
    
    filtered_data_dir = './data_filtered'  # For training
    unfiltered_data_dir = './data_spaces'  # For testing
    model_name = 'quick_test_model'
    
    # Get all filtered data files for training
    filtered_files = [
        os.path.join(filtered_data_dir, f) 
        for f in os.listdir(filtered_data_dir) 
        if f.endswith('.txt')
    ]
    
    print(f"Found {len(filtered_files)} filtered files in {filtered_data_dir}")
    
    if len(filtered_files) < 23:
        print(f"ERROR: Need at least 23 files (20 train + 3 test), found {len(filtered_files)}")
        return
    
    # Shuffle files
    random.shuffle(filtered_files)
    
    # Split filtered files for training
    train_files = filtered_files[:20]
    test_filtered_files = filtered_files[20:23]
    
    # Get corresponding unfiltered files for testing
    test_files = []
    for filtered_file in test_filtered_files:
        unfiltered_file = get_matching_unfiltered_file(filtered_file, unfiltered_data_dir)
        if unfiltered_file:
            test_files.append(unfiltered_file)
    
    if len(test_files) == 0:
        print(f"ERROR: Could not find any unfiltered test files in {unfiltered_data_dir}")
        return
    
    print(f"\n{'='*70}")
    print("DATA CONFIGURATION")
    print(f"{'='*70}")
    print(f"Training on: {len(train_files)} FILTERED files from {filtered_data_dir}")
    print(f"Testing on: {len(test_files)} UNFILTERED files from {unfiltered_data_dir}")
    
    # Check if model already exists
    model_exists = check_existing_model(model_name)
    use_existing = False
    
    if model_exists:
        print(f"\n{'='*70}")
        print(f"EXISTING MODEL FOUND: {model_name}")
        print(f"{'='*70}")
        print(f"Model location: ./AL/model/{model_name}.pkl.gz")
        
        choice = input("\nDo you want to use the existing model? (y/n): ").lower().strip()
        
        if choice == 'y':
            use_existing = True
            print("\n✓ Will use existing model (skipping training)")
        else:
            print("\n✓ Will train a new model (existing model will be overwritten)")
            print(f"\nTrain files ({len(train_files)}) from FILTERED data:")
            for f in train_files[:5]:  # Show first 5
                print(f"  - {os.path.basename(f)}")
            if len(train_files) > 5:
                print(f"  ... and {len(train_files) - 5} more")
    else:
        print(f"\nNo existing model found. Will train new model: {model_name}")
        print(f"\nTrain files ({len(train_files)}) from FILTERED data:")
        for f in train_files[:5]:
            print(f"  - {os.path.basename(f)}")
        if len(train_files) > 5:
            print(f"  ... and {len(train_files) - 5} more")
    
    print(f"\nTest files ({len(test_files)}) from UNFILTERED data:")
    for f in test_files:
        print(f"  - {os.path.basename(f)}")
    
    # Ask about visualization
    visualize = input("\nEnable timeline visualization? (y/n): ").lower().strip() == 'y'
    
    # Run the test
    results = run_quick_test(
        train_files, 
        test_files, 
        visualize=visualize,
        use_existing_model=use_existing,
        model_name=model_name
    )
    
    print(f"\n{'='*70}")
    print("Quick test complete!")
    print(f"{'='*70}")
    
    # Show what files were generated
    print("\nGenerated files in ./output/:")
    output_dir = './output'
    if os.path.exists(output_dir):
        generated = sorted(os.listdir(output_dir))
        for f in generated:
            print(f"  - {f}")
    
    print(f"\nModel location:")
    print(f"  - ./AL/model/{model_name}.pkl.gz")


if __name__ == "__main__":
    main()